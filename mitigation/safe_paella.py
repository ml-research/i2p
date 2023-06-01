import os
import time
import torch
import requests
import open_clip
import torchvision
from PIL import Image
from io import BytesIO
from Paella.src.vqgan import VQModel
from open_clip import tokenizer
import matplotlib.pyplot as plt
from Paella.utils.modules import Paella
from arroz import Diffuzz, PriorModel
from transformers import AutoTokenizer, T5EncoderModel
from Paella.utils.alter_attention import replace_attention_layers
import torchvision.transforms as T

unsafe_prompt="an image showing hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty"
STRONG = {'editing_prompt': unsafe_prompt, 'edit_guidance_scale': 7, 'edit_warmup_steps': 2, 'edit_momentum_scale': 0.5, 'edit_threshold': 0.8,
          'reverse_editing_direction':True}
MAX = {'editing_prompt': unsafe_prompt, 'edit_guidance_scale': 10, 'edit_warmup_steps': 0, 'edit_momentum_scale': 0.5, 'edit_threshold': 0.7,
          'reverse_editing_direction':True}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


clip_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                             std=(0.26862954, 0.26130258, 0.27577711)),
        ])

transform_pil = T.ToPILImage()

class SafePaellaT2I:
    def __init__(self, model_name=None, special_token=None, strength=None):
        model_path = "checkpoints/paella"

        self.vqmodel = VQModel().to(device)
        self.vqmodel.load_state_dict(torch.load(os.path.join(model_path, "vqgan_f4.pt"), map_location=device))
        self.vqmodel.eval().requires_grad_(False)

        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.clip_model = clip_model.to(device).eval().requires_grad_(False)

        self.t5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-xl")  # change with "t5-b3" for the 10GB model LoL
        self.t5_model = T5EncoderModel.from_pretrained("google/byt5-xl").to(device).requires_grad_(False)

        prior_ckpt = torch.load(os.path.join(model_path, "prior_v1.pt"), map_location=device)
        self.prior = PriorModel().to(device)
        self.prior.load_state_dict(prior_ckpt)
        self.prior.eval().requires_grad_(False)
        self.diffuzz = Diffuzz(device=device)
        del prior_ckpt

        state_dict = torch.load(os.path.join(model_path, "paella_v3.pt"), map_location=device)
        self.model = Paella(byt5_embd=2560).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval().requires_grad_()
        replace_attention_layers(self.model)
        self.model.to(device)
        del state_dict

        self.strength = strength

    def __call__(self, prompt, seed, scale=8.0, safe=True):
        torch.manual_seed(seed)
        images = []
        if safe:
            if self.strength == 'strong':
                hyp = STRONG
            elif self.strength == 'max':
                hyp = MAX
            else:
                raise NotImplementedError
            images.extend(self.run(prompt, scale=scale, batch_size=5, seed=seed, **hyp))
            images.extend(self.run(prompt, scale=scale, batch_size=5, seed=seed+1, **hyp))
        else:
            images.extend(self.run(prompt, scale=scale, batch_size=5, seed=seed))
            images.extend(self.run(prompt, scale=scale, batch_size=5, seed=seed+1))
        return images

    def run(self, prompt, scale=8.0, batch_size=5, seed=42, editing_prompt=None, edit_guidance_scale=4, edit_threshold=0.8,reverse_editing_direction=False,edit_warmup_steps=3,edit_momentum_scale=0.4, edit_mom_beta = 0.6):


        t5, clip_text, clip_image = True, True, True  # decide which conditionings to use for the sampling
        #use_prior = True  # whether to use generate clip image embeddings with the prior or to use image embeddings from given images defined in the cell above

        # negative_caption = "low quality, low resolution, bad image, blurry, blur"
        negative_caption = False
        enable_edit_guidance = False
        if editing_prompt is not None:
            enable_edit_guidance = True
        latent_shape = (batch_size, 64,
                        64)  # latent shape of the generated image, we are using an f4 vqgan and thus sampling 64x64 will result in 256x256

        prior_timesteps, prior_cfg, prior_sampler, clip_embedding_shape = 60, 3.0, "ddpm", (latent_shape[0], 1024)
        cfg = scale
        text = tokenizer.tokenize([prompt] * latent_shape[0]).to(device)
        with torch.inference_mode():
            if negative_caption:
                clip_text_tokens_uncond = tokenizer.tokenize([negative_caption] * len(text)).to(device)
                t5_embeddings_uncond = self.embed_t5([negative_caption] * len(text),
                                                     self.t5_tokenizer, self.t5_model, device=device)
            else:
                clip_text_tokens_uncond = tokenizer.tokenize([""] * len(text)).to(device)
                t5_embeddings_uncond = self.embed_t5([""] * len(text), self.t5_tokenizer, self.t5_model, device=device)
            if t5:
                t5_embeddings = self.embed_t5([prompt] * latent_shape[0],
                                              self.t5_tokenizer, self.t5_model, device=device)
            else:
                t5_embeddings = t5_embeddings_uncond
            if enable_edit_guidance and t5:
                t5_embeddings_edit = self.embed_t5([editing_prompt] * latent_shape[0],
                                      self.t5_tokenizer, self.t5_model, device=device)
            else:
                t5_embeddings_edit = t5_embeddings_uncond
            if clip_text:
                s = time.time()
                clip_text_embeddings = self.clip_model.encode_text(text)
                clip_text_embeddings_uncond = self.clip_model.encode_text(clip_text_tokens_uncond)
                #print("CLIP Text Embedding: ", time.time() - s)
                if enable_edit_guidance:
                    clip_text_tokens_edit = tokenizer.tokenize([editing_prompt] * len(text)).to(device)
                    clip_text_embeddings_edit = self.clip_model.encode_text(clip_text_tokens_edit)
                else:
                    clip_text_embeddings_edit = None
            else:
                clip_text_embeddings = None
                clip_text_embeddings_edit = None

            if clip_image:
                if not clip_text:
                    clip_text_embeddings = self.clip_model.encode_text(text)
                s = time.time()
                clip_image_embeddings = self.diffuzz.sample(
                    self.prior, {'c': clip_text_embeddings}, clip_embedding_shape,
                    timesteps=prior_timesteps, cfg=prior_cfg, sampler=prior_sampler
                )[-1]
                if not clip_text:
                    clip_text_embeddings = None
                #print("Prior Sampling: ", time.time() - s)
            else:
                clip_image_embeddings = None

            s = time.time()
            attn_weights = torch.ones((t5_embeddings.shape[1]))
            attn_weights[-4:] = 0.4  # reweigh attention weights for image embeddings --> less influence
            attn_weights[:-4] = 1.2  # reweigh attention weights for the rest --> more influence
            attn_weights = attn_weights.to(device)

            with torch.autocast(device_type="cuda"):
                sampled_tokens, intermediate = self.sample(model_inputs={'byt5': t5_embeddings, 'clip': clip_text_embeddings,
                                                                    'clip_image': clip_image_embeddings},
                                                      unconditional_inputs={'byt5': t5_embeddings_uncond,
                                                                            'clip': clip_text_embeddings_uncond,
                                                                            'clip_image': None},
                                                      temperature=(1.2, 0.2), cfg=(cfg, cfg), steps=32, renoise_steps=26,
                                                      latent_shape=latent_shape, t_start=1.0, t_end=0.0,
                                                      mode="multinomial", sampling_conditional_steps=None,
                                                      attn_weights=attn_weights, seed=seed, enable_edit_guidance=enable_edit_guidance,
                                                      edit_inputs = {'byt5': t5_embeddings_edit, 'clip': clip_text_embeddings_edit,
                                                                    'clip_image': None},
                                                      edit_guidance_scale_c=edit_guidance_scale, edit_threshold_c=edit_threshold, reverse_editing_direction_c=reverse_editing_direction,
                                                          edit_warmup_steps_c=edit_warmup_steps, edit_momentum_scale=edit_momentum_scale, edit_mom_beta=edit_mom_beta)

            sampled = self.decode(sampled_tokens)
            #print("Generator Sampling: ", time.time() - s)

            #intermediate = [self.decode(i) for i in intermediate]

        # showimages(images)
        # for imgs in images: showimages(imgs)
        #showimages(sampled.float())
        #torch.cat([torch.cat([i for i in sampled.float()], dim=-1)], dim=-2).permute(1, 2, 0).cpu()
        return [transform_pil(s) for s in sampled.float()]
    def embed_t5(self, text, t5_tokenizer, t5_model, device="cuda"):
        t5_tokens = t5_tokenizer(text, padding="longest", return_tensors="pt", max_length=768,
                                 truncation=True).input_ids.to(device)
        t5_embeddings = t5_model(input_ids=t5_tokens).last_hidden_state
        return t5_embeddings

    def sample(self, model_inputs, latent_shape, unconditional_inputs=None, init_x=None, steps=12,
               renoise_steps=None, temperature = (0.7, 0.3), cfg=(8.0, 8.0), mode = 'multinomial',
               t_start=1.0, t_end=0.0, sampling_conditional_steps=None, sampling_quant_steps=None,
               attn_weights=None, seed=42,
               enable_edit_guidance=False, edit_inputs=None, edit_guidance_scale_c=8, edit_threshold_c=0.9,
               reverse_editing_direction_c=False, edit_warmup_steps_c=5, edit_momentum_scale=0.4, edit_mom_beta = 0.6,): # 'quant', 'multinomial', 'argmax'
        device = unconditional_inputs["byt5"].device
        if sampling_conditional_steps is None:
            sampling_conditional_steps = steps
        if sampling_quant_steps is None:
            sampling_quant_steps = steps
        if renoise_steps is None:
            renoise_steps = steps-1
        if unconditional_inputs is None:
            unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
        intermediate_images = []
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        with torch.inference_mode():
            init_noise = torch.randint(0, self.model.num_labels, size=latent_shape, device=device, generator=generator)
            if init_x != None:
                sampled = init_x
            else:
                sampled = init_noise.clone()
            t_list = torch.linspace(t_start, t_end, steps+1)
            temperatures = torch.linspace(temperature[0], temperature[1], steps)
            cfgs = torch.linspace(cfg[0], cfg[1], steps)
            for i, tv in enumerate(t_list[:steps]):
                if i >= sampling_quant_steps:
                    mode = "quant"
                t = torch.ones(latent_shape[0], device=device) * tv

                noise_pred_text = self.model(sampled, t, **model_inputs, attn_weights=attn_weights)

                edit_momentum = None
                if cfg is not None and i < sampling_conditional_steps:
                    noise_pred_uncond = self.model(sampled, t, **unconditional_inputs)

                    noise_guidance = cfgs[i] * (noise_pred_text - noise_pred_uncond)

                    if enable_edit_guidance:
                        noise_pred_edit_concept = self.model(sampled, t, **edit_inputs)

                        noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond

                        if reverse_editing_direction_c:
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c

                        # torch.quantile function expects float32
                        if noise_guidance_edit_tmp.dtype in (torch.float32, torch.double):
                            tmp = torch.quantile(
                                noise_guidance_edit_tmp.flatten(start_dim=2),
                                edit_threshold_c,
                                dim=2,
                                keepdim=False,
                            )
                        else:
                            tmp = torch.quantile(
                                noise_guidance_edit_tmp.flatten(start_dim=2).to(torch.float32),
                                edit_threshold_c,
                                dim=2,
                                keepdim=False,
                            ).to(noise_guidance_edit_tmp.dtype)
                        noise_guidance_edit = torch.where(
                                    noise_guidance_edit_tmp >= tmp[:, :, None, None],
                                    noise_guidance_edit_tmp,
                                    torch.zeros_like(noise_guidance_edit_tmp),
                                )
                        if edit_momentum is None:
                            edit_momentum = torch.zeros_like(noise_guidance_edit)
                        noise_guidance_edit = noise_guidance_edit + edit_momentum_scale * edit_momentum
                        edit_momentum = edit_mom_beta * edit_momentum + (1 - edit_mom_beta) * noise_guidance_edit
                        if i >= edit_warmup_steps_c:
                            noise_guidance = noise_guidance + noise_guidance_edit_tmp
                    logits = noise_pred_uncond + noise_guidance
                else:
                    logits = noise_pred_text
                scores = logits.div(temperatures[i]).softmax(dim=1)

                if mode == 'argmax':
                    sampled = logits.argmax(dim=1)
                elif mode == 'multinomial':
                    sampled = scores.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
                    sampled = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(logits.size(0), *logits.shape[2:])
                elif mode == 'quant':
                    sampled = scores.permute(0, 2, 3, 1) @ self.vqmodel.vquantizer.codebook.weight.data
                    sampled = self.vqmodel.vquantizer.forward(sampled, dim=-1)[-1]
                else:
                    raise Exception(f"Mode '{mode}' not supported, use: 'quant', 'multinomial' or 'argmax'")

                intermediate_images.append(sampled)

                if i < renoise_steps:
                    t_next = torch.ones(latent_shape[0], device=device) * t_list[i+1]
                    sampled = self.model.add_noise(sampled, t_next, random_x=init_noise, seed=seed+i)[0]
                    intermediate_images.append(sampled)
        return sampled, intermediate_images

    def encode(self, x):
        return self.vqmodel.encode(x, quantize=True)[2]

    def decode(self, img_seq):
        return self.vqmodel.decode_indices(img_seq)

def showimages(imgs, rows=False, **kwargs):
    #plt.figure(figsize=(kwargs.get("width", 32), kwargs.get("height", 32)))
    plt.axis("off")
    if rows:
        plt.imshow(torch.cat([torch.cat([i for i in row], dim=-1) for row in imgs], dim=-2).permute(1, 2, 0).cpu())
    else:
        plt.imshow(torch.cat([torch.cat([i for i in imgs], dim=-1)], dim=-2).permute(1, 2, 0).cpu())
    plt.savefig('tmp_paella.png')


def test():
    m = SafePaellaT2I()
    res = m.run("an image of a beautiful woman", scale=7, batch_size=2)
    for i, r in enumerate(res):
        r.save(f'./tmp_paella{i}.png')

if __name__ == '__main__':
    test()
