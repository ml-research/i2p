from mitigation.SemanticIF import SemanticIFPipeline
from diffusers.utils import pt_to_pil
import torch
from diffusers import DiffusionPipeline

unsafe_prompt = "an image showing hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty"
STRONG = {'editing_prompt': unsafe_prompt, 'edit_guidance_scale': 7, 'edit_warmup_steps': 5, 'edit_momentum_scale': 0.5,
          'edit_threshold': 0.8,
          'reverse_editing_direction': True}
MAX = {'editing_prompt': unsafe_prompt, 'edit_guidance_scale': 10, 'edit_warmup_steps': 0, 'edit_momentum_scale': 0.5,
       'edit_threshold': 0.7,
       'reverse_editing_direction': True}


class SafeIfT2I:
    def __init__(self, model_name=None, special_token=None, strength=None):

        # stage 1
        self.stage_1 = SemanticIFPipeline.from_pretrained("/checkpoints/DeepFloyd-IF/IF-I-XL-v1.0", variant="fp16",
                                                          torch_dtype=torch.float16)
        self.stage_1.enable_model_cpu_offload()
        print('Stage 1 loading done')
        # stage 2
        self.stage_2 = DiffusionPipeline.from_pretrained(
            "/checkpoints/DeepFloyd-IF/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        )
        self.stage_2.enable_model_cpu_offload()
        print('Stage 2 loading done')

        # stage 3
        safety_modules = {
            "feature_extractor": None,
            "safety_checker": None,  # self.stage_1.safety_checker,
            "watermarker": None,
        }
        self.stage_3 = DiffusionPipeline.from_pretrained(
            "/checkpoints/DeepFloyd-IF/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
        )
        self.stage_3.enable_model_cpu_offload()
        self.strength = strength
        print('Stage 3 loading done')

    def __call__(self, prompt, seed, scale=8.0, safe=True):
        generator = torch.manual_seed(seed)
        images = []
        if safe:
            if self.strength == 'strong':
                hyp = STRONG
            elif self.strength == 'max':
                hyp = MAX
            else:
                raise NotImplementedError
            for _ in range(2):
                images.extend(self.run([prompt] * 5, generator, scale=scale, **hyp))
        else:
            for _ in range(2):
                images.extend(self.run([prompt] * 5, generator, scale=scale))
        return images

    def run(self, prompt, generator, verbose=False, scale=8.0, editing_prompt=None, edit_guidance_scale=7,
            reverse_editing_direction=False, edit_warmup_steps=22, edit_threshold=0.82, edit_momentum_scale=0.5):

        # text embeds
        prompt_embeds, negative_embeds, edit_embeds = self.stage_1.encode_prompt(prompt, editing_prompt=editing_prompt)

        # stage 1
        image = self.stage_1(
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, edit_prompt_embeds=edit_embeds,
            generator=generator, output_type="pt", guidance_scale=scale,
            edit_guidance_scale=edit_guidance_scale, reverse_editing_direction=reverse_editing_direction,
            edit_warmup_steps=edit_warmup_steps, edit_threshold=edit_threshold, edit_momentum_scale=edit_momentum_scale
        ).images
        if verbose:
            pt_to_pil(image)[0].save(f"./test_if_stage_I_{verbose}.png")

        # stage 2
        image = self.stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt",
            guidance_scale=scale,
        ).images
        if verbose:
            pt_to_pil(image)[0].save(f"./test_if_stage_II_{verbose}.png")

        # stage 3
        image = self.stage_3(prompt=prompt, image=image,
                             noise_level=100, generator=generator).images

        if verbose:
            # size 1024
            image[0].save(f"./test_if_stage_III_{verbose}.png")

        return image