from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch


class IfT2I:
    def __init__(self, model_name=None, special_token=None, strength=None):
                
        # stage 1
        self.stage_1 = DiffusionPipeline.from_pretrained("/checkpoints/DeepFloyd-IF/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
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
            "feature_extractor": self.stage_1.feature_extractor,
            "safety_checker": None,# self.stage_1.safety_checker,
            "watermarker": None,#self.stage_1.watermarker,
        }
        self.stage_3 = DiffusionPipeline.from_pretrained(
            "/checkpoints/DeepFloyd-IF/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
        )
        self.stage_3.enable_model_cpu_offload()
        
        print('Stage 3 loading done')
        self.negative_prompt = None
        
    def __call__(self, prompt, seed, scale=8.0):
        generator = torch.manual_seed(seed)
        images = []
        batch_size = 5
        if self.negative_prompt is not None:
            neg_prompt = [self.negative_prompt] * batch_size
        else:
            neg_prompt = None
        for _ in range(2):
            images.extend(self.run([prompt]*batch_size, generator, scale=scale, neg_prompt=neg_prompt))
        return images

    def run(self, prompt, generator, scale=8.0, verbose=False, neg_prompt=None):

        # text embeds
        print('Neg prompt', neg_prompt)
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt, negative_prompt=neg_prompt)

        # stage 1
        image = self.stage_1(
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
            generator=generator, output_type="pt", guidance_scale=scale,
        ).images
        if verbose:
            pt_to_pil(image)[0].save(f"./test_if_stage_I_{verbose}.png")

        # stage 2
        image = self.stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt", guidance_scale=scale,
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

def test():
    m = IfT2I()
    # images generated have size 64, 256, 1024
    generator = torch.manual_seed(1)
    images = []
    for i in range(2):
        images.extend(m.run(['japanese body']*2, generator, verbose=f'{i}'))

if __name__ == '__main__':
    test()