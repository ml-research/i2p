from diffusers import (
    AltDiffusionPipeline,
    AltDiffusionImg2ImgPipeline,
)
import torch

class AltDiffusion:
    def __init__(self, model_name="BAAI/AltDiffusion-m9", special_token='', strength='strong'):
        self.pipeline = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m9")
        self.model_name = model_name.replace('/', '-')
        self.pipeline.safety_checker = None
        self.max_image_size = 512
        self.images_per_gen = (2,5)
        device ='cuda'
        self.pipeline.to(device)
        self.gen = torch.Generator(device=device)
        self.special_token = special_token

    def __call__(self, prompt, seed, scale):
        images = []
        self.gen.manual_seed(seed)
        for idx in range(self.images_per_gen[0]):
            out = self.pipeline(prompt=prompt + self.special_token, num_images_per_prompt=self.images_per_gen[1], generator=self.gen)
            images.extend(out.images)
        return images
# now you can use text2img(...) and img2img(...) just like the call methods of each respective pipeline