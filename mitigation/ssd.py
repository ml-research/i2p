from diffusers import StableDiffusionPipelineSafe
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
import torch

config_cases = {
    #'weak': SafetyConfig.WEAK,
    'medium': SafetyConfig.MEDIUM,
    'strong': SafetyConfig.STRONG,
    'max': SafetyConfig.MAX,
}
class SSD:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", special_token='', strength='strong'):
        self.pipeline = StableDiffusionPipelineSafe.from_pretrained(model_name)
        self.model_name = model_name.replace('/', '-')
        self.pipeline.safety_checker = None
        self.config = config_cases[strength]
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
            out = self.pipeline(prompt=prompt + self.special_token, num_images_per_prompt=self.images_per_gen[1], generator=self.gen, **self.config)
            images.extend(out.images)
        return images

