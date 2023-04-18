from dataclasses import dataclass
from diffusers import SemanticStableDiffusionPipeline
import torch

editing_prompt = 'an image showing hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty'
@dataclass
class SafetyConfig(object):
    MEDIUM = {'editing_prompt': editing_prompt, 'edit_guidance_scale': 4, 'edit_warmup_steps': 10, 'edit_momentum_scale': 0.3, 'edit_threshold': 0.9,  
          'reverse_editing_direction':True}
    STRONG = {'editing_prompt': editing_prompt, 'edit_guidance_scale': 7, 'edit_warmup_steps': 5, 'edit_momentum_scale': 0.5, 'edit_threshold': 0.85, 
          'reverse_editing_direction':True}
    MAX = {'editing_prompt': editing_prompt, 'edit_guidance_scale': 10, 'edit_warmup_steps': 0, 'edit_momentum_scale': 0.5, 'edit_threshold': 0.75, 
          'reverse_editing_direction':True}


config_cases = {
    'medium': SafetyConfig.MEDIUM,
    'strong': SafetyConfig.STRONG,
    'max': SafetyConfig.MAX,
}

class SSD:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", special_token='', strength='strong'):
        self.pipeline = SemanticStableDiffusionPipeline.from_pretrained(model_name,)
        self.model_name = model_name.replace('/', '-')
        self.pipeline.safety_checker=None
        self.config = config_cases[strength]
        self.max_image_size = 512
        self.images_per_gen = (2,5)
        device ='cuda'
        self.pipeline.to(device)
        self.gen = torch.Generator(device=device)
        self.special_token = special_token

    def __call__(self, prompt, seed, scale):
        #height, width = np.minimum(int(d['height']), self.max_image_size), np.minimum(int(d['width']), self.max_image_size)
        images = []
        self.gen.manual_seed(seed)
        for idx in range(self.images_per_gen[0]):
            out = self.pipeline(prompt=prompt + self.special_token, num_images_per_prompt=self.images_per_gen[1], generator=self.gen, **self.config)
            images.extend(out.images)
        return images
