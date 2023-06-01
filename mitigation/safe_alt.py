from mitigation.SemanticAlt import SemanticAltDiffusionPipeline
from dataclasses import dataclass
import torch

editing_prompt = 'an image showing hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty'

MEDIUM = {'editing_prompt': editing_prompt, 'edit_guidance_scale': 4, 'edit_warmup_steps': 10,
          'edit_momentum_scale': 0.3, 'edit_threshold': 0.9,
          'reverse_editing_direction': True}
STRONG = {'editing_prompt': editing_prompt, 'edit_guidance_scale': 7, 'edit_warmup_steps': 5,
          'edit_momentum_scale': 0.5, 'edit_threshold': 0.85,
          'reverse_editing_direction': True}
MAX = {'editing_prompt': editing_prompt, 'edit_guidance_scale': 10, 'edit_warmup_steps': 0, 'edit_momentum_scale': 0.5,
       'edit_threshold': 0.75,
       'reverse_editing_direction': True}



class SafeAltDiffusion:
    def __init__(self, model_name="altdiffusion-m9", special_token='', strength='strong'):

        self.model_name = model_name#.replace('/', '-')

        self.pipeline = SemanticAltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m9")
        self.pipeline.safety_checker = None
        self.strength = strength
        device = 'cuda'
        self.pipeline.to(device)

    def __call__(self, prompt, seed, scale, safe=True):
        # height, width = np.minimum(int(d['height']), self.max_image_size), np.minimum(int(d['width']), self.max_image_size)
        generator = torch.manual_seed(seed)
        images = []
        if safe:
            if self.strength == 'medium':
                hyp = MEDIUM
            elif self.strength == 'strong':
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

        return self.pipeline(prompt=prompt, generator=generator, guidance_scale=scale,
                             editing_prompt=editing_prompt, edit_guidance_scale=edit_guidance_scale,
                             edit_warmup_steps=edit_warmup_steps, edit_threshold=edit_threshold,
                             reverse_editing_direction=reverse_editing_direction,
                             edit_momentum_scale=edit_momentum_scale).images