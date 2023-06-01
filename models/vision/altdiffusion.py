from diffusers import (
    AltDiffusionPipeline,
    AltDiffusionImg2ImgPipeline,
)
import torch


class AltDiffusion:
    def __init__(self, model_name="BAAI/AltDiffusion-m9", special_token='', strength='strong'):
        self.pipeline = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m9")
        self.model_name = model_name#.replace('/', '-')
        self.pipeline.safety_checker = None
        self.max_image_size = 512
        self.images_per_gen = (2, 5)
        device = 'cuda'
        self.pipeline.to(device)
        self.gen = torch.Generator(device=device)
        self.special_token = special_token
        self.negative_prompt = None

    def __call__(self, prompt, seed, scale):
        images = []
        self.gen.manual_seed(seed)
        for idx in range(self.images_per_gen[0]):
            out = self.pipeline(prompt=prompt + self.special_token, num_images_per_prompt=self.images_per_gen[1],
                                generator=self.gen, negative_prompt=self.negative_prompt)
            images.extend(out.images)
        return images

def main():
    m = AltDiffusion()
    images = m('bouquet of roses', seed=2, scale=7)
    # images[0].save('tmp_altdiffusion.png')


if __name__ == '__main__':
    main()
# now you can use text2img(...) and img2img(...) just like the call methods of each respective pipeline
