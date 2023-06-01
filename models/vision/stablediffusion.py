from diffusers import StableDiffusionPipeline
import torch
class SD:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", special_token='', strength='strong'):
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_name)
        self.model_name = model_name.replace('/', '-')
        self.pipeline.safety_checker = None
        self.max_image_size = 512
        self.images_per_gen = (2,5)
        device ='cuda'
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


def test():
    m = SD(model_name='dreamlike-art/dreamlike-photoreal-2.0')
    res = m.run("an image of a beautiful woman", scale=7, batch_size=2)
    for i, r in enumerate(res):
        r.save(f'./tmp_sd{i}.png')


if __name__ == '__main__':
    test()