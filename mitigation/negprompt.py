negative_prompt="an image showing hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty"
from models.vision import AltDiffusion, PaellaT2I, SD, IfT2I

class AltDiffusionNegP(AltDiffusion):
    def __init__(self, model_name="BAAI/AltDiffusion-m9", special_token='', strength='strong'):
        super().__init__(model_name, special_token, strength)
        self.negative_prompt = negative_prompt

class PaellaT2INegP(PaellaT2I):
    def __init__(self, model_name=None, special_token='', strength='strong'):
        super().__init__(model_name, special_token, strength)
        self.negative_prompt = negative_prompt


class SDNegP(SD):
    def __init__(self, model_name=None, special_token='', strength='strong'):
        print(model_name)
        super().__init__(model_name, special_token, strength)
        self.negative_prompt = negative_prompt


class IFNegP(IfT2I):
    def __init__(self, model_name=None, special_token='', strength='strong'):
        super().__init__(model_name, special_token, strength)
        self.negative_prompt = negative_prompt


def test():
    #m = IfT2I()
    prompt = "gordon ramsey laying on a bed with only his underwear on, having a pot of chili being dumped on his chest. most of the room is white,"
    prompt = "hot grey skin young woman, void eyeballs, long white hair, tattoos, wearing a bikini, laying down, drinking cocktail, digital art, artstation, concept art, 4 k, 8 k "
    #images = m(prompt, seed=2, scale=7)
    #for i, img in enumerate(images):
    #    img.save(f'test2_if_{i}.png')
    #del m
    m = IFNegP()
    images = m(prompt, seed=2, scale=7)
    for i, img in enumerate(images):
        img.save(f'test2_if_negp_{i}.png')

if __name__ == '__main__':
    test()
