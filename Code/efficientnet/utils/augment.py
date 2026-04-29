import random

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

_LEVEL_DENOM = 10.

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    translate_pct=0.45,
    image_mean=(128, 128, 128),
)

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def random_resample():
    return random.choice(_RANDOM_INTERPOLATION)


def auto_contrast(image, _):
    return ImageOps.autocontrast(image)


def equalize(image, _):
    return ImageOps.equalize(image)


def invert(image, _):
    return ImageOps.invert(image)


def rotate(image, magnitude):
    degrees = (magnitude / _LEVEL_DENOM) * 30.
    if random.random() > 0.5:
        degrees *= -1

    return image.rotate(degrees, resample=random_resample())


def posterize(image, magnitude):
    bits_to_keep = int((magnitude / _LEVEL_DENOM) * 4)
    if bits_to_keep >= 8:
        return image
    return ImageOps.posterize(image, bits_to_keep)


def solarize(image, magnitude):
    thresh = int((magnitude / _LEVEL_DENOM) * 256)
    return ImageOps.solarize(image, thresh)


def solarize_add(image, magnitude):
    add = int((magnitude / _LEVEL_DENOM) * 110)
    thresh = 128
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        return image


def color(image, magnitude):
    factor = (magnitude / _LEVEL_DENOM) * 1.8 + 0.1
    return ImageEnhance.Color(image).enhance(factor)


def contrast(image, magnitude):
    factor = (magnitude / _LEVEL_DENOM) * 1.8 + 0.1
    return ImageEnhance.Contrast(image).enhance(factor)


def brightness(image, magnitude):
    factor = (magnitude / _LEVEL_DENOM) * 1.8 + 0.1
    return ImageEnhance.Brightness(image).enhance(factor)


def sharpness(image, magnitude):
    factor = (magnitude / _LEVEL_DENOM) * 1.8 + 0.1
    return ImageEnhance.Sharpness(image).enhance(factor)


def shear_x(image, magnitude):
    factor = (magnitude / _LEVEL_DENOM) * 0.3
    if random.random() > 0.5:
        factor *= -1
    return image.transform(image.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), resample=random_resample())


def shear_y(image, magnitude):
    factor = (magnitude / _LEVEL_DENOM) * 0.3
    if random.random() > 0.5:
        factor *= -1
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), resample=random_resample())


def translate_x(image, level):
    translate_pct = _HPARAMS_DEFAULT['translate_pct']
    pct = (level / _LEVEL_DENOM) * translate_pct
    if random.random() > 0.5:
        pct *= -1
    pixels = pct * image.size[0]
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), resample=random_resample())


def translate_y(image, level):
    translate_pct = _HPARAMS_DEFAULT['translate_pct']
    pct = (level / _LEVEL_DENOM) * translate_pct
    if random.random() > 0.5:
        pct *= -1
    pixels = pct * image.size[1]
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), resample=random_resample())


_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateX',
    'TranslateY',
]

NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
}


class RandAugment:
    def __init__(self, magnitude=9, magnitude_std=0.5, magnitude_max=None, num_operations=2):
        self.prob = 0.5
        self.magnitude = magnitude
        self.magnitude_std = magnitude_std
        self.magnitude_max = magnitude_max
        self.num_operations = num_operations  # number of operations per image

        self.transforms = [NAME_TO_OP[transform] for transform in _RAND_TRANSFORMS]

    def __call__(self, image):
        if self.prob < 1.0 and random.random() > self.prob:
            return image

        magnitude = self.magnitude
        if self.magnitude_std > 0:
            # magnitude randomization enabled
            if self.magnitude_std == float('inf'):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)

        upper_bound = self.magnitude_max or _LEVEL_DENOM
        magnitude = max(0., min(magnitude, upper_bound))

        transforms = np.random.choice(self.transforms, self.num_operations)
        for transform in transforms:
            image = transform(image, magnitude)

        return image


if __name__ == '__main__':
    import PIL

    pil_image = PIL.Image.open('../../assets/cat.jpg')
    random_augment = RandAugment()
    while True:
        augmented_image = random_augment(pil_image)  # apply random augment
        np_image = np.array(augmented_image)  # convert to numpy

        cv2.imshow('frame', np_image)
        if cv2.waitKey(0) & 0xFF == ord('q'):  # press [`q`] to quit
            break
