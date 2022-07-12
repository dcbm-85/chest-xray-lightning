# Adapted from https://github.com/stanfordmlgroup/MoCo-CXR

import numpy as np
import torchvision.transforms as T
from PIL import ImageEnhance

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_enhance_transform(f, enhance_min, enhance_max):
        def do_enhancement(img):
            factor = np.random.uniform(enhance_min, enhance_max)
            enhancer = f(img)
            return enhancer.enhance(factor)
        return do_enhancement
    
def GetTransforms(img, cfg, mode):
    """Set the transforms to be applied when loading."""

    transforms_list = [T.Resize((cfg.scale, cfg.scale))]

    # Data augmentation
    if mode == 'train':
        if np.random.rand() < cfg.rotate_prob:
            transforms_list += [T.RandomRotation((cfg.rotate_min,
                                                  cfg.rotate_max),fill=128)]

        if np.random.rand() < cfg.contrast_prob:
            transforms_list += [get_enhance_transform(ImageEnhance.Contrast,
                                                           cfg.contrast_min,
                                                           cfg.contrast_max)]

        if np.random.rand() < cfg.brightness_prob:
            transforms_list += [get_enhance_transform(ImageEnhance.Brightness,
                                                           cfg.brightness_min,
                                                           cfg.brightness_max)]

        if np.random.rand() < cfg.sharpness_prob:
            transforms_list += [get_enhance_transform(ImageEnhance.Sharpness,
                                                           cfg.sharpness_min,
                                                           cfg.sharpness_max)]

        if np.random.rand() < cfg.horizontal_flip_prob:
            transforms_list += [T.RandomHorizontalFlip()]

        if cfg.crop != 0:
            transforms_list += [T.RandomCrop((cfg.crop, cfg.crop))]

    else:
        transforms_list += [T.CenterCrop((cfg.crop,
                                          cfg.crop))
                            if cfg.crop else None]

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    transforms_list += [T.ToTensor(), normalize]

    return T.Compose([transform for transform in transforms_list if transform])(img)