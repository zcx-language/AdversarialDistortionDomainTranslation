#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : ps_effects.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/5/23 下午10:45

# Import lib here
import numpy as np
import cv2
import torch
import torch.nn as nn
import albumentations as A
from kornia.utils import tensor_to_image


def tensor_ps_effects(tensor: torch.Tensor, effect: str):
    effect = effect.lower()

    if effect == 'rainy':
        transform = A.RandomRain(rain_type='heavy', always_apply=True)
    elif effect == 'solarize':
        transform = A.Solarize(always_apply=True)
    elif effect == 'fisheye':
        transform = A.OpticalDistortion(distort_limit=(0.9, 0.91), always_apply=True)
    elif effect == 'emboss':
        transform = A.Emboss((0.9, 0.91), strength=(0.9, 0.91), always_apply=True)
    else:
        raise ValueError

    assert tensor.min() >= 0. and tensor.max() <= 1., \
        f'Error, Need tensor value range in [0, 1], but got [{tensor.min()}, {tensor.max()}]'

    tensor_shape = tensor.shape
    img = tensor_to_image(tensor, keepdim=True)
    img = np.clip(img * 255. + 0.5, 0., 255.).astype(np.uint8)

    if len(tensor_shape) == 3:
        transformed_img = transform(image=img)['image']
        transformed_tsr = torch.tensor(transformed_img.transpose((2, 0, 1)), dtype=torch.float32, device=tensor.device)
    elif len(tensor_shape) == 4:
        transformed_imgs = []
        for i in img:
            transformed_img = transform(image=i)['image']
            transformed_imgs.append(transformed_img)
        transformed_imgs = np.stack(transformed_imgs, axis=0)
        transformed_tsr = torch.tensor(transformed_imgs, dtype=torch.float32, device=tensor.device).permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Cannot process tensor with shape {tensor_shape}")
    return transformed_tsr / 255.


class RainyEffect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        return tensor_ps_effects(image, 'rainy')


class EmbossEffect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        return tensor_ps_effects(image, 'emboss')


class FisheyeEffect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        return tensor_ps_effects(image, 'fisheye')


def run():
    from torchvision import transforms
    from utils.image_tools import image_show
    import torch.nn.functional as F

    image_path = '/home/chengxin/Desktop/lena.png'
    # image_path = '/mnt/chengxin/Datasets/Mirflickr/im41.jpg'
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))

    image_show(image)
    rainy_img = A.RandomRain(blur_value=3, rain_type='heavy', always_apply=True)(image=image)['image']
    image_show(rainy_img)
    emboss_img = A.Emboss(alpha=(0.9, 0.91), strength=(0.9, 0.91), always_apply=True)(image=image)['image']
    image_show(emboss_img)
    elastic_img = A.OpticalDistortion(distort_limit=(0.9, 0.91), always_apply=True)(image=image)['image']
    image_show(elastic_img)

    # image_tsr = transforms.ToTensor()(image)
    # image_show(image_tsr)
    # image_tsr = torch.stack([image_tsr, image_tsr], dim=0)
    # image_tsr = F.interpolate(image_tsr, size=(256, 256))
    # effected_img = tensor_ps_effects(image_tsr, 'rainy')
    # effected_img = F.interpolate(effected_img, size=(128, 128))
    # image_show(effected_img[1])
    # effected_img = tensor_ps_effects(image_tsr, 'emboss')
    # image_show(effected_img[1])
    # effected_img = tensor_ps_effects(image_tsr, 'fisheye')
    # image_show(effected_img[1])

    pass


if __name__ == '__main__':
    run()
