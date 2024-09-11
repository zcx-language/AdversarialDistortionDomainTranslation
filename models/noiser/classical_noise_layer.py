#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : classical_noise_layer.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/7/6 上午9:41

# Import lib here
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from typing import Tuple


class RandomResize(nn.Module):
    def __init__(self, ratio_range: Tuple[float, float],
                 keep_aspect_ratio: bool = False,
                 interpolation: str = 'bilinear',
                 resize_back: bool = True):
        super().__init__()
        self.ratio_range = ratio_range
        self.keep_aspect_ratio = keep_aspect_ratio
        self.interpolation = interpolation
        self.resize_back = resize_back

    def forward(self, inputs: torch.Tensor):
        h_factor = random_float(self.ratio_range[0], self.ratio_range[1])
        w_factor = random_float(self.ratio_range[0], self.ratio_range[1]) if not self.keep_aspect_ratio else h_factor
        resized_image = F.interpolate(inputs, scale_factor=(h_factor, w_factor), mode=self.interpolation)
        if not self.resize_back:
            return resized_image
        return F.interpolate(resized_image, size=inputs.shape[-2:], mode=self.interpolation)


class PseudoCrop(nn.Module):
    """This implement is borrowed from MBRS"""
    def __init__(self, height_ratio: float,
                 width_ratio: float):
        super().__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def forward(self, inputs: torch.Tensor):
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(inputs.shape, self.height_ratio, self.width_ratio)
        mask = torch.zeros_like(inputs)
        mask[:, :, h_start:h_end, w_start:w_end] = 1
        return inputs * mask

    @staticmethod
    def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
        image_height = image_shape[2]
        image_width = image_shape[3]

        remaining_height = int(height_ratio * image_height)
        remaining_width = int(width_ratio * image_width)

        if remaining_height == image_height:
            height_start = 0
        else:
            height_start = np.random.randint(0, image_height - remaining_height)

        if remaining_width == image_width:
            width_start = 0
        else:
            width_start = np.random.randint(0, image_width - remaining_width)

        return height_start, height_start + remaining_height, width_start, width_start + remaining_width


def tensor_gaussian_noise(image: torch.Tensor, mean: float = 0.0, std: float = 0.2):
    noise = torch.randn(image.shape, device=image.device)
    return torch.clip(image + noise * std + mean, 0., 1.)


def tensor_gaussian_blur(image: torch.Tensor,
                         kernel_size: Tuple[int, int] = (3, 3),
                         sigma: Tuple[float, float] = (1., 1.),
                         border_type: str = "reflect"):
    return gaussian_blur2d(image, kernel_size, sigma, border_type.lower())


def random_float(min_v: float, max_v: float):
    return np.random.rand() * (max_v - min_v) + min_v


def tensor_resize(image: torch.Tensor,
                  resize_ratio_range: Tuple[float, float] = (0.8, 1.2),
                  interpolation: str = 'bilinear'):
    resize_ratio_height = random_float(resize_ratio_range[0], resize_ratio_range[1])
    resize_ratio_width = random_float(resize_ratio_range[0], resize_ratio_range[1])
    resized_image = F.interpolate(image, scale_factor=(resize_ratio_height, resize_ratio_width), mode=interpolation)
    return F.interpolate(resized_image, size=image.shape[2:], mode=interpolation)


def get_random_rectangle_inside(image, height, width):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range
    and width_ratio_range. This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9),
    then a random number in that range will be chosen (say it is 0.75 for illustration), and the image will be cropped
    such that the remaining height equals 0.75. In fact, a random 'starting' position rs will be chosen from (0, 0.25),
    and the crop will start at rs and end at rs + 0.75.
    This ensures that we crop from top/bottom with equal probability. The same logic applies to the width of the image,
    where width_ratio_range controls the width crop range.

    Args:
        image: The image we want to crop
        height: The range of remaining height ratio
        width: The range of remaining width ratio.
    """
    image_height = image.shape[2]
    image_width = image.shape[3]

    if height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - height)

    if width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - width)

    return height_start, height_start+height, width_start, width_start+width


def tensor_cropout(image: torch.Tensor,
                   container: torch.Tensor = None,
                   height_ratio_range: Tuple[float, float] = (0.7, 0.95),
                   width_ratio_range: Tuple[float, float] = (0.7, 0.95)):
    """
    Combines the container and cover images into a single image, as follows: Takes a crop of the container image, and
    takes the rest from the cover image. The resulting image has the same size as the original and the container images.
    """
    assert image.shape == container.shape
    _, _, height, width = image.shape
    mask_height = int(math.ceil(random_float(height_ratio_range[0], height_ratio_range[1]) * height))
    mask_width = int(math.ceil(random_float(width_ratio_range[0], width_ratio_range[1]) * width))
    h_start, h_end, w_start, w_end = get_random_rectangle_inside(container, mask_height, mask_width)
    cropout_mask = torch.zeros_like(image, dtype=torch.float32, device=image.device)
    cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1
    return container * cropout_mask + image * (1 - cropout_mask)


def tensor_dropout(image: torch.Tensor,
                   container: torch.Tensor,
                   ratio_range: Tuple[float, float] = (0.7, 0.95)):
    """
    Drops random pixels from the container image and substitues them with the pixels from the cover image
    """
    percent = np.random.uniform(ratio_range[0], ratio_range[1])
    mask = np.random.choice([0., 1.], image.shape[2:], p=[1-percent, percent])
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=image.device)
    mask_tensor = mask_tensor.expand_as(image)
    return container * mask_tensor + image * (1 - mask_tensor)


def run():
    import cv2
    from torchvision import transforms
    from utils.image_tools import image_show
    import torch.nn.functional as F

    # image_path = '/home/chengxin/Desktop/lena.png'
    image_path = '/mnt/chengxin/Datasets/Mirflickr/im41.jpg'
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))

    image_tsr = transforms.ToTensor()(image)
    image_show(image_tsr)
    image_tsr = torch.stack([image_tsr, image_tsr], dim=0)
    # blured_img = tensor_gaussian_blur(image_tsr)
    # image_show(blured_img[0])
    # noisy_img = tensor_gaussian_noise(image_tsr)
    # image_show(noisy_img[0])
    # resized_img = tensor_resize(image_tsr)
    # image_show(resized_img[0])
    cropout_img = tensor_cropout(image_tsr, image_tsr * 0.5)
    image_show(cropout_img[0])
    # dropout_img = tensor_dropout(image_tsr, image_tsr * 0.5)
    # image_show(dropout_img[0])

    pass


if __name__ == '__main__':
    run()
