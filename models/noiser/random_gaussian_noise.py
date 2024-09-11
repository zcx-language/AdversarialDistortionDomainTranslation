#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : random_gaussian_noise.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/8 20:37
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
import torch.nn as nn
from .ramp_base import RampBase


class RandomGaussianNoise(RampBase):
    """Add gaussian noise to the input image.
    Args:
        mean (float): Mean of the gaussian noise.
        max_sigma (float): Max standard deviation of the gaussian noise.
        p (float): Probability of the gaussian noise transformation. Default value is 1.
        step0 (int): The step to start ramping.
        ramp (int): The number of steps to ramp.
    """
    def __init__(self,
                 mean: float = 0.0,
                 max_sigma: float = 0.05,
                 p: float = 1.,
                 step0: int = 0,
                 ramp: int = 10000,
                 random: bool = False):
        super().__init__(step0=step0, ramp=ramp)
        self.mean = mean
        self.max_sigma = max_sigma
        self.p = p
        self.random = random

    def __call__(self, inputs, step: int = None):
        if torch.rand(1).item() > self.p:
            return inputs

        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        sigma = self.max_sigma * ramp_ratio
        if self.random:
            sigma = sigma * torch.rand(1).item()
        noise = torch.randn_like(inputs) * sigma + self.mean
        return (inputs + noise).clamp(0., 1.)


def run():
    from PIL import Image
    from matplotlib import pyplot as plt
    import numpy as np
    import torchvision.transforms.functional as tvf

    img = Image.open('/home/chengxin/Desktop/my.png').convert('RGB')
    img = img.resize((256, 256))
    img = tvf.to_tensor(img).unsqueeze(0)

    d_img_list = []
    ramp_distortion = RandomGaussianNoise()
    for i in list([0, 2500, 5000, 7500, 10000]):
        d_img_list.append(ramp_distortion(img, i).squeeze(0).permute(1, 2, 0).numpy())

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, ax in enumerate(axes):
        ax.imshow(d_img_list[i])
        ax.set_title(f'step: {i * 2500}')
    plt.show()
    pass


if __name__ == '__main__':
    run()
