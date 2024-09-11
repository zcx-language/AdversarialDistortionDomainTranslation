#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : random_gaussian_blur.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/16 15:13
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
from kornia.filters import gaussian_blur2d as _gaussian_blur2d
from .ramp_base import RampBase


class RandomGaussianBlur(RampBase):
    """Randomly blur an image with gaussian kernel.
    Args:
        max_kernel_size (int): Max kernel size of the gaussian kernel, the default min kernel size is 3.
        p (float): Probability of the gaussian blur transformation. Default value is 1.
        step0 (int): The step to start ramping.
        ramp (int): The number of steps to ramp.
    """
    def __init__(self,
                 max_kernel_size: int = 9,
                 p: float = 1.,
                 step0: int = 0,
                 ramp: int = 10000,
                 random: bool = False):
        super().__init__(step0=step0, ramp=ramp)
        self.max_kernel_size = max_kernel_size
        self.p = p
        self.random = random

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return inputs

        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        kernel_size = self.max_kernel_size * ramp_ratio
        if self.random:
            kernel_size = kernel_size * torch.rand(1).item()
        kernel_size = max(kernel_size // 2 * 2 + 1, 3)
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8   # This is a heuristic formula
        return _gaussian_blur2d(inputs, (kernel_size, kernel_size), (sigma, sigma))


def run():
    from PIL import Image
    from matplotlib import pyplot as plt
    import numpy as np
    import torchvision.transforms.functional as tvf

    img = Image.open('/home/chengxin/Desktop/my.png').convert('RGB')
    img = img.resize((256, 256))
    img = tvf.to_tensor(img).unsqueeze(0)

    d_img_list = []
    ramp_distortion = RandomGaussianBlur()
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
