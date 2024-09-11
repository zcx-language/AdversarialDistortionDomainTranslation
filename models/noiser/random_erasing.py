#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : random_erasing.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/12 18:03
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
from functools import partial
from kornia.augmentation import RandomErasing as _RandomErasing
from .ramp_base import RampBase


class RandomErasing(RampBase):
    def __init__(self,
                 scale: Tuple[float, float] = (0.02, 0.25),
                 ratio: float = (0.3, 3.3),
                 value: float = 0.0,
                 p: float = 1.,
                 step0: int = 0,
                 ramp: int = 10000):
        super().__init__(step0=step0, ramp=ramp)
        self.scale = scale
        self.ratio = ratio

        self.random_erasing = partial(_RandomErasing, value=value, p=p)

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        # up_scale = self.scale[1] * self.ramp_fn(step)
        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        scale = (self.scale[0], self.scale[0] + (self.scale[1] - self.scale[0]) * ramp_ratio)
        random_erasing = self.random_erasing(scale=scale, ratio=self.ratio)
        return random_erasing(inputs)


def run():
    from PIL import Image
    from matplotlib import pyplot as plt
    import numpy as np
    import torchvision.transforms.functional as tvf

    img = Image.open('/home/chengxin/Desktop/my.png').convert('RGB')
    img = img.resize((256, 256))
    img = tvf.to_tensor(img).unsqueeze(0)

    d_img_list = []
    ramp_distortion = RandomErasing()
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
