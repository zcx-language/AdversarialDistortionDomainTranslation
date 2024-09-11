#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : random_resize.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/18 00:46
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
import torch.nn.functional as F
from .ramp_base import RampBase


class RandomResize(RampBase):
    """Randomly resize an image."""
    def __init__(self,
                 ratio: Tuple[float, float] = (0.5, 2.0),
                 keep_aspect: bool = False,
                 size: Optional[Tuple[int, int]] = None,
                 p: float = 1.,
                 step0: int = 0,
                 ramp: int = 10000,
                 random: bool = False):
        super().__init__(step0=step0, ramp=ramp)
        assert ratio[0] < 1. < ratio[1]
        self.ratio = ratio
        self.keep_aspect = keep_aspect
        self.size = size
        self.p = p
        self.random = random

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return inputs

        ramp_raio = self.ramp_fn(step) if step is not None else 1.
        if torch.rand(1).item() < 0.5:
            # down sample image
            down_ratio = 1 - (1 - self.ratio[0]) * ramp_raio
            if self.random:
                height_factor = torch.rand(1).item() * (1 - down_ratio) + down_ratio
                width_factor = height_factor if self.keep_aspect else torch.rand(1).item() * (1 - down_ratio) + down_ratio
            else:
                height_factor = down_ratio
                width_factor = down_ratio
        else:
            # up sample image
            up_ratio = 1 + (self.ratio[1] - 1) * ramp_raio
            if self.random:
                height_factor = torch.rand(1).item() * (up_ratio - 1) + 1
                width_factor = height_factor if self.keep_aspect else torch.rand(1).item() * (up_ratio - 1) + 1
            else:
                height_factor = up_ratio
                width_factor = up_ratio
        inputs = F.interpolate(inputs, scale_factor=(height_factor, width_factor))

        if self.size is not None:
            inputs = F.interpolate(inputs, size=tuple(self.size))
        return inputs


def run():
    from PIL import Image
    from matplotlib import pyplot as plt
    import numpy as np
    import torchvision.transforms.functional as tvf

    img = Image.open('/home/chengxin/Desktop/my.png').convert('RGB')
    img = img.resize((256, 256))
    img = tvf.to_tensor(img).unsqueeze(0)

    d_img_list = []
    ramp_distortion = RandomResize(size=(256, 256))
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
