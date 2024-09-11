#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : random_affine.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/12 23:04
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
from functools import partial
from kornia.augmentation import RandomAffine as _RandomAffine
from .ramp_base import RampBase


class RandomAffine(RampBase):
    def __init__(self,
                 degrees: float = 10,
                 translate: Optional[float] = None,
                 scale: Optional[Tuple[float, float]] = None,
                 shear: Optional[float] = None,
                 p: float = 1.,
                 padding_mode: str = "zeros",
                 step0: int = 0,
                 ramp: int = 10000):
        super().__init__(step0=step0, ramp=ramp)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.random_affine = partial(_RandomAffine, p=p, padding_mode=padding_mode)

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        degrees = (-self.degrees * ramp_ratio, self.degrees * ramp_ratio)
        if self.translate:
            translate = (0, self.translate * ramp_ratio)
        else:
            translate = None
        if self.scale:
            zoom_in_scale = 1 - self.scale[0]
            zoom_out_scale = self.scale[1] - 1
            scale = (1 - zoom_in_scale * ramp_ratio, 1 + zoom_out_scale * ramp_ratio)
        else:
            scale = None
        if self.shear:
            shear = (-self.shear * ramp_ratio, self.shear * ramp_ratio)
        else:
            shear = None
        random_affine = self.random_affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
        return random_affine(inputs)


def run():
    from PIL import Image
    from matplotlib import pyplot as plt
    import numpy as np
    import torchvision.transforms.functional as tvf

    img = Image.open('/home/chengxin/Desktop/my.png').convert('RGB')
    img = img.resize((256, 256))
    img = tvf.to_tensor(img).unsqueeze(0)

    d_img_list = []
    ramp_distortion = RandomAffine()
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
