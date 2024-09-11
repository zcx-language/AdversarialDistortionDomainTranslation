#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : random_perspective.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/12 22:52
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
from functools import partial
from kornia.augmentation import RandomPerspective as _RandomPerspective
from .ramp_base import RampBase


class RandomPerspective(RampBase):
    def __init__(self,
                 distortion_scale: float = 0.5,
                 p: float = 1.,
                 sampling_method: str = "basic",
                 step0: int = 0,
                 ramp: int = 10000):
        super().__init__(step0=step0, ramp=ramp)
        self.distortion_scale = distortion_scale
        self.random_perspective = partial(_RandomPerspective, p=p, sampling_method=sampling_method)

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        distortion_scale = self.distortion_scale * ramp_ratio
        random_perspective = self.random_perspective(distortion_scale=distortion_scale)
        return random_perspective(inputs)


def run():
    from PIL import Image
    from matplotlib import pyplot as plt
    import numpy as np
    import torchvision.transforms.functional as tvf

    img = Image.open('/home/chengxin/Desktop/my.png').convert('RGB')
    img = img.resize((256, 256))
    img = tvf.to_tensor(img).unsqueeze(0)

    d_img_list = []
    ramp_distortion = RandomPerspective()
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
