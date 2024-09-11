#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : random_crop.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/18 00:04
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
import torch.nn.functional as F
from .ramp_base import RampBase


class RandomCrop(RampBase):
    def __init__(self,
                 min_ratio: float = 0.5,
                 keep_aspect: bool = False,
                 size: Optional[Tuple[int, int]] = None,
                 p: float = 1.,
                 step0: int = 0,
                 ramp: int = 10000):
        super().__init__(step0=step0, ramp=ramp)
        self.min_ratio = min_ratio
        self.keep_aspect = keep_aspect
        self.size = size
        self.p = p

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return inputs

        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        min_ratio = 1. - ramp_ratio * (1 - self.min_ratio)

        height_ratio = min_ratio + (1. - min_ratio) * torch.rand(1).item()
        width_ratio = height_ratio if self.keep_aspect else min_ratio + (1. - min_ratio) * torch.rand(1).item()

        height, width = inputs.shape[-2:]
        h = int(height * height_ratio)
        w = int(width * width_ratio)
        x = torch.randint(0, width - w + 1, (1,)).item()
        y = torch.randint(0, height - h + 1, (1,)).item()
        outputs = inputs[:, :, y:y + h, x:x + w]
        if self.size:
            outputs = F.interpolate(outputs, size=tuple(self.size))
        return outputs


def run():
    from PIL import Image
    from matplotlib import pyplot as plt
    import numpy as np
    import torchvision.transforms.functional as tvf

    img = Image.open('/home/chengxin/Desktop/my.png').convert('RGB')
    img = img.resize((256, 256))
    img = tvf.to_tensor(img).unsqueeze(0)

    d_img_list = []
    ramp_distortion = RandomCrop(size=(256, 256))
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
