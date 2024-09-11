#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : random_salt_pepper.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/22 23:52
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
from .ramp_base import RampBase


class RandomSaltPepper(RampBase):
    def __init__(self,
                 prob: float = 0.04,
                 p: float = 1.,
                 step0: int = 0,
                 ramp: int = 10000):
        super().__init__(step0=step0, ramp=ramp)
        self.prob = prob
        self.p = p

    def __call__(self, inputs, step: int = None):
        if torch.rand(1).item() > self.p:
            return inputs

        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        prob = self.prob * ramp_ratio
        salt_prob = prob / 2
        pepper_prob = prob / 2
        noise = torch.rand_like(inputs)
        outputs = torch.where(noise < salt_prob, torch.ones_like(inputs), inputs)
        outputs = torch.where(noise > 1 - pepper_prob, torch.zeros_like(inputs), outputs)
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
    ramp_distortion = RandomSaltPepper()
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
