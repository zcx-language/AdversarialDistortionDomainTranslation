#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : random_colorjitter.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/19 19:56
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
import torch.nn as nn
from functools import partial
from kornia.augmentation import ColorJitter
from .ramp_base import RampBase


class RandomColorJitter(RampBase):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor is chosen
            uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor is chosen uniformly
            from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor is chosen
            uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue (float): How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue].
             Should have 0<= hue <= 0.5.
        p (float): Probability of the color jitter transformation. Default value is 1.
        step0 (int): The step to start ramping.
        ramp (int): The number of steps to ramp.
    """
    def __init__(self,
                 brightness: float = 0.5,
                 contrast: float = 0.5,
                 saturation: float = 0.5,
                 hue: float = 0.2,
                 p: float = 1.0,
                 step0: int = 0,
                 ramp: int = 10000):
        super().__init__(step0=step0, ramp=ramp)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.color_jitter = partial(ColorJitter, p=p)

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        bright = self.brightness * ramp_ratio
        contrast = self.contrast * ramp_ratio
        saturation = self.saturation * ramp_ratio
        hue = self.hue * ramp_ratio
        color_jitter_fn = self.color_jitter(brightness=bright, contrast=contrast, saturation=saturation, hue=hue)
        return color_jitter_fn(inputs)


def run():
    pass


if __name__ == '__main__':
    run()
