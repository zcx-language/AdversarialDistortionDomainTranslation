#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : random_rotation.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/18 00:59
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
from kornia.augmentation import RandomRotation as _RandomRotation
from functools import partial
from .ramp_base import RampBase


class RandomRotation(RampBase):
    def __init__(self,
                 degrees: Union[float, Tuple[float, float]] = 30.,
                 p: float = 1.,
                 step0: int = 0,
                 ramp: int = 10000):
        super().__init__(step0=step0, ramp=ramp)
        self.degrees = degrees
        self.rotation_fn = partial(_RandomRotation, p=p)

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        rotation_fn = self.rotation_fn(degrees=self.degrees * ramp_ratio)
        return rotation_fn(inputs)


def run():
    pass


if __name__ == '__main__':
    run()
