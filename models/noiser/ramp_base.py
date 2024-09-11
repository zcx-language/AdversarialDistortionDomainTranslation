#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : ramp_base.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/16 15:45
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch


class RampBase:
    def __init__(self,
                 step0: int = 0,
                 ramp: int = 10000):
        super().__init__()
        self.step0 = step0
        self.ramp = ramp

    def ramp_fn(self, step: int):
        return min(max(step - self.step0, 1) / self.ramp, 1.)

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        raise NotImplementedError


def run():
    pass


if __name__ == '__main__':
    run()
