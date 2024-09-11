#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : denoiser.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/28 23:10
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network.gan import DefaultUNet

from omegaconf import DictConfig


class DeNoiser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, predict: str = 'container'):
        super().__init__()
        self.model = DefaultUNet(in_channels, out_channels)
        self.predict = predict

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)


def run():
    pass


if __name__ == '__main__':
    run()
