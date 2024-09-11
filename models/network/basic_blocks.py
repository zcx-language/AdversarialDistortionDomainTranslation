#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : basic_blocks.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/17 15:49
#
# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2dPack as DeformConv2d

from typing import Optional, Tuple


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size: int = 3,
                 stride: int = 1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class DeformConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size: int = 3,
                 stride: int = 1,
                 deformable_groups: int = 1,
                 dilation: int = 1):
        super(DeformConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            DeformConv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=False,
                         deformable_groups=deformable_groups, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


def run():
    pass


if __name__ == '__main__':
    run()
