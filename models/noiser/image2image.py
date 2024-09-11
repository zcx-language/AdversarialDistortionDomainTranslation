#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : image2image.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/4 14:46
#
# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.layers import ConvLayer
from fastai.vision.models.unet import DynamicUnet
from typing import Tuple, List


class UNet(nn.Module):
    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 out_channels: int,
                 out_range: Tuple[float, float]):
        super().__init__()
        in_channels = image_shape[0]
        image_size = image_shape[1:]

        encoder = nn.Sequential(
            ConvLayer(in_channels, 32, ks=4, stride=2, padding=1),  # 64
            ConvLayer(32, 64, ks=4, stride=2, padding=1),   # 32
            ConvLayer(64, 128, ks=4, stride=2, padding=1),  # 16
            ConvLayer(128, 256, ks=4, stride=2, padding=1),     # 8
        )
        self.model = DynamicUnet(encoder, out_channels, image_size, y_range=out_range)

    def forward(self, x):
        return self.model(x)


class Image2Image(nn.Module):
    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 out_channels: int,
                 out_range: Tuple[float, float],
                 ):
        super().__init__()

        self.generator = UNet(image_shape, out_channels, out_range)


def run():
    pass


if __name__ == '__main__':
    run()
