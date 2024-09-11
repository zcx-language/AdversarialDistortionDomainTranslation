#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : Steganography
# @File         : stegastamp_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2021/8/9 上午10:35

# Import lib here
import math
import torch.nn as nn
from models.network.basic_blocks import ConvBNRelu, DeformConvBNRelu
from typing import Tuple


class StegaStampDecoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], secret_len: int = 100, conv_layer: str = 'conv'):
        super().__init__()
        in_channels, height, width = image_shape
        flatten_dims = 128 * math.ceil(height / 32) * math.ceil(width / 32)

        if conv_layer == 'conv':
            conv_layer = ConvBNRelu
        elif conv_layer == 'deformable':
            conv_layer = DeformConvBNRelu
        else:
            raise ValueError(f"conv_layer should be 'conv' or 'deformable', but got {conv_layer}")

        self.conv_layers = nn.Sequential(
            conv_layer(in_channels, 32, 3, stride=2),
            conv_layer(32, 64, 3, stride=2),
            conv_layer(64, 64, 3, stride=2),
            conv_layer(64, 128, 3, stride=2),
            conv_layer(128, 128, 3, stride=2),
        )

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dims, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, secret_len)
        )

    def forward(self, image):
        conv_img = self.conv_layers(image.contiguous())
        secret_logits = self.projector(conv_img)
        return secret_logits


def test():
    pass


if __name__ == '__main__':
    test()
