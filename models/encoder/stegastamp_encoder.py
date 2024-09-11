#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : Steganography
# @File         : stegastamp_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2021/8/9 上午10:35

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network.basic_blocks import ConvBNRelu

from typing import Tuple, Optional


class StegaStampEncoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], secret_len: int = 100):
        super().__init__()

        self.image_shape = image_shape
        self.secret_len = secret_len

        self.secret_dense = nn.Sequential(
            nn.Linear(self.secret_len, 3 * 16 * 16),
            nn.ReLU(inplace=True),
        )

        self.conv1 = ConvBNRelu(6, 32, 3)
        self.conv2 = ConvBNRelu(32, 32, 3, stride=2)
        self.conv3 = ConvBNRelu(32, 64, 3, stride=2)
        self.conv4 = ConvBNRelu(64, 128, 3, stride=2)
        self.conv5 = ConvBNRelu(128, 256, 3, stride=2)
        self.up6 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBNRelu(256, 128, 3)
        )
        self.conv6 = ConvBNRelu(256, 128, 3)
        self.up7 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBNRelu(128, 64, 3)
        )
        self.conv7 = ConvBNRelu(128, 64, 3)
        self.up8 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBNRelu(64, 32, 3)
        )
        self.conv8 = ConvBNRelu(64, 32, 3)
        self.up9 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBNRelu(32, 32, 3)
        )
        self.conv9 = ConvBNRelu(64, 32, 3)
        self.residual = nn.Conv2d(32, 3, 1)

    @property
    def msg_len(self):
        return self.secret_len

    def forward(self, image, secret):
        _, _, height, width = image.shape

        secret = self.secret_dense(secret).reshape(-1, 3, 16, 16)
        secret_enlarged = F.interpolate(secret, (height, width))

        inputs = torch.cat([secret_enlarged, image], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(conv5)
        conv6 = self.conv6(torch.cat([conv4, up6], dim=1))
        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], dim=1))
        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], dim=1))
        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9], dim=1))
        residual = self.residual(conv9)
        return (residual + image).clamp(-1, 1)


def test():
    pass


if __name__ == '__main__':
    test()
