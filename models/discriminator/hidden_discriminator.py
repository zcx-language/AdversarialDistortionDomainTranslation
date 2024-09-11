#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : hidden_discriminator.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2022/3/30 下午4:13

# Import lib here
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from fastai.layers import ConvLayer


class HiDDeNDiscriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """

    def __init__(self, n_channels: int = 64):
        super().__init__()

        layers = [ConvLayer(3, n_channels)]
        for _ in range(2):
            layers.append(ConvLayer(n_channels, n_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(n_channels, 1)

    def forward(self, img):
        x = self.before_linear(img)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.

        x = self.linear(x.squeeze())
        # X = torch.sigmoid(X)
        return x


def run():
    pass


if __name__ == '__main__':
    run()
