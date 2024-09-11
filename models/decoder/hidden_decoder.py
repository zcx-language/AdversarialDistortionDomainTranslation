#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : hidden_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2022/3/30 下午4:06

# Import lib here
import torch
import torch.nn as nn


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class HiDDeNDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, msg_len: int,
                 n_channels: int = 64):

        super().__init__()

        layers = [ConvBNRelu(3, n_channels)]
        for _ in range(6):
            layers.append(ConvBNRelu(n_channels, n_channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(n_channels, msg_len))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(msg_len, msg_len)

    def forward(self, img):
        x = self.layers(img)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x = self.linear(x.squeeze())
        return x


def run():
    pass


if __name__ == '__main__':
    run()
