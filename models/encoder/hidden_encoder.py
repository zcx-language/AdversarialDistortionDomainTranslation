#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : hidden_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2022/3/30 下午3:06

# Import lib here
import torch
import torch.nn as nn
from fastai.layers import ConvLayer


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


class HiDDeNEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, msg_len: int,
                 n_channels: int = 64):
        super().__init__()

        layers = [ConvBNRelu(3, n_channels)]

        for _ in range(3):
            layer = ConvBNRelu(n_channels, n_channels)
            layers.append(layer)

        self.msg_linear = nn.Linear(msg_len, msg_len)
        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(n_channels + 3 + msg_len, n_channels)
        self.final_layer = nn.Conv2d(n_channels, 3, kernel_size=1)

    def forward(self, img, msg):
        batch_size, n_channels, height, width = img.shape

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_msg = self.msg_linear(msg).unsqueeze(-1).unsqueeze(-1)

        expanded_msg = expanded_msg.expand(-1, -1, height, width)
        conv_image = self.conv_layers(img)
        # concatenate expanded message and image
        concat = torch.cat([expanded_msg, conv_image, img], dim=1)
        conv_concat = self.after_concat_layer(concat)
        container = self.final_layer(conv_concat)
        return container


def run():
    pass


if __name__ == '__main__':
    run()
