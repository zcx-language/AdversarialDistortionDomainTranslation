#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : pimog_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/3/3 17:32
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class SingleConv(nn.Module):
    def __init__(self, inchannel, outchannel, s):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=s, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, s):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if s != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            ConvBNRelu(3, num_channels),
            ConvBNRelu(num_channels, num_channels),
            ConvBNRelu(num_channels, num_channels),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.linear = nn.Linear(num_channels, 1)

    def forward(self, x):
        D = self.discriminator(x)
        D.squeeze_(3).squeeze_(2)
        D = self.linear(D)
        return D


class DoubleConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net_Encoder_Diffusion(nn.Module):
    def __init__(self, inchannel=3, outchannel=3):
        super(U_Net_Encoder_Diffusion, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Conv1 = DoubleConv(inchannel, 16)
        self.Conv2 = DoubleConv(16, 32)
        self.Conv3 = DoubleConv(32, 64)

        self.Up4 = up_conv(64 * 3, 64)
        self.Conv7 = DoubleConv(64 * 3, 64)

        self.Up3 = up_conv(64, 32)
        self.Conv8 = DoubleConv(32 * 2 + 64, 32)

        self.Up2 = up_conv(32, 16)
        self.Conv9 = DoubleConv(16 * 2 + 64, 16)

        self.Conv_1x1 = nn.Conv2d(16, outchannel, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(30, 256)
        self.Conv_message = DoubleConv(1, 64)

    def forward(self, x, watermark):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)

        x6 = self.Globalpool(x4)
        x7 = x6.repeat(1, 1, 4, 4)
        expanded_message = self.linear(watermark)
        expanded_message = expanded_message.view(-1, 1, 16, 16)
        expanded_message = self.Conv_message(expanded_message)
        x4 = torch.cat((x4, x7, expanded_message), dim=1)

        d4 = self.Up4(x4)
        expanded_message = self.linear(watermark)
        expanded_message = expanded_message.view(-1, 1, 16, 16)
        expanded_message = torch.nn.functional.interpolate(expanded_message, size=(d4.shape[2], d4.shape[3]),
                                                           mode='bilinear')
        expanded_message = self.Conv_message(expanded_message)
        d4 = torch.cat((x3, d4, expanded_message), dim=1)
        d4 = self.Conv7(d4)

        d3 = self.Up3(d4)
        expanded_message = self.linear(watermark)
        expanded_message = expanded_message.view(-1, 1, 16, 16)
        expanded_message = torch.nn.functional.interpolate(expanded_message, size=(d3.shape[2], d3.shape[3]),
                                                           mode='bilinear')
        expanded_message = self.Conv_message(expanded_message)
        d3 = torch.cat((x2, d3, expanded_message), dim=1)
        d3 = self.Conv8(d3)

        d2 = self.Up2(d3)
        expanded_message = self.linear(watermark)
        expanded_message = expanded_message.view(-1, 1, 16, 16)
        expanded_message = torch.nn.functional.interpolate(expanded_message, size=(d2.shape[2], d2.shape[3]),
                                                           mode='bilinear')
        expanded_message = self.Conv_message(expanded_message)
        d2 = torch.cat((x1, d2, expanded_message), dim=1)
        d2 = self.Conv9(d2)

        out = self.Conv_1x1(d2)

        return out


def run():
    img = torch.rand(1, 3, 128, 128)
    msg = torch.rand(1, 30)
    encoder = U_Net_Encoder_Diffusion()
    print(encoder(img, msg).shape)
    pass


if __name__ == '__main__':
    run()
