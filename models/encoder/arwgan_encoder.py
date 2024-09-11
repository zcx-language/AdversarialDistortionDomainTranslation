#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : arwgan_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/3/3 17:35
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import torch
import torch.nn as nn


class HiDDenConfiguration:

    def __init__(self):
        self.H = 128
        self.W = 128
        self.message_length = 30
        self.encoder_blocks = 4
        self.encoder_channels = 64
        self.use_discriminator = True
        self.use_vgg = False
        self.decoder_blocks = 7
        self.decoder_channels = 64
        self.discriminator_blocks = 3
        self.discriminator_channels = 64
        self.decoder_loss = 1
        self.encoder_loss = 0.7
        self.adversarial_loss = 0.001
        self.enable_fp16 = False


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.relu=nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x,last=False):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        if last:
            return out
        else:
            return torch.cat((x, out), 1)

class Encoder(nn.Module):

    def conv1(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=7, padding=3)

    def conv2(self, in_channel, out_chanenl):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_chanenl,
                         stride=1,
                         kernel_size=3,
                         padding=1)

    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        self.first_layer = nn.Sequential(
            self.conv2(3, self.conv_channels)
        )

        self.second_layer = nn.Sequential(
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.third_layer = nn.Sequential(
            self.conv2(self.conv_channels * 2, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.fourth_layer = nn.Sequential(
            self.conv2(self.conv_channels * 3 + config.message_length, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.Dense_block1 = Bottleneck(self.conv_channels + config.message_length, self.conv_channels)
        self.Dense_block2 = Bottleneck(self.conv_channels * 2 + config.message_length, self.conv_channels)
        self.Dense_block3 = Bottleneck(self.conv_channels * 3 + config.message_length, self.conv_channels)
        self.Dense_block_a1 = Bottleneck(self.conv_channels, self.conv_channels)
        self.Dense_block_a2 = Bottleneck(self.conv_channels * 2, self.conv_channels)
        self.Dense_block_a3 = Bottleneck(self.conv_channels * 3, self.conv_channels)

        self.fivth_layer = nn.Sequential(
            nn.BatchNorm2d(self.conv_channels + config.message_length),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels + config.message_length, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, config.message_length),
        )
        self.sixth_layer = nn.Sequential(
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, config.message_length),
            nn.Softmax(dim=1)
        )
        self.softmax = nn.Sequential(nn.Softmax(dim=1))

        self.final_layer = nn.Sequential(nn.Conv2d(config.message_length, 3, kernel_size=3, padding=1),
                                         )

    def forward(self, image, message):
        H, W = image.size()[2], image.size()[3]

        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1, -1, H, W)

        feature0 = self.first_layer(image)
        feature1 = self.Dense_block1(torch.cat((feature0, expanded_message), 1), last=True)
        feature2 = self.Dense_block2(torch.cat((feature0, expanded_message, feature1), 1), last=True)
        feature3 = self.Dense_block3(torch.cat((feature0, expanded_message, feature1, feature2), 1), last=True)
        feature3 = self.fivth_layer(torch.cat((feature3, expanded_message), 1))
        feature_attention = self.Dense_block_a3(self.Dense_block_a2(self.Dense_block_a1(feature0)), last=True)
        feature_mask = (self.sixth_layer(feature_attention)) * 30
        feature = feature3 * feature_mask
        im_w = self.final_layer(feature)
        im_w = im_w + image
        return im_w


def run():
    img = torch.randn(1, 3, 128, 128)
    message = torch.randn(1, 30)
    config = HiDDenConfiguration()
    encoder = Encoder(config)
    output = encoder(img, message)
    print(output.size())
    pass


if __name__ == '__main__':
    run()
