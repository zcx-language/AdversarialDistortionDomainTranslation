#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : stegastamp_discriminator.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/5/9 下午5:10

# Import lib here
import torch.nn as nn
from models.network.gan import NLayerDiscriminator


class StegaStampDiscriminator(nn.Module):
    def __init__(self,
                 in_channels: int):
        super().__init__()
        self.model = NLayerDiscriminator(in_channels)

    def forward(self, image):
        return self.model(image)


def run():
    pass


if __name__ == '__main__':
    run()
