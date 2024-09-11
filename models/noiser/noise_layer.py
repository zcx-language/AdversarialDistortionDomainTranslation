#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : noise_layer.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/10 13:29
#
# Import lib here
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig


class NoiseLayer:
    def __init__(self, distortions: DictConfig,
                 forward_asl: bool = False,
                 device: str = 'cpu'):
        self.distortions = nn.ModuleDict(distortions).to(device)
        self.forward_asl = forward_asl

    def __call__(self, inputs: torch.Tensor, step: int):
        distortion_types = list(self.distortions.keys())
        distortion_type = random.choice(distortion_types)
        distortion = self.distortions[distortion_type](inputs, step)
        if self.forward_asl:
            distortion = inputs + (distortion - inputs).detach()
        return distortion


def run():
    pass


if __name__ == '__main__':
    run()
