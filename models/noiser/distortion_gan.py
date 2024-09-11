#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : distortion_gan.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/5/15 下午10:29

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.layers import ConvLayer
from fastai.vision.models.unet import DynamicUnet
from models.network.gan import DefaultUNet, NLayerDiscriminator

from omegaconf import DictConfig
from typing import Tuple, List, Optional, Dict


class Generator(nn.Module):
    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 out_channels: int,
                 out_range: Tuple[float, float]):
        super().__init__()
        in_channels = image_shape[0]
        image_size = image_shape[1:]

        encoder = nn.Sequential(
            ConvLayer(in_channels, 32, ks=4, stride=2, padding=1),
            ConvLayer(32, 64, ks=4, stride=2, padding=1),
            ConvLayer(64, 128, ks=4, stride=2, padding=1),
            ConvLayer(128, 128, ks=4, stride=2, padding=1),
            ConvLayer(128, 128, ks=4, stride=2, padding=1),
        )
        self.generator = DynamicUnet(encoder, out_channels, image_size, y_range=out_range)

    def forward(self, x):
        return self.generator(x)


# class DistortionGAN(nn.Module):
#     def __init__(self,
#                  image_shape: Tuple[int, int, int],
#                  g_out_channels: int,
#                  loss_cfg: DictConfig):
#         super().__init__()
#
#         self.generator = Generator(image_shape, g_out_channels)
#         self.discriminator_loss = DiscriminatorLoss(g_out_channels)
#         self.loss_cfg = loss_cfg
#         self.optimizer = torch.optim.Adam(self.generator.parameters())
#
#     def optimize_step(self, image: torch.Tensor, target: torch.Tensor):
#         pred = self.generator(image)
#         dis_loss = self.discriminator_loss(pred, target) * self.loss_cfg.dis_weight
#         l1_loss = F.l1_loss(pred, target) * self.loss_cfg.pix_weight
#         loss = dis_loss + l1_loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return {dis_loss: dis_loss.item(), l1_loss: l1_loss.item()}
#
#     # def forward(self):
#     #     raise NotImplementedError
#
#     def forward(self, image: torch.Tensor):
#         return self.generator(image)


class DistortionGAN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 forwardasl: bool = False, loss_cfg: Optional[Dict] = None):
        super().__init__()
        self.generator = DefaultUNet(in_channels, out_channels)
        self.discriminator = NLayerDiscriminator(in_channels+out_channels, norm_layer=nn.InstanceNorm2d)
        self.forwardasl = forwardasl
        self.loss_cfg = loss_cfg

    def distort(self, image: torch.Tensor):
        distorted = self.generator(image)
        if self.forwardasl:
            distorted = image + (distorted - image).detach()
        return distorted


def run():
    pass


if __name__ == '__main__':
    run()
