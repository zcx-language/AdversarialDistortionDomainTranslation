#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : da_distortion_generator.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/6/18 下午1:28

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.layers import  SigmoidRange
from typing import Callable, Tuple


class DADistortionGenerator(nn.Module):
    def __init__(self, in_channels, n_iter: int = 10):
        super(DADistortionGenerator, self).__init__()
        self.n_iter = n_iter

        self.generator = nn.Sequential(
            nn.Conv2d(in_channels, 16, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 3, (3, 3), (1, 1), (1, 1)),
            SigmoidRange(-1, 1)
        )

        self.optimizer = torch.optim.Adam(self.generator.parameters())

    def forward(self,
                image: torch.Tensor,
                msg: torch.Tensor,
                model: nn.Module):

        image = image.detach()
        for _ in range(self.n_iter):
            # self.decode(container + residual)
            adv_image = self.generator(image)
            msg_hat_logits = model(image)
            loss = F.mse_loss(adv_image, image) * 5. - F.binary_cross_entropy_with_logits(msg_hat_logits, msg)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self.generator(image)


def run():
    pass


if __name__ == '__main__':
    run()
