#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : distortion_simulation.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/13 13:26
#
# Import lib here
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.network.gan import cal_gradient_penalty, DefaultUNet, NLayerDiscriminator
from typing import Tuple, Any, List
from omegaconf import DictConfig

from utils import utils
log = utils.get_logger(__name__)


class DistortionSimulation(pl.LightningModule):
    def __init__(self,
                 loss_cfg: DictConfig = None,
                 train_cfg: DictConfig = None,
                 ckpt_path: str = None):
        super().__init__()
        self.automatic_optimization = False

        # model
        self.generator = DefaultUNet(in_channels=3, out_channels=3, use_dropout=train_cfg.use_dropout)
        self.discriminator = NLayerDiscriminator(input_nc=6)
        self.loss_cfg = loss_cfg
        self.train_cfg = train_cfg

        if ckpt_path:
            state_dict = torch.load(ckpt_path)['state_dict']
            self.load_state_dict(state_dict)

    def forward(self, inputs: torch.Tensor):
        # Only used for inference
        self.eval()
        return self.generator(inputs)

    def on_train_start(self) -> None:
        pass

    def training_step(self, batch, batch_idx):
        image, target = batch
        pred = self.generator(image)
        fake_data = torch.cat([image, pred.detach()], dim=1)
        real_data = torch.cat([image, target], dim=1)

        gen_optim, dis_optim = self.optimizers()
        # Train discriminator
        self.set_requires_grad(self.discriminator, True)
        fake_logit = self.discriminator(fake_data)
        real_logit = self.discriminator(real_data)
        if self.train_cfg.gan_type == 'wgan-gp':
            gp, _ = cal_gradient_penalty(self.discriminator, real_data, fake_data)
            dis_loss = torch.mean(fake_logit) - torch.mean(real_logit) + gp
        elif self.train_cfg.gan_type == 'lsgan':
            dis_loss = F.mse_loss(fake_logit, torch.zeros_like(fake_logit)) + \
                       F.mse_loss(real_logit, torch.ones_like(real_logit))
        else:
            raise NotImplementedError
        self.log('train/dis_loss', dis_loss)
        dis_optim.zero_grad()
        self.manual_backward(dis_loss)
        dis_optim.step()

        # Train generator
        self.set_requires_grad(self.discriminator, False)
        fake_logit = self.discriminator(fake_data)
        if self.train_cfg.gan_type == 'wgan-gp':
            gan_loss = -torch.mean(fake_logit) * self.loss_cfg.gan_weight
        elif self.train_cfg.gan_type == 'lsgan':
            gan_loss = F.mse_loss(fake_logit, torch.ones_like(fake_logit)) * self.loss_cfg.gan_weight
        else:
            raise NotImplementedError
        self.log('train/gan_loss', gan_loss)

        pix_loss = F.l1_loss(pred, target) * self.loss_cfg.pix_weight
        self.log('train/pix_loss', pix_loss, prog_bar=True)
        gen_loss = gan_loss + pix_loss
        gen_optim.zero_grad()
        self.manual_backward(gen_loss)
        gen_optim.step()

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        image, target = batch
        pred = self.generator(image)

        for idx, (i_image, i_pred, i_target) in enumerate(zip(image, pred, target)):
            if idx < 3:
                if not self.train_cfg.target2diff:
                    diff1 = torch.abs(i_target - i_image)
                    diff1 = (diff1 - diff1.min()) / (diff1.max() - diff1.min())
                    diff2 = torch.abs(i_pred - i_image)
                    diff2 = (diff2 - diff2.min()) / (diff2.max() - diff2.min())
                    diff3 = torch.abs(i_pred - i_target)
                    diff3 = (diff3 - diff3.min()) / (diff3.max() - diff3.min())
                    show_image = torch.cat([self.de_normalize(i_image), self.de_normalize(i_pred),
                                            self.de_normalize(i_target), diff1, diff2, diff3], dim=-1)
                else:
                    show_image = torch.cat([self.de_normalize(i_image), self.de_normalize(i_pred),
                                            self.de_normalize(i_target)], dim=-1)
                self.logger_instance.add_image(f'valid/example{idx}', show_image, self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=self.train_cfg.lr, weight_decay=1e-5)
        dis_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.train_cfg.lr, weight_decay=1e-5)
        return gen_optim, dis_optim

    def on_test_start(self) -> None:
        pl.seed_everything(42, workers=True)

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def on_test_epoch_end(self) -> None:
        pass

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, List):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def de_normalize(image: torch.Tensor) -> torch.Tensor:
        return (image + 1.) / 2.

    @property
    def total_steps(self):
        return self.trainer.fit_loop.total_batch_idx

    @property
    def logger_instance(self):
        return self.logger.experiment


def run():
    pass


if __name__ == '__main__':
    run()
