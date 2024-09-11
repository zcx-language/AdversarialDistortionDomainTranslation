#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : general_end.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/9 21:15
#
# Import lib here
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, seed_everything
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.classification import MultilabelAccuracy
from models.network.gan import DiscriminatorLoss
from lpips.lpips import LPIPS

from omegaconf import DictConfig
from typing import Tuple, List, Any, Callable
from utils import utils

log = utils.get_logger(__name__)
DISCRIMINATOR_LOSS = DiscriminatorLoss(input_nc=3, loss_mode='lsgan', image_pool_size=0)    # Not used
LPIPS_LOSS = LPIPS(pretrained=True, net='vgg')


class GeneralEND(LightningModule):
    def __init__(self, encoder: nn.Module,
                 noise_layer: Callable,
                 decoder: nn.Module,
                 test_distortions: DictConfig,
                 loss_cfg: DictConfig,
                 train_cfg: DictConfig):
        super().__init__()
        torch.set_float32_matmul_precision('high')
        self.automatic_optimization = False

        # model
        self.encoder = encoder
        self.noiser_layer = noise_layer
        self.decoder = decoder
        self.loss_cfg = loss_cfg
        self.test_distortions = nn.ModuleDict(test_distortions)
        self.train_cfg = train_cfg
        msg_len = self.encoder.msg_len
        num_distortions = len(test_distortions)

        # metric
        self.train_img_metric = MetricCollection([
            PeakSignalNoiseRatio(),
            StructuralSimilarityIndexMeasure()
        ])
        self.train_tmsg_metric = MultilabelAccuracy(msg_len)
        self.valid_tmsg_metric = MultilabelAccuracy(msg_len)

        self.test_img_metric = MetricCollection([
            PeakSignalNoiseRatio(),
            StructuralSimilarityIndexMeasure(),
            # LearnedPerceptualImagePatchSimilarity(compute_on_step=False),
        ])
        self.test_msg_metric = MultilabelAccuracy(msg_len)
        self.test_tmsg_metrics = nn.ModuleList([MultilabelAccuracy(msg_len) for _ in range(num_distortions)])

    def encode(self, img, msg):
        """Only for prediction, do not use this in train phase."""
        pass

    def decode(self, img):
        """Only for prediction, do not use this in train phase."""
        pass

    def on_train_start(self) -> None:
        # clear the cache of validation metrics due to the sanity checks
        self.valid_tmsg_metric.reset()

        # Using global variable to avoid saving model in pickle
        global DISCRIMINATOR_LOSS, LPIPS_LOSS
        DISCRIMINATOR_LOSS = DISCRIMINATOR_LOSS.to(self.device)
        LPIPS_LOSS = LPIPS_LOSS.to(self.device)

    def training_step(self, batch, batch_idx):
        img, msg = batch
        container = self.encoder(img, msg)
        # container = self.clip_container_psnr(container, img, psnr=self.train_cfg.base_psnr)
        distortion = self.noiser_layer(container)
        tmsg_logit = self.decoder(distortion)

        # Update encoder and decoder parameters
        if self.loss_cfg.dis_weight:
            dis_loss = DISCRIMINATOR_LOSS(container, img) * self.loss_cfg.dis_weight
            self.log('train/dis_loss', dis_loss)
        else:
            dis_loss = torch.tensor(0., device=self.device)

        if self.loss_cfg.lpips_weight:
            # The lpips return a tensor with shape (batch_size, 1, 1, 1)
            lpips_loss = LPIPS_LOSS(container, img).mean() * self.loss_cfg.lpips_weight
            self.log('train/lpips_loss', lpips_loss)
        else:
            lpips_loss = torch.tensor(0., device=self.device)

        pix_loss = F.mse_loss(container, img) * self.loss_cfg.pix_weight
        self.log('train/pix_loss', pix_loss)

        tmsg_loss = F.binary_cross_entropy_with_logits(tmsg_logit, msg) * self.loss_cfg.msg_weight
        self.log('train/tmsg_loss', tmsg_loss)

        loss = dis_loss + lpips_loss + pix_loss + tmsg_loss
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if self.total_steps % 3000 == 0:
            show_imgs = [
                self.image_denorm(img[0]), self.image_denorm(container[0]),
                self.image_denorm(container[0] - img[0], mode='min_max'),
                self.image_denorm(distortion[0])
            ]
            self.logger_instance.add_image('train/example', torch.cat(show_imgs, dim=-1), self.total_steps)

        # Los metric
        with torch.no_grad():
            train_img_metric = self.train_img_metric(container, img)
            tbit_acc = self.train_tmsg_metric(tmsg_logit.sigmoid(), msg)
            self.log('train/psnr', train_img_metric['PeakSignalNoiseRatio'].item())
            self.log('train/ssim', train_img_metric['StructuralSimilarityIndexMeasure'].item())
            self.log('train/tbit_acc', tbit_acc.item())

    def on_train_epoch_end(self) -> None:
        self.train_img_metric.reset()
        self.train_tmsg_metric.reset()

    def validation_step(self, batch, batch_idx):
        host, msg = batch
        with torch.no_grad():
            container = self.encoder(host, msg)
            for psnr_level in [35]:
                clip_container = self.clip_container_psnr(container, host, psnr_level)
                for idx, d_type in enumerate(self.test_distortions.keys()):
                    distorted_container = self.test_distortions[d_type](clip_container)
                    tmsg_logit = self.decoder(distorted_container)
                    self.valid_tmsg_metric(tmsg_logit.sigmoid(), msg)

    def on_validation_epoch_end(self) -> None:
        tbit_acc = self.valid_tmsg_metric.compute()
        self.valid_tmsg_metric.reset()
        self.logger_instance.add_scalar(f'valid/tbit_acc', tbit_acc, self.current_epoch)

        # Used to select the best checkpoint in ModelCheckpoint callback
        self.log('tbit_acc', tbit_acc, logger=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], lr=self.train_cfg.lr, weight_decay=1e-5)
        return optimizer

    def on_test_start(self) -> None:
        seed_everything(42, workers=True)

    def test_step(self, batch: Any, batch_idx: int):
        img, msg = batch
        with torch.no_grad():
            container = self.encoder(img, msg)
            self.test_img_metric.update(container, img)
            msg_logit = self.decoder(container)
            self.test_msg_metric.update(msg_logit.sigmoid(), msg)

            for psnr_level in [35]:
                clip_container = self.clip_container_psnr(container, img, psnr=psnr_level)
                for idx, d_type in enumerate(self.test_distortions.keys()):
                    distortion = self.test_distortions[d_type](clip_container)
                    tmsg_logit = self.decoder(distortion)
                    self.test_tmsg_metrics[idx].update(tmsg_logit.sigmoid(), msg)

        show_imgs = [self.image_denorm(img[0]), self.image_denorm(container[0]),
                     self.image_denorm(container[0] - img[0], mode='min_max')]
        self.logger_instance.add_image('test/example', torch.cat(show_imgs, dim=-1), batch_idx)

    def on_test_epoch_end(self) -> None:
        test_img_metric = self.test_img_metric.compute()
        psnr = test_img_metric['PeakSignalNoiseRatio'].item()
        ssim = test_img_metric['StructuralSimilarityIndexMeasure'].item()
        bit_acc = self.test_msg_metric.compute().item()
        self.test_img_metric.reset()
        self.test_msg_metric.reset()

        self.logger_instance.add_scalar('test/psnr', psnr, self.current_epoch)
        self.logger_instance.add_scalar('test/ssim', ssim, self.current_epoch)
        self.logger_instance.add_scalar('test/bit_acc', bit_acc, self.current_epoch)

        log_str = f'Test results, psnr: {psnr:.6f}, ssim: {ssim:.6f}, bit_acc: {bit_acc:.6f}\n'
        tbit_acc_list = []
        for idx, distortion_type in enumerate(list(self.test_distortions.keys())):
            tbit_acc = self.test_tmsg_metrics[idx].compute().item()
            self.test_tmsg_metrics[idx].reset()
            tbit_acc_list.append(tbit_acc)
            log_str += f'{distortion_type}_tbit_acc: {tbit_acc:.6f}\n'
            self.logger_instance.add_scalar(f'test/{distortion_type}_tbit_acc', tbit_acc, self.current_epoch)
        avg_tbit_acc = sum(tbit_acc_list) / len(tbit_acc_list)
        log.info(log_str + f'avg_tbit_acc: {avg_tbit_acc:.6f}')
        self.logger_instance.add_scalar('test/avg_tbit_acc', avg_tbit_acc, self.current_epoch)

    @staticmethod
    def clip_container_psnr(container: torch.tensor, host: torch.Tensor, psnr: float = 35) -> torch.Tensor:
        """Clip the container to make sure its PSNR is round to 35 dB.

        According to the formula of PSNR, we can get:
            PSNR = 10 * log10(255^2 / MSE)
        where MSE is the mean squared error between the container and host.
        So, when the PSNR is set to `psnr`, we can get:
            MSE = 255^2 / 10^(psnr / 10)
        Then, we use a factor to scale the residual:
            factor = sqrt(MSE / mean(residual^2))
        Finally,
            clipped_container = host + residual * factor
        """
        data_range = host.max() - host.min()
        target_mse = data_range ** 2 / 10 ** (psnr / 10)
        residual = container - host
        factor = torch.sqrt(target_mse / torch.mean(residual ** 2))
        return (host + residual * factor).clamp(-1, 1)

    @staticmethod
    def image_denorm(image: torch.Tensor, mode: str = 'default') -> torch.Tensor:
        if mode == 'default':
            return (image + 1.) / 2.
        elif mode == 'min_max':
            return (image - image.min()) / (image.max() - image.min())
        else:
            raise ValueError

    @property
    def total_steps(self):
        return self.trainer.fit_loop.total_batch_idx

    @property
    def logger_instance(self):
        return self.logger.experiment


def run():
    general_end = GeneralEND(None, None, None, None, None, None)
    pass


if __name__ == '__main__':
    run()
