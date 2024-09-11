#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : differentiable_distortion.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/8/19 下午5:19

# Import lib here
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import pytorch_lightning as pl

from models.encoder import StegaStampEncoder
from models.decoder import StegaStampDecoder
from models.discriminator import StegaStampDiscriminator
from models.network.gan import ImagePool, cal_gradient_penalty
from models.noiser.lossy_compression import tensor_lossy_compression
from models.noiser.ps_effects import tensor_ps_effects
from models.noiser.classical_noise_layer import (tensor_gaussian_noise, tensor_resize, tensor_cropout,
                                                 tensor_dropout, tensor_gaussian_blur)
from models.noiser.jpeg_simulator import JpegMask

from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import NoTrainLpips
from kornia.color import yuv_to_rgb, rgb_to_yuv
from metrics.bit_accuracy import BitAccuracy
from omegaconf import DictConfig
from typing import Tuple, List, Any

from utils import utils
log = utils.get_logger(__name__)


# Attention: do not support multiple gpus
class DifferentiableDistortion(pl.LightningModule):
    def __init__(self, img_shape: Tuple[int, int, int],
                 msg_len: int,
                 distortion_types: List[str],
                 loss_cfg: DictConfig,
                 min_psnr: int = 30,   # make sure that the saved ckpts psnr > min_psnr.
                 lr: float = 1e-3,
                 enable_gan: bool = True,
                 forward_asl: bool = False):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # model
        self.encoder = StegaStampEncoder(img_shape, msg_len)
        self.decoder = StegaStampDecoder(img_shape, msg_len)
        self.discriminator = StegaStampDiscriminator(in_channels=3)
        self.image_pool = ImagePool()

        n_distortion = len(distortion_types)
        # metric
        self.train_img_metric = MetricCollection([
            PeakSignalNoiseRatio(data_range=2.),
            StructuralSimilarityIndexMeasure(data_range=2.),
            # LearnedPerceptualImagePatchSimilarity()
        ])
        self.train_msg_metric = BitAccuracy()
        self.train_tmsg_metrics = nn.ModuleList([BitAccuracy() for _ in range(n_distortion)])

        self.valid_img_metric = MetricCollection([
            PeakSignalNoiseRatio(data_range=2.),
            StructuralSimilarityIndexMeasure(data_range=2.),
        ])
        self.valid_msg_metric = BitAccuracy()
        self.valid_tmsg_metrics = nn.ModuleList([BitAccuracy() for _ in range(n_distortion)])

        self.test_img_metric = MetricCollection([
            PeakSignalNoiseRatio(data_range=2.),
            StructuralSimilarityIndexMeasure(data_range=2.),
            # LearnedPerceptualImagePatchSimilarity(compute_on_step=False),
        ])
        self.test_msg_metric = BitAccuracy()
        self.test_tmsg_metrics = nn.ModuleList([BitAccuracy() for _ in range(n_distortion)])

        # func
        self.image_norm = Normalize(0.5, 0.5)
        self.image_denorm = Normalize(-1., 2.)
        self.lpips_loss = NoTrainLpips()

        # JPEG
        self.JPEG_Mask = JpegMask()

    def encode(self, img, msg):
        """Only for prediction, do not use this in train phase."""
        pass

    def decode(self, img):
        """Only for prediction, do not use this in train phase."""
        pass

    def get_true_distortion(self, image, container, distortion_type: str, quality_factor: int = 50):
        container = self.image_denorm(container)

        if distortion_type == 'identity':
            t_distortion = container
        elif distortion_type == 'jpeg':
            # t_distortion = tensor_lossy_compression(container, quality_factor, ext='.jpg')
            # t_distortion = container + (t_distortion - container).detach()
            t_distortion = self.JPEG_Mask(container)
        elif distortion_type == 'jpeg2000':
            t_distortion = tensor_lossy_compression(container, quality_factor, ext='.jp2')
        elif distortion_type == 'webp':
            t_distortion = tensor_lossy_compression(container, quality_factor, ext='.webp')
        elif distortion_type == 'rainy':
            # Enlarge container for better visual
            container = F.interpolate(container, (256, 256))
            t_distortion = tensor_ps_effects(container, 'rainy')
            t_distortion = F.interpolate(t_distortion, (128, 128))
        elif distortion_type == 'emboss':
            t_distortion = tensor_ps_effects(container, 'emboss')
        elif distortion_type == 'fisheye':
            t_distortion = tensor_ps_effects(container, 'fisheye')
        elif distortion_type == 'solarize':
            t_distortion = tensor_ps_effects(container, 'solarize')
        elif distortion_type == 'gaussian_blur':
            t_distortion = tensor_gaussian_blur(container)
        elif distortion_type == 'gaussian_noise':
            t_distortion = tensor_gaussian_noise(container)
        elif distortion_type == 'resize':
            t_distortion = tensor_resize(container)
        elif distortion_type == 'dropout':
            t_distortion = tensor_dropout(self.image_denorm(image), container)
        elif distortion_type == 'cropout':
            t_distortion = tensor_cropout(self.image_denorm(image), container)
        else:
            raise ValueError
        return self.image_norm(t_distortion)

    def cal_discriminator_loss(self, img, container):
        real_pred = self.discriminator(img)
        container = self.image_pool.push_and_pop(container.detach())
        fake_pred = self.discriminator(container)
        gradient_penalty = cal_gradient_penalty(self.discriminator, img, container)[0]
        loss = -torch.mean(real_pred) + torch.mean(fake_pred) + gradient_penalty
        self.log('train/wgan_dis_loss', loss)
        return loss

    def cal_encoder_decoder_loss(self, img, container, distortions, msg):
        # Generator loss
        gan_loss = -torch.mean(self.discriminator(container)) if self.hparams.enable_gan else 0.

        # vis loss
        img_yuv = rgb_to_yuv(self.image_denorm(img))
        container_yuv = rgb_to_yuv(self.image_denorm(container))
        yuv_scale = torch.tensor([1., 1., 1.], device=img.device)
        l1_loss = F.l1_loss(img_yuv, container_yuv, reduction='none')
        l1_loss = torch.dot(torch.mean(l1_loss, dim=[0, 2, 3]), yuv_scale)
        lpips_loss = torch.mean(self.lpips_loss(img, container))
        vis_loss = l1_loss + lpips_loss

        # Msg loss
        hat_msg_logits = self.decoder(container)
        msg_loss = F.binary_cross_entropy_with_logits(hat_msg_logits, msg)

        with torch.no_grad():
            train_img_metric = self.train_img_metric(container, img)
            hat_msg = torch.round(torch.sigmoid(hat_msg_logits))
            train_msg_metric = self.train_msg_metric(hat_msg, msg)

        self.log('train/psnr', train_img_metric['PeakSignalNoiseRatio'])
        self.log('train/ssim', train_img_metric['StructuralSimilarityIndexMeasure'])
        self.log('train/bit_acc', train_msg_metric)

        if len(distortions) != 0:
            # Random sample a distortion for train
            idx = random.randint(0, len(distortions)-1)
            distortion_type = self.hparams.distortion_types[idx]
            distortion = distortions[idx]

            if self.hparams.forward_asl == True:
                distortion = container + (distortion - container).detach()

            hat_tmsg_logits = self.decoder(distortion)
            tmsg_loss = F.binary_cross_entropy_with_logits(hat_tmsg_logits, msg)
            msg_loss = msg_loss + tmsg_loss
            self.log(f'train/{distortion_type}_wgan_tmsg_loss', tmsg_loss)

            with torch.no_grad():
                hat_tmsg = torch.round(torch.sigmoid(hat_tmsg_logits))
                train_tmsg_metric = self.train_tmsg_metrics[idx](hat_tmsg, msg)
            self.log(f'train/{distortion_type}_tbit_acc', train_tmsg_metric, prog_bar=True)

        loss_cfg = self.hparams.loss_cfg.wgan.generator
        loss = (loss_cfg.vis_loss_scale * vis_loss +
                loss_cfg.msg_loss_scale * msg_loss +
                loss_cfg.gan_loss_scale * gan_loss)

        self.log('train/wgan_gen_loss', gan_loss)
        self.log('train/wgan_lpips_loss', lpips_loss)
        self.log('train/wgan_l1_loss', l1_loss)
        self.log('train/wgan_msg_loss', msg_loss)
        self.log('train/wgan_loss', loss)
        return loss

    def on_train_start(self) -> None:
        # Show metrics in tensorboard
        # self.logger.log_hyperparams(self.hparams, {"hp/tbit_acc": 0, "hp/psnr": 0, "hp/ssim": 0})
        metrics_dict = {"hp/bit_acc": 0, "hp/tbit_acc": 0, "hp/psnr": 0, "hp/ssim": 0}
        for distortion_type in self.hparams.distortion_types:
            metrics_dict.update({f"hp/{distortion_type}_tbit_acc": 0})
        self.logger.log_metrics(metrics_dict)

    def training_step(self, batch, batch_idx, optimizer_idx):
        img, msg = batch
        container = self.encoder(img, msg)

        total_batch_idx = self.trainer.fit_loop.total_batch_idx
        if total_batch_idx % 500 == 0:
            self.logger.experiment.add_image('train/image', self.image_denorm(img[0]), total_batch_idx)
            self.logger.experiment.add_image('train/container', self.image_denorm(container[0]), total_batch_idx)

        t_distortions = []
        for distortion_type in self.hparams.distortion_types:
            t_distortion = self.get_true_distortion(img, container, distortion_type, quality_factor=50)
            t_distortions.append(t_distortion)

            if total_batch_idx % 500 == 0:
                self.logger.experiment.add_image(f'train/t_{distortion_type}',
                                                 self.image_denorm(t_distortion[0]), total_batch_idx)

        # Train Discriminator
        if optimizer_idx == 0:
            return self.cal_discriminator_loss(img, container)
        # Train En/Decoder
        elif optimizer_idx == 1:
            return self.cal_encoder_decoder_loss(img, container, t_distortions, msg)

    def training_epoch_end(self, outputs: Any) -> None:
        self.train_img_metric.reset()
        self.train_msg_metric.reset()
        for idx in range(len(self.hparams.distortion_types)):
            self.train_tmsg_metrics[idx].reset()

    def validation_step(self, batch, batch_idx):
        img, msg = batch
        container = self.encoder(img, msg)
        hat_msg_logits = self.decoder(container)
        self.valid_img_metric(container, img)
        hat_msg = torch.round(torch.sigmoid(hat_msg_logits))
        self.valid_msg_metric(hat_msg, msg)

        if self.trainer.is_last_batch:
            self.logger.experiment.add_image(f'valid/image', self.image_denorm(img[0]), self.current_epoch)
            self.logger.experiment.add_image('valid/container', self.image_denorm(container[0]), self.current_epoch)

        for idx, distortion_type in enumerate(self.hparams.distortion_types):
            quality_factor = random.randint(10, 95)
            distortion = self.get_true_distortion(img, container, distortion_type, quality_factor)
            hat_tmsg_logits = self.decoder(distortion)

            if self.trainer.is_last_batch:
                self.logger.experiment.add_image(f'valid/t_{distortion_type}',
                                                 self.image_denorm(distortion[0]), self.current_epoch)

            hat_tmsg = torch.round(torch.sigmoid(hat_tmsg_logits))
            self.valid_tmsg_metrics[idx](hat_tmsg, msg)

    def validation_epoch_end(self, outputs: Any) -> None:
        valid_img_metric = self.valid_img_metric.compute()
        valid_msg_metric = self.valid_msg_metric.compute()
        self.valid_img_metric.reset()
        self.valid_msg_metric.reset()

        self.logger.experiment.add_scalar('valid/psnr',
                                          valid_img_metric['PeakSignalNoiseRatio'], self.current_epoch)
        self.logger.experiment.add_scalar('valid/ssim', valid_img_metric['StructuralSimilarityIndexMeasure'], self.current_epoch)
        self.logger.experiment.add_scalar('valid/bit_acc', valid_msg_metric, self.current_epoch)

        log_str = f'Epoch:{self.current_epoch}, psnr:{valid_img_metric["PeakSignalNoiseRatio"]}'
        valid_tmsg_metrics = []
        for idx, distortion_type in enumerate(self.hparams.distortion_types):
            valid_tmsg_metric = self.valid_tmsg_metrics[idx].compute()
            valid_tmsg_metrics.append(valid_tmsg_metric)
            self.valid_tmsg_metrics[idx].reset()
            log_str += f', {distortion_type}_tbit_acc:{valid_tmsg_metric}'

            self.logger.experiment.add_scalar(f'valid/{distortion_type}_tbit_acc',
                                              valid_tmsg_metric, self.current_epoch)
        avg_tbit_acc = sum(valid_tmsg_metrics) / len(valid_tmsg_metrics)
        log.info(log_str + f', tbit_acc:{avg_tbit_acc}')
        self.logger.experiment.add_scalar('valid/tbit_acc', avg_tbit_acc, self.current_epoch)

        # Used for model checkpoint callback, only save ckpts that PSNR > 33.
        if valid_img_metric['PeakSignalNoiseRatio'] > self.hparams.min_psnr:
            avg_tbit_acc = avg_tbit_acc
        else:
            avg_tbit_acc = 0.0
        self.log('tbit_acc', avg_tbit_acc, prog_bar=True, logger=False)
        self.log('psnr', valid_img_metric['PeakSignalNoiseRatio'], logger=False)

        # Show metrics in tensorboard
        if self.current_epoch + 1 == self.trainer.max_epochs:
            self.logger.experiment.add_scalar('hp/bit_acc', valid_msg_metric, self.current_epoch)
            self.logger.experiment.add_scalar('hp/tbit_acc', avg_tbit_acc, self.current_epoch)
            self.logger.experiment.add_scalar('hp/psnr',
                                              valid_img_metric['PeakSignalNoiseRatio'], self.current_epoch)
            self.logger.experiment.add_scalar('hp/ssim',
                                              valid_img_metric['StructuralSimilarityIndexMeasure'], self.current_epoch)

            for idx, distortion_type in enumerate(self.hparams.distortion_types):
                self.logger.experiment.add_scalar(f'hp/{distortion_type}_tbit_acc', valid_tmsg_metrics[idx])

    def configure_optimizers(self):
        enc_dec_optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], lr=self.hparams.lr, weight_decay=1e-5)

        discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr, weight_decay=1e-5)

        return [enc_dec_optim, discriminator_optim]

    def test_step(self, batch: Any, batch_idx: int):
        img, msg = batch
        container = self.encoder(img, msg)
        hat_msg_logits = self.decoder(container)
        hat_msg = torch.round(torch.sigmoid(hat_msg_logits))
        self.test_img_metric(container, img)
        self.test_msg_metric(hat_msg, msg)

        # if self.trainer.is_last_batch:
        #     self.logger.experiment.add_image(f'valid/image', self.image_denorm(img[0]), self.current_epoch)
        #     self.logger.experiment.add_image('valid/container', self.image_denorm(container[0]), self.current_epoch)

        for idx, distortion_type in enumerate(self.hparams.distortion_types):
            quality_factor = random.randint(10, 95)
            distortion = self.get_true_distortion(img, container, distortion_type, quality_factor)
            hat_tmsg_logits = self.decoder(distortion)
            hat_tmsg = torch.round(torch.sigmoid(hat_tmsg_logits))
            self.test_tmsg_metrics[idx](hat_tmsg, msg)

            self.logger.experiment.add_image(f'test/container',
                                             self.image_denorm(container[0]), batch_idx)
            self.logger.experiment.add_image(f'test/{distortion_type}_container',
                                             self.image_denorm(distortion[0]), batch_idx)

    def test_epoch_end(self, outputs) -> None:
        test_img_metric = self.test_img_metric.compute()
        test_msg_metric = self.test_msg_metric.compute()
        self.test_img_metric.reset()
        self.test_msg_metric.reset()
        self.logger.experiment.add_scalar('test/psnr', test_img_metric['PeakSignalNoiseRatio'], self.current_epoch)
        self.logger.experiment.add_scalar('test/ssim',
                                          test_img_metric['StructuralSimilarityIndexMeasure'], self.current_epoch)
        self.logger.experiment.add_scalar('test/bit_acc', test_msg_metric, self.current_epoch)

        test_tmsg_metrics = []
        for idx, distortion_type in enumerate(self.hparams.distortion_types):
            test_tmsg_metric = self.test_tmsg_metrics[idx].compute()
            self.test_tmsg_metrics[idx].reset()
            test_tmsg_metrics.append(test_tmsg_metric)
            self.logger.experiment.add_scalar(f'test/{distortion_type}_tbit_acc', test_tmsg_metric, self.current_epoch)

        pass


def run():
    pass


if __name__ == '__main__':
    run()
