#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : adversarial_simulate_jpeg.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2022/3/30 下午3:04

# Import lib here
import torch
import random
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.transforms import Normalize
from torchmetrics import MetricCollection
from torchmetrics.image import (PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure)
from torchmetrics.image.lpip import NoTrainLpips
from metrics import BitAccuracy
from kornia.color import rgb_to_yuv

from omegaconf import DictConfig
from models.encoder import StegaStampEncoder
from models.decoder import StegaStampDecoder
from models.discriminator import StegaStampDiscriminator
from models.noiser.jpeg_simulator import JpegSS, JpegMask
from models.noiser.distortion_gan import DistortionGAN
from models.network.gan import ImagePool, cal_gradient_penalty
from utils.image_tools import image_compression_by_cv2
from typing import Any, Tuple


class AdversarialSimulateJPEG(pl.LightningModule):
    def __init__(self, img_shape: Tuple[int, int, int],
                 msg_len: int,
                 simulate_type: str,
                 enable_gan: bool,
                 loss_cfg: DictConfig,
                 lr: float,
                 forward_asl: bool = False):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.encoder = StegaStampEncoder(img_shape, msg_len)
        self.decoder = StegaStampDecoder(img_shape, msg_len)
        self.discriminator = StegaStampDiscriminator(in_channels=3)
        self.image_pool = ImagePool()

        if self.hparams.simulate_type == 'advgan':
            self.jpeg_gan = DistortionGAN(img_shape, g_out_channels=3, d_in_channels=6)

        self.train_img_metric = MetricCollection([
            PeakSignalNoiseRatio(data_range=2.),
            StructuralSimilarityIndexMeasure(data_range=2.),
            # LearnedPerceptualImagePatchSimilarity()
        ])
        self.train_msg_metric = BitAccuracy()
        self.train_tmsg_metric = BitAccuracy()

        self.valid_img_metric = MetricCollection([
            PeakSignalNoiseRatio(data_range=2.),
            StructuralSimilarityIndexMeasure(data_range=2.),
            # LearnedPerceptualImagePatchSimilarity(compute_on_step=False),
        ])
        self.valid_msg_metric = BitAccuracy()
        self.valid_tmsg_metric = BitAccuracy()
        self.image_norm = Normalize(0.5, 0.5)
        self.image_denorm = Normalize(-1., 2.)
        self.lpips_loss = NoTrainLpips()

    def encode(self, img, msg):
        """Only for prediction, do not use this in train phase."""
        pass

    def decode(self, img):
        """Only for prediction, do not use this in train phase."""
        pass

    def cal_discriminator_loss(self, img, container):
        real_pred = self.discriminator(img)
        container = self.image_pool.push_and_pop(container.detach())
        fake_pred = self.discriminator(container)
        gradient_penalty = cal_gradient_penalty(self.discriminator, img, container)[0]
        loss = -torch.mean(real_pred) + torch.mean(fake_pred) + gradient_penalty
        self.log('train/dis_loss', loss)
        return loss

    def cal_encoder_decoder_loss(self, img, container, jpeg_container, msg):
        hat_msg_logits = self.decoder(container)
        hat_tmsg_logits = self.decoder(jpeg_container)

        # Img loss
        g_loss = -torch.mean(self.discriminator(container)) if self.hparams.enable_gan else 0.

        img_yuv = rgb_to_yuv(self.image_denorm(img))
        container_yuv = rgb_to_yuv(self.image_denorm(container))
        yuv_scale = torch.tensor([1., 0.5, 0.5], device=img.device)
        l1_loss = F.l1_loss(img_yuv, container_yuv, reduction='none')
        l1_loss = torch.dot(torch.mean(l1_loss, dim=[0, 2, 3]), yuv_scale)

        lpips_loss = torch.mean(self.lpips_loss(img, container))

        # Msg loss
        msg_loss = F.binary_cross_entropy_with_logits(hat_msg_logits, msg)
        tmsg_loss = F.binary_cross_entropy_with_logits(hat_tmsg_logits, msg)

        loss_cfg = self.hparams.loss_cfg.encoder_decoder
        loss = (loss_cfg.l1_loss_scale * l1_loss +
                loss_cfg.g_loss_scale * g_loss +
                loss_cfg.lpips_loss_scale * lpips_loss +
                loss_cfg.msg_loss_scale * msg_loss +
                loss_cfg.tmsg_loss_scale * tmsg_loss)

        self.log('train/l2_loss', l1_loss)
        self.log('train/g_loss', g_loss)
        self.log('train/lpips_loss', lpips_loss)
        self.log('train/msg_loss', msg_loss)
        self.log('train/tmsg_loss', tmsg_loss)
        self.log('train/loss', loss)

        with torch.no_grad():
            train_img_metric = self.train_img_metric(container, img)
            hat_msg = torch.round(torch.sigmoid(hat_msg_logits))
            hat_tmsg = torch.round(torch.sigmoid(hat_tmsg_logits))
            train_msg_metric = self.train_msg_metric(hat_msg, msg)
            train_tmsg_metric = self.train_tmsg_metric(hat_tmsg, msg)

        self.log('train/psnr', train_img_metric['PeakSignalNoiseRatio'])
        self.log('train/ssim', train_img_metric['StructuralSimilarityIndexMeasure'])
        self.log('train/bit_acc', train_msg_metric)
        self.log('train/tbit_acc', train_tmsg_metric, prog_bar=True)
        return loss

    def simulate_training_step(self, batch, batch_idx, optimizer_idx):
        img, msg = batch
        container = self.encoder(img, msg)
        random_quality = random.randint(50, 95)
        t_jpeg = self.image_norm(image_compression_by_cv2(container, quality=random_quality))

        if self.hparams.simulate_type == 'hidden':
            jpeg_op = JpegMask()
            s_jpeg = jpeg_op(container)
        elif self.hparams.simulate_type == 'stegastamp':
            jpeg_op = JpegSS(Q=random_quality)
            s_jpeg = jpeg_op(container)
        elif self.hparams.simulate_type == 'forward_asl':
            s_jpeg = container + (t_jpeg - container).detach()
        else:
            raise NotImplementedError

        if self.global_step % 200 == 0:
            self.logger.experiment.add_image('train/image', self.image_denorm(img[0]), self.global_step)
            self.logger.experiment.add_image('train/container', self.image_denorm(container[0]), self.global_step)
            self.logger.experiment.add_image('train/s_jpeg', self.image_denorm(s_jpeg[0]), self.global_step)
            self.logger.experiment.add_image('train/t_jpeg', self.image_denorm(t_jpeg[0]), self.global_step)

        # Train Discriminator
        if optimizer_idx == 0:
            return self.cal_discriminator_loss(img, container)
        # Train En/Decoder
        elif optimizer_idx == 1:
            return self.cal_encoder_decoder_loss(img, container, s_jpeg, msg)

    def adversarial_simulate_training_step(self, batch, batch_idx, optimizer_idx):
        img, msg = batch
        container = self.encoder(img, msg)
        g_jpeg = self.jpeg_gan.generator(container)
        t_jpeg = self.image_norm(image_compression_by_cv2(container, quality=50))

        if self.global_step % 200 == 0:
            self.logger.experiment.add_image('train/image', self.image_denorm(img[0]), self.global_step)
            self.logger.experiment.add_image('train/container', self.image_denorm(container[0]), self.global_step)
            self.logger.experiment.add_image('train/g_jpeg', self.image_denorm(g_jpeg[0]), self.global_step)
            self.logger.experiment.add_image('train/t_jpeg', self.image_denorm(t_jpeg[0]), self.global_step)

        # Train JPEG GAN discriminator
        if optimizer_idx == 0:
            real_pair = torch.cat([container, t_jpeg], dim=1).detach()
            fake_pair = torch.cat([container, g_jpeg], dim=1).detach()
            fake_pair = self.jpeg_gan.gen_pool.push_and_pop(fake_pair)

            real_pred = self.jpeg_gan.discriminator(real_pair)
            fake_pred = self.jpeg_gan.discriminator(fake_pair)
            gradient_penalty = cal_gradient_penalty(self.jpeg_gan.discriminator, real_pair, fake_pair)[0]
            loss = -torch.mean(real_pred) + torch.mean(fake_pred) + gradient_penalty
            self.log('train/gan_dis_loss', loss)
            return loss
        # Train JPEG GAN generator
        elif optimizer_idx == 1:
            fake_pair = torch.cat([container, g_jpeg], dim=1)
            fake_pred = self.jpeg_gan.discriminator(fake_pair)
            gan_loss = -torch.mean(fake_pred)
            content_loss = F.l1_loss(g_jpeg, t_jpeg)
            adv_loss = -F.binary_cross_entropy_with_logits(self.decoder(g_jpeg), msg)

            loss_cfg = self.hparams.loss_cfg.jpeg_gan.generator
            loss = loss_cfg.gan_loss_scale * gan_loss
            loss += loss_cfg.content_loss_scale * content_loss
            loss += loss_cfg.adv_loss_scale * adv_loss
            self.log('train/gen_gan_loss', gan_loss)
            self.log('train/gen_content_loss', content_loss)
            self.log('train/gen_adv_loss', adv_loss)
            self.log('train/gen_loss', loss)
            return loss
        # Train Discriminator
        elif optimizer_idx == 2:
            return self.cal_discriminator_loss(img, container)
        # Train En/Decoder
        elif optimizer_idx == 3:
            if self.hparams.forward_asl:
                g_jpeg = container + (g_jpeg - container).detach()
                t_jpeg = container + (t_jpeg - container).detach()
            return self.cal_encoder_decoder_loss(img, container, g_jpeg, msg)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.hparams.simulate_type == 'advgan':
            return self.adversarial_simulate_training_step(batch, batch_idx, optimizer_idx)
        else:
            return self.simulate_training_step(batch, batch_idx, optimizer_idx)

    def training_epoch_end(self, outputs: Any) -> None:
        self.train_img_metric.reset()
        self.train_msg_metric.reset()
        self.train_tmsg_metric.reset()

    def validation_step(self, batch, batch_idx):
        img, msg = batch
        container = self.encoder(img, msg)
        quality = random.randint(10, 95)
        # quality = 50
        jpeg_container = self.image_norm(image_compression_by_cv2(container, quality))
        hat_msg_logits = self.decoder(container)
        hat_tmsg_logits = self.decoder(jpeg_container)

        if batch_idx == 0:
            self.logger.experiment.add_image('valid/image', self.image_denorm(img[0]), self.current_epoch)
            self.logger.experiment.add_image('valid/container', self.image_denorm(container[0]), self.current_epoch)
            self.logger.experiment.add_image('valid/jpeg_container', self.image_denorm(jpeg_container[0]),
                                             self.current_epoch)

        self.valid_img_metric(container, img)
        hat_msg = torch.round(torch.sigmoid(hat_msg_logits))
        hat_tmsg = torch.round(torch.sigmoid(hat_tmsg_logits))
        self.valid_msg_metric(hat_msg, msg)
        self.valid_tmsg_metric(hat_tmsg, msg)

    def validation_epoch_end(self, outputs: Any) -> None:
        valid_img_metric = self.valid_img_metric.compute()
        valid_msg_metric = self.valid_msg_metric.compute()
        valid_tmsg_metric = self.valid_tmsg_metric.compute()
        self.valid_img_metric.reset()
        self.valid_msg_metric.reset()
        self.valid_tmsg_metric.reset()

        self.logger.experiment.add_scalar('valid/psnr', valid_img_metric['PeakSignalNoiseRatio'], self.current_epoch)
        self.logger.experiment.add_scalar('valid/ssim',
                                          valid_img_metric['StructuralSimilarityIndexMeasure'], self.current_epoch)
        self.logger.experiment.add_scalar('valid/bit_acc', valid_msg_metric, self.current_epoch)
        self.logger.experiment.add_scalar('valid/tbit_acc', valid_tmsg_metric, self.current_epoch)
        self.log('valid/tbit_acc', valid_tmsg_metric, prog_bar=True, logger=False)
        # self.log('valid/bit_acc', valid_msg_metric)

    def configure_optimizers(self):
        enc_dec_optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], lr=self.hparams.lr)

        discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=1e-3,
        )
        discriminator_freq = 1 if self.hparams.enable_gan else 0

        if self.hparams.simulate_type == 'advgan':
            jpeg_dis_optim = torch.optim.Adam(self.jpeg_gan.discriminator.parameters(), lr=self.hparams.lr)
            jpeg_gen_optim = torch.optim.Adam(self.jpeg_gan.generator.parameters(), lr=self.hparams.lr)

            return ({'optimizer': jpeg_dis_optim, 'frequency': 2},
                    {'optimizer': jpeg_gen_optim, 'frequency': 1},
                    {'optimizer': discriminator_optim, 'frequency': discriminator_freq},
                    {'optimizer': enc_dec_optim, 'frequency': 1})
        else:
            return ({'optimizer': discriminator_optim, 'frequency': discriminator_freq},
                    {'optimizer': enc_dec_optim, 'frequency': 1})


def run():
    pass


if __name__ == '__main__':
    run()
