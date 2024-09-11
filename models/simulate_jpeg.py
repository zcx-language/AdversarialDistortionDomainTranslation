#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : simulate_jpeg.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/4/28 上午10:39

# Import lib here
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.transforms import Normalize
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import NoTrainLpips
from metrics import BitAccuracy, SSIM
from models.encoder import StegaStampEncoder
from models.decoder import StegaStampDecoder
from models.discriminator import StegaStampDiscriminator
from models.network.gan import ImagePool, cal_gradient_penalty
from models.noiser.lossy_compression import tensor_lossy_compression
from models.noiser.jpeg_simulator import JpegMask, JpegSS, Jpeg
from omegaconf import DictConfig
from typing import Any, Tuple

from utils import utils
log = utils.get_logger(__name__)


# Attention: do not support multiple gpus
class SimulateJPEG(pl.LightningModule):
    def __init__(self, img_shape: Tuple[int, int, int],
                 msg_len: int,
                 simulate_type: str,
                 loss_cfg: DictConfig,
                 lr: float = 1e-3,
                 min_psnr: int = 33,
                 enable_gan: bool = True):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.encoder = StegaStampEncoder(img_shape, msg_len)
        self.decoder = StegaStampDecoder(img_shape, msg_len)
        self.discriminator = StegaStampDiscriminator(in_channels=3)
        self.image_pool = ImagePool()

        self.JPEG = Jpeg(50)
        self.JPEG_Mask = JpegMask()
        self.JPEG_SSes = [JpegSS(Q=10*i) for i in range(1, 10)]

        # metric
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

        self.test_img_metric = MetricCollection([
            PeakSignalNoiseRatio(data_range=2.),
            StructuralSimilarityIndexMeasure(data_range=2.),
        ])
        self.test_msg_metric = BitAccuracy()
        self.test_tmsg_metric = BitAccuracy()
        self.test_tmsg_metrics = nn.ModuleList([BitAccuracy() for _ in range(10)])

        self.image_norm = Normalize(0.5, 0.5)
        self.image_denorm = Normalize(-1., 2.)
        self.lpips_loss = NoTrainLpips()

    def encode(self, img, msg):
        """Only for prediction, do not use this in train phase."""
        pass

    def decode(self, img):
        """Only for prediction, do not use this in train phase."""
        pass

    def on_train_start(self) -> None:
        # Show metrics in tensorboard
        self.logger.log_metrics({'hp/tbit_acc': 0, 'hp/psnr': 0, 'hp/ssim': 0})

    def _discriminator_step(self, image, container, optimizer):
        real_pred = self.discriminator(image)
        container = self.image_pool.push_and_pop(container.detach())
        fake_pred = self.discriminator(container)
        gradient_penalty = cal_gradient_penalty(self.discriminator, image, container)[0]
        loss = -torch.mean(real_pred) + torch.mean(fake_pred) + gradient_penalty
        self.log('train/dis_loss', loss)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

    def _encoder_decoder_step(self, image, container, distortion, msg, optimizer):
        # Generator loss
        gan_loss = -torch.mean(self.discriminator(container)) if self.hparams.enable_gan else 0.

        # vis loss
        l1_loss = F.l1_loss(image, container)
        lpips_loss = torch.mean(self.lpips_loss(image, container))
        vis_loss = l1_loss + lpips_loss

        # Msg loss
        hat_msg_logits = self.decoder(container)
        msg_loss = F.binary_cross_entropy_with_logits(hat_msg_logits, msg)

        with torch.no_grad():
            train_img_metric = self.train_img_metric(container, image)
            hat_msg = torch.round(torch.sigmoid(hat_msg_logits))
            train_msg_metric = self.train_msg_metric(hat_msg, msg)
        self.log('train/psnr', train_img_metric['PeakSignalNoiseRatio'])
        self.log('train/ssim', train_img_metric['StructuralSimilarityIndexMeasure'])
        self.log('train/bit_acc', train_msg_metric)

        hat_tmsg_logits = self.decoder(distortion)
        tmsg_loss = F.binary_cross_entropy_with_logits(hat_tmsg_logits, msg)
        self.log('train/tmsg_loss', tmsg_loss)
        msg_loss = msg_loss + tmsg_loss

        with torch.no_grad():
            hat_tmsg = torch.round(torch.sigmoid(hat_tmsg_logits))
            train_tmsg_metric = self.train_tmsg_metric(hat_tmsg, msg)
        self.log('train/tbit_acc', train_tmsg_metric, prog_bar=True)

        loss_cfg = self.hparams.loss_cfg
        loss = (loss_cfg.vis_loss_scale * vis_loss +
                loss_cfg.msg_loss_scale * msg_loss +
                loss_cfg.gan_loss_scale * gan_loss)

        self.log('train/gen_loss', gan_loss)
        self.log('train/lpips_loss', lpips_loss)
        self.log('train/l1_loss', l1_loss)
        self.log('train/msg_loss', msg_loss)
        self.log('train/loss', loss)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

    def training_step(self, batch, batch_idx):
        img, msg = batch
        container = self.encoder(img, msg)

        if self.hparams.simulate_type == 'JPEG_Mask':
            s_jpeg_container = self.JPEG_Mask(container)
        elif self.hparams.simulate_type == 'JPEG_SS':
            s_jpeg_container = self.JPEG_SSes[4](container)
        elif self.hparams.simulate_type == 'JPEG_SS_Adv':
            max_loss, max_idx = 0., 0
            with torch.no_grad():
                for idx, JPEG_SS in enumerate(self.JPEG_SSes):
                    jpeg_container = JPEG_SS(container)
                    msg_logits = self.decoder(jpeg_container)
                    loss = F.binary_cross_entropy_with_logits(msg_logits, msg)
                    if loss > max_loss:
                        max_loss = loss
                        max_idx = idx
            s_jpeg_container = self.JPEG_SSes[max_idx](container)
        elif self.hparams.simulate_type == 'JPEG_Mix':
            rnd = random.randint(1, 3)
            if rnd == 1:
                s_jpeg_container = container
            elif rnd == 2:
                s_jpeg_container = self.JPEG_Mask(container)
            elif rnd == 3:
                s_jpeg_container = self.JPEG(container)
            else:
                raise ValueError
        else:
            raise NotImplementedError

        if self.global_step % 500 == 0:
            self.logger.experiment.add_image('train/image', self.image_denorm(img[0]), self.global_step)
            self.logger.experiment.add_image('train/container', self.image_denorm(container[0]), self.global_step)
            self.logger.experiment.add_image('train/s_jpeg_container',
                                             self.image_denorm(s_jpeg_container[0]), self.global_step)

        enc_dec_opt, dis_opt = self.optimizers()
        if self.hparams.enable_gan:
            self._discriminator_step(img, container, dis_opt)
        self._encoder_decoder_step(img, container, s_jpeg_container, msg, enc_dec_opt)

    def training_epoch_end(self, outputs: Any) -> None:
        self.train_img_metric.reset()
        self.train_msg_metric.reset()
        self.train_tmsg_metric.reset()

    def _get_true_jpeg(self, container, quality_factor: int = 50):
        container = self.image_denorm(container)
        t_jpeg_container = tensor_lossy_compression(container, quality_factor, ext='.jpg')
        return self.image_norm(t_jpeg_container)

    def validation_step(self, batch, batch_idx):
        img, msg = batch
        container = self.encoder(img, msg)
        quality = random.randint(10, 95)
        jpeg_container = self._get_true_jpeg(container, quality)
        hat_msg_logits = self.decoder(container)
        hat_tmsg_logits = self.decoder(jpeg_container)

        if self.trainer.is_last_batch and self.global_rank == 0:
            self.logger.experiment.add_image('valid/image', self.image_denorm(img[0]), self.current_epoch)
            self.logger.experiment.add_image('valid/container', self.image_denorm(container[0]), self.current_epoch)
            self.logger.experiment.add_image('valid/jpeg_container',
                                             self.image_denorm(jpeg_container[0]), self.current_epoch)

        self.valid_img_metric.update(container, img)
        hat_msg = torch.round(torch.sigmoid(hat_msg_logits))
        hat_tmsg = torch.round(torch.sigmoid(hat_tmsg_logits))
        self.valid_msg_metric.update(hat_msg, msg)
        self.valid_tmsg_metric.update(hat_tmsg, msg)

    def validation_epoch_end(self, outputs: Any) -> None:
        valid_img_metric = self.valid_img_metric.compute()
        valid_msg_metric = self.valid_msg_metric.compute()
        valid_tmsg_metric = self.valid_tmsg_metric.compute()
        self.valid_img_metric.reset()
        self.valid_msg_metric.reset()
        self.valid_tmsg_metric.reset()

        if self.global_rank == 0:
            log.info(f'Epoch: {self.current_epoch:}, '
                     f'psnr: {valid_img_metric["PeakSignalNoiseRatio"]}, '
                     f'ssim: {valid_img_metric["StructuralSimilarityIndexMeasure"]}, '
                     f'tbit_acc:{valid_tmsg_metric}')

            self.logger.experiment.add_scalar('valid/psnr',
                                              valid_img_metric['PeakSignalNoiseRatio'], self.current_epoch)
            self.logger.experiment.add_scalar('valid/ssim',
                                              valid_img_metric['StructuralSimilarityIndexMeasure'], self.current_epoch)
            self.logger.experiment.add_scalar('valid/bit_acc', valid_msg_metric, self.current_epoch)
            self.logger.experiment.add_scalar('valid/tbit_acc', valid_tmsg_metric, self.current_epoch)

            # Used for model checkpoint callback, only save ckpts that PSNR > min_psnr.
            if valid_img_metric['PeakSignalNoiseRatio'] > self.hparams.min_psnr:
                tbit_acc = valid_tmsg_metric
            else:
                tbit_acc = 0.0
            self.log('tbit_acc', tbit_acc, prog_bar=True, logger=False)
            self.log('psnr', valid_img_metric['PeakSignalNoiseRatio'], logger=False)

            # Show metrics in tensorboard
            if self.current_epoch + 1 == self.trainer.max_epochs:
                self.logger.experiment.add_scalar('hp/bit_acc', valid_msg_metric, self.current_epoch)
                self.logger.experiment.add_scalar('hp/tbit_acc', valid_tmsg_metric, self.current_epoch)
                self.logger.experiment.add_scalar('hp/psnr',
                                                  valid_img_metric['PeakSignalNoiseRatio'], self.current_epoch)
                self.logger.experiment.add_scalar('hp/ssim',
                                                  valid_img_metric['StructuralSimilarityIndexMeasure'], self.current_epoch)

    def configure_optimizers(self):
        enc_dec_optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], lr=self.hparams.lr)

        discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=1e-3,
        )
        return [enc_dec_optim, discriminator_optim]

    def test_step(self, batch: Any, batch_idx: int):
        img, msg = batch
        container = self.encoder(img, msg)
        quality_factor = random.randint(10, 95)
        jpeg_container = self._get_true_jpeg(container, quality_factor)
        hat_msg_logits = self.decoder(container)
        hat_tmsg_logits = self.decoder(jpeg_container)

        self.logger.experiment.add_image('test/image', self.image_denorm(img[0]), batch_idx)
        self.logger.experiment.add_image('test/container', self.image_denorm(container[0]), batch_idx)
        self.logger.experiment.add_image('test/distortion', self.image_denorm(jpeg_container[0]), batch_idx)

        self.test_img_metric.update(container, img)
        hat_msg = torch.round(torch.sigmoid(hat_msg_logits))
        hat_tmsg = torch.round(torch.sigmoid(hat_tmsg_logits))
        self.test_msg_metric.update(hat_msg, msg)
        self.test_tmsg_metric.update(hat_tmsg, msg)

        for i in range(10):
            quality_factor = i * 10 + 10
            distortion = self._get_true_jpeg(container, quality_factor)
            hat_tmsg_logits = self.decoder(distortion)
            hat_tmsg = torch.round(torch.sigmoid(hat_tmsg_logits))
            self.test_tmsg_metrics[i].update(hat_tmsg, msg)

    def test_epoch_end(self, outputs) -> None:
        test_img_metric = self.test_img_metric.compute()
        test_msg_metric = self.test_msg_metric.compute()
        test_tmsg_metric = self.test_tmsg_metric.compute()

        self.logger.experiment.add_scalar('test/psnr', test_img_metric['PeakSignalNoiseRatio'], self.current_epoch)
        self.logger.experiment.add_scalar('test/ssim',
                                          test_img_metric['StructuralSimilarityIndexMeasure'], self.current_epoch)
        self.logger.experiment.add_scalar('test/bit_acc', test_msg_metric, self.current_epoch)
        self.logger.experiment.add_scalar('test/tbit_acc', test_tmsg_metric, self.current_epoch)

        self.test_img_metric.reset()
        self.test_msg_metric.reset()
        self.test_tmsg_metric.reset()

        for i in range(10):
            test_tmsg_metric = self.test_tmsg_metrics[i].compute()
            self.test_tmsg_metrics[i].reset()
            self.logger.experiment.add_scalar('test/tbit_accs', test_tmsg_metric, i)


def run():
    pass


if __name__ == '__main__':
    run()
