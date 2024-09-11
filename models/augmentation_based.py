#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : augmentation_based.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/6/18 下午12:00

# Import lib here
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torchvision.transforms import Normalize
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.classification import MultilabelAccuracy

from models.encoder import StegaStampEncoder
from models.decoder import StegaStampDecoder
from models.discriminator import StegaStampDiscriminator
from models.network.gan import ImagePool, cal_gradient_penalty
from models.noiser.lossy_compression import tensor_lossy_compression
from models.noiser.ps_effects import tensor_ps_effects
from models.noiser.da_distortion_generator import DADistortionGenerator
from models.noiser.classical_noise_layer import (tensor_gaussian_noise, tensor_gaussian_blur, tensor_resize,
                                                 tensor_dropout, tensor_cropout)
from omegaconf import DictConfig
from typing import Any, Tuple, List, Callable

from utils import utils
log = utils.get_logger(__name__)


# Attention: do not support multiple gpus
class AugmentationBased(pl.LightningModule):
    def __init__(self, img_shape: Tuple[int, int, int],
                 msg_len: int,
                 method: str,
                 train_distortions: nn.ModuleDict,
                 test_distortions: nn.ModuleDict,
                 loss_cfg: DictConfig,
                 lr: float = 1e-4,
                 enable_gan: bool = True):
        super().__init__()
        torch.set_float32_matmul_precision('high')
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.encoder = StegaStampEncoder(img_shape, msg_len)
        self.decoder = StegaStampDecoder(img_shape, msg_len)
        self.discriminator = StegaStampDiscriminator(in_channels=3)
        self.image_pool = ImagePool()
        self.distortion_generator = DADistortionGenerator(3)
        self.train_distortions = train_distortions
        self.test_distortions = test_distortions

        # metric
        self.train_img_metric = MetricCollection([
            PeakSignalNoiseRatio(data_range=2.),
            StructuralSimilarityIndexMeasure(data_range=2.),
            # LearnedPerceptualImagePatchSimilarity()
        ])
        self.train_tmsg_metric = MultilabelAccuracy(msg_len)

        self.valid_img_metric = MetricCollection([
            PeakSignalNoiseRatio(data_range=2.),
            StructuralSimilarityIndexMeasure(data_range=2.),
            # LearnedPerceptualImagePatchSimilarity(compute_on_step=False),
        ])
        self.valid_msg_metric = MultilabelAccuracy(msg_len)
        n_distortion = len(test_distortions)
        self.valid_tmsg_metrics = nn.ModuleList(MultilabelAccuracy(msg_len) for _ in range(n_distortion))

        self.test_img_metric = MetricCollection([
            PeakSignalNoiseRatio(data_range=2.),
            StructuralSimilarityIndexMeasure(data_range=2.),
        ])
        self.test_msg_metric = MultilabelAccuracy(msg_len)
        self.test_tmsg_metrics = nn.ModuleList(MultilabelAccuracy(msg_len) for _ in range(n_distortion))

        self.image_norm = Normalize(0.5, 0.5)
        self.image_denorm = Normalize(-1., 2.)

    def encode(self, img, msg):
        """Only for prediction, do not use this in train phase."""
        pass

    def decode(self, img):
        """Only for prediction, do not use this in train phase."""
        pass

    def on_train_start(self) -> None:
        # Show metrics in tensorboard
        self.logger.log_metrics({"hp/tbit_acc": 0, "hp/psnr": 0, "hp/ssim": 0})

    def get_true_distortion(self, image, container, distortion_type: str, quality_factor: int = 50):
        # Do not use for inference
        container = self.image_denorm(container)

        if distortion_type is None:
            t_distortion = container
        elif distortion_type == 'jpeg':
            t_distortion = tensor_lossy_compression(container, quality_factor, ext='.jpg')
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
            ks = random.choice([3, 5, 7])
            t_distortion = tensor_gaussian_blur(container, (ks, ks), (2., 2.))
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

    def training_step(self, batch, batch_idx):
        img, msg = batch
        container = self.encoder(img, msg)

        if self.hparams.method.lower() == 'da':
            distortion = self.distortion_generator(container, msg, self.decoder)
        elif self.hparams.method.lower() == 'forwardasl':
            distortion_type = random.choice(list(self.train_distortions.keys()))
            distortion = self.train_distortions[distortion_type](container)
            distortion = container + (distortion - container).detach()
        elif self.hparams.method.lower() == 'standardasl':
            distortion_type = random.choice(list(self.train_distortions.keys()))
            distortion = self.train_distortions[distortion_type](container)
        else:
            raise NotImplementedError

        if self.total_steps % 500 == 0:
            self.logger_instance.add_image('train/image', self.image_denorm(img[0]), self.total_steps)
            self.logger_instance.add_image('train/container', self.image_denorm(container[0]), self.total_steps)
            self.logger_instance.add_image('train/distortion', self.image_denorm(distortion[0]), self.total_steps)

        # Adopt the identity container in a probability of 0.1
        if random.random() < 0.1:
            distortion = container

        gen_optim, dis_optim = self.optimizers()

        # Train discriminator
        real_pred = self.discriminator(img)
        temp_container = self.image_pool.push_and_pop(container.detach())
        fake_pred = self.discriminator(temp_container)
        gradient_penalty = cal_gradient_penalty(self.discriminator, img, temp_container)[0]
        loss = -torch.mean(real_pred) + torch.mean(fake_pred) + gradient_penalty
        dis_optim.zero_grad()
        self.manual_backward(loss)
        dis_optim.step()
        self.log('train/dis_loss', loss)

        # Train Encoder&Decoder
        # Generator loss
        gan_loss = -torch.mean(self.discriminator(container)) * self.hparams.loss_cfg.gan_loss_scale
        # vis loss
        l1_loss = F.l1_loss(img, container) * self.hparams.loss_cfg.l1_loss_scale
        # Msg loss
        hat_tmsg_logits = self.decoder(distortion)
        msg_loss = F.binary_cross_entropy_with_logits(hat_tmsg_logits, msg) * self.hparams.loss_cfg.msg_loss_scale
        loss = l1_loss + gan_loss + msg_loss
        gen_optim.zero_grad()
        self.manual_backward(loss)
        gen_optim.step()
        self.log('train/gan_loss', gan_loss)
        self.log('train/l1_loss', l1_loss)
        self.log('train/msg_loss', msg_loss)
        self.log('train/loss', loss)

        with torch.no_grad():
            train_img_metric = self.train_img_metric(container, img)
            tbit_acc = self.train_tmsg_metric(hat_tmsg_logits.sigmoid(), msg)
        self.log('train/psnr', train_img_metric['PeakSignalNoiseRatio'].item())
        self.log('train/ssim', train_img_metric['StructuralSimilarityIndexMeasure'].item())
        self.log('train/tbit_acc', tbit_acc.item())

    def on_train_epoch_end(self) -> None:
        self.train_img_metric.reset()
        self.train_tmsg_metric.reset()

    def validation_step(self, batch, batch_idx):
        img, msg = batch

        with torch.no_grad():
            container = self.encoder(img, msg)
            hat_msg_logits = self.decoder(container)
            self.valid_img_metric.update(container, img)
            self.valid_msg_metric.update(hat_msg_logits.sigmoid(), msg)

        if self.trainer.is_last_batch:
            self.logger_instance.add_image('valid/image', self.image_denorm(img[0]), self.current_epoch)
            self.logger_instance.add_image('valid/container', self.image_denorm(container[0]), self.current_epoch)

        for idx, distortion_type in enumerate(list(self.test_distortions.keys())):
            distortion = self.test_distortions[distortion_type](container)
            with torch.no_grad():
                hat_tmsg_logits = self.decoder(distortion)
            self.valid_tmsg_metrics[idx].update(hat_tmsg_logits.sigmoid(), msg)

            if self.trainer.is_last_batch:
                self.logger_instance.add_image(f'valid/{distortion_type}',
                                               self.image_denorm(distortion[0]), self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        img_metric = self.valid_img_metric.compute()
        psnr = img_metric['PeakSignalNoiseRatio'].item()
        ssim = img_metric['StructuralSimilarityIndexMeasure'].item()
        bit_acc = self.valid_msg_metric.compute().item()
        self.valid_img_metric.reset()
        self.valid_msg_metric.reset()

        self.logger_instance.add_scalar('valid/psnr', psnr, self.current_epoch)
        self.logger_instance.add_scalar('valid/ssim', ssim, self.current_epoch)
        self.logger_instance.add_scalar('valid/bit_acc', bit_acc, self.current_epoch)

        log_str = f'Epoch:{self.current_epoch}, psnr:{psnr:.6f}'
        tbit_acc_list = []
        for idx, distortion_type in enumerate(list(self.test_distortions.keys())):
            tbit_acc = self.valid_tmsg_metrics[idx].compute().item()
            tbit_acc_list.append(tbit_acc)
            self.valid_tmsg_metrics[idx].reset()

            log_str += f', {distortion_type}_tbit_acc:{tbit_acc:.6f}'
            self.logger_instance.add_scalar(f'valid/{distortion_type}_tbit_acc',
                                            tbit_acc, self.current_epoch)

        avg_tbit_acc = sum(tbit_acc_list) / len(tbit_acc_list)
        log.info(log_str + f', avg_tbit_acc:{avg_tbit_acc:.6f}')
        self.logger_instance.add_scalar('valid/avg_tbit_acc', avg_tbit_acc, self.current_epoch)

        # Used for model checkpoint callback
        self.log('overall', avg_tbit_acc + ssim, logger=False)
        self.log('tbit_acc', avg_tbit_acc, logger=False)
        self.log('psnr', psnr, logger=False)

    def configure_optimizers(self):
        enc_dec_optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], lr=self.hparams.lr, weight_decay=1e-5)

        discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
        )
        return [enc_dec_optim, discriminator_optim]

    def on_test_start(self) -> None:
        seed_everything(42, workers=True)

    def test_step(self, batch: Any, batch_idx: int):
        img, msg = batch
        with torch.no_grad():
            container = self.encoder(img, msg)
            hat_msg_logits = self.decoder(container)
            self.test_img_metric.update(container, img)
            self.test_msg_metric.update(hat_msg_logits.sigmoid(), msg)

        for idx, distortion_type in enumerate(list(self.test_distortions.keys())):
            distortion = self.test_distortions[distortion_type](container)
            with torch.no_grad():
                hat_tmsg_logits = self.decoder(distortion)
            self.test_tmsg_metrics[idx].update(hat_tmsg_logits.sigmoid(), msg)

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
