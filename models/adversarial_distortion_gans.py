#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : adversarial_distortion_gans2.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/20 16:07
#
# Import lib here
import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, seed_everything
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, Accuracy
from torchmetrics.functional import peak_signal_noise_ratio, accuracy
from lpips.lpips import LPIPS
from kornia.color import rgb_to_yuv
from models.noiser.lossy_compression import tensor_lossy_compression
from models.noiser.ps_effects import tensor_ps_effects
from models.network.gan import cal_gradient_penalty

from omegaconf import DictConfig
from typing import Tuple, List, Any, Callable, Dict
from utils import utils

log = utils.get_logger(__name__)
LPIPS_LOSS = LPIPS(pretrained=True, net='vgg')


class AdversarialDistortionGANs(LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, discriminator: nn.Module,
                 distortion_gans: nn.ModuleDict,
                 train_distortions: Dict[str, Callable],
                 test_distortions: Dict[str, Callable],
                 train_cfg: DictConfig, loss_cfg: DictConfig):
        super().__init__()
        self.automatic_optimization = False

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.distortion_gans = distortion_gans
        self.train_distortions = train_distortions
        self.test_distortions = test_distortions
        self.train_cfg = train_cfg
        self.loss_cfg = loss_cfg

        num_distortion = len(self.test_distortions)

        # metric
        self.valid_psnr = PeakSignalNoiseRatio()
        self.valid_tbit_acc = Accuracy(task='binary')

        self.test_psnr = PeakSignalNoiseRatio()
        self.test_ssim = StructuralSimilarityIndexMeasure()
        self.test_tbit_acc = Accuracy(task='binary')
        self.test_tbit_acc_dict = nn.ModuleDict()
        for idx, d_type in enumerate(self.test_distortions.keys()):
            self.test_tbit_acc_dict[d_type] = Accuracy(task='binary')

    def forward(self):
        pass

    def get_true_distortion(self, image, container, distortion_type: str):
        container = self.image_denorm(container)

        if distortion_type == 'jpeg':
            t_distortion = tensor_lossy_compression(container, 50, ext='.jpg')
        elif distortion_type == 'jpeg2000':
            t_distortion = tensor_lossy_compression(container, 50, ext='.jp2')
        elif distortion_type == 'webp':
            t_distortion = tensor_lossy_compression(container, 50, ext='.webp')
        elif distortion_type == 'rainy':
            # Enlarge container for better visual
            original_size = container.shape[-2:]
            container = F.interpolate(container, (256, 256))
            t_distortion = tensor_ps_effects(container, 'rainy')
            t_distortion = F.interpolate(t_distortion, original_size)
        elif distortion_type == 'emboss':
            t_distortion = tensor_ps_effects(container, 'emboss')
        elif distortion_type == 'fisheye':
            t_distortion = tensor_ps_effects(container, 'fisheye')
        elif distortion_type == 'solarize':
            t_distortion = tensor_ps_effects(container, 'solarize')
        else:
            raise ValueError

        return self.image_norm(t_distortion)

    def on_train_start(self) -> None:
        # clear the cache of validation metrics due to the sanity checks
        self.valid_psnr.reset()
        self.valid_tbit_acc.reset()

        # Using global variable to avoid saving model in pickle
        global LPIPS_LOSS
        LPIPS_LOSS = LPIPS_LOSS.to(self.device)

    def _train_distortion_gan(self, source, msg, distortion_types: List[str], adversarial_loss: bool = False):
        optimizers = self.optimizers()
        dt_gen_optims = optimizers[2:2+len(self.distortion_gans)]
        dt_dis_optims = optimizers[2+len(self.distortion_gans):]

        for idx, d_type in enumerate(distortion_types):
            self.set_requires_grad(self.distortion_gans[d_type].generator, True)
            fake = self.distortion_gans[d_type].generator(source)
            fake_data = torch.cat([source, fake], dim=1)
            real = self.get_true_distortion(None, source, d_type.lower()[:-3])
            real_data = torch.cat([source, real], dim=1)

            # Update distortion_gan discriminator
            if self.distortion_gans[d_type].loss_cfg.gan_weight:
                self.set_requires_grad(self.distortion_gans[d_type].discriminator, True)
                fake_logit = self.distortion_gans[d_type].discriminator(fake_data.detach())
                real_logit = self.distortion_gans[d_type].discriminator(real_data)
                # dis_loss = F.mse_loss(fake_logit, torch.zeros_like(fake_logit)) + \
                #            F.mse_loss(real_logit, torch.ones_like(real_logit))
                gp = cal_gradient_penalty(self.distortion_gans[d_type].discriminator, real_data, fake_data.detach())[0]
                dis_loss = -torch.mean(real_logit) + torch.mean(fake_logit) + gp
                dt_dis_optims[idx].zero_grad()
                self.manual_backward(dis_loss)
                dt_dis_optims[idx].step()
                self.log(f'train/{d_type}_dis_loss', dis_loss.item())
            else:
                self.log(f'train/{d_type}_dis_loss', 0)

            # Update generator
            self.set_requires_grad(self.distortion_gans[d_type].discriminator, False)
            fake_logit = self.distortion_gans[d_type].discriminator(fake_data)
            # gen_loss = F.mse_loss(fake_logit, torch.ones_like(fake_logit)) * self.loss_cfg.adgan.gan_weight
            gen_loss = -torch.mean(fake_logit) * self.distortion_gans[d_type].loss_cfg.gan_weight
            gen_loss = gen_loss + F.l1_loss(fake, real) * self.distortion_gans[d_type].loss_cfg.l1_weight
            if self.distortion_gans[d_type].loss_cfg.adv_weight and adversarial_loss:
                self.set_requires_grad(self.decoder, False)
                tmsg_logit = self.decoder(fake)
                adv_loss = -F.binary_cross_entropy_with_logits(tmsg_logit, msg) * self.distortion_gans[d_type].loss_cfg.adv_weight
                gen_loss = gen_loss + adv_loss
                self.log(f'train/{d_type}_adv_loss', adv_loss.item())
            dt_gen_optims[idx].zero_grad()
            self.manual_backward(gen_loss)
            dt_gen_optims[idx].step()
            self.log(f'train/{d_type}_gen_loss', gen_loss.item())

            if self.total_steps % 300 == 0:
                show_imgs = [self.image_denorm(source[0]),
                             self.image_denorm(fake[0]),
                             self.image_denorm(real[0])]
                diff = torch.abs(real - fake)
                show_imgs.append(self.image_denorm(diff[0], mode='min_max'))
                self.logger_instance.add_image(f'train/{d_type}_example',
                                               torch.cat(show_imgs, dim=-1), self.total_steps)
                self.log(f'train/{d_type}_simulate_diff', diff.mean().item())

                # Calculate the FID score
                

    def training_step(self, batch, batch_idx):
        # prepare data
        host, msg = batch[:2]

        # prepare optimizers
        optimizers = self.optimizers()
        wt_gen_optim, wt_dis_optim = optimizers[:2]

        # Training process begin
        # Encoding process
        container = self.encoder(host, msg)

        # Addition training for distortion gans
        # if batch_idx % 3 == 0:
        self._train_distortion_gan(container.detach(), None, list(self.distortion_gans.keys()), False)
        # End

        if self.loss_cfg.wgan.gan_weight:
            # Update discriminator
            self.set_requires_grad(self.discriminator, True)
            fake_logit = self.discriminator(container.detach())
            real_logit = self.discriminator(host)
            # dis_loss = F.mse_loss(fake_logit, torch.zeros_like(fake_logit)) + \
            #            F.mse_loss(real_logit, torch.ones_like(real_logit))
            gp = cal_gradient_penalty(self.discriminator, host, container.detach())[0]
            dis_loss = -torch.mean(real_logit) + torch.mean(fake_logit) + gp
            wt_dis_optim.zero_grad()
            self.manual_backward(dis_loss)
            wt_dis_optim.step()
            self.log('train/dis_loss', dis_loss.item())

            # Update generator
            self.set_requires_grad(self.discriminator, False)
            fake_logit = self.discriminator(container)
            gen_loss = -torch.mean(fake_logit) * self.loss_cfg.wgan.gan_weight
            # gen_loss = F.mse_loss(fake_logit, torch.ones_like(fake_logit)) * self.loss_cfg.wgan.gan_weight
            self.log('train/gen_loss', gen_loss.item())
        else:
            gen_loss = torch.tensor(0., device=self.device)
        if self.loss_cfg.wgan.lpips_weight:
            lpips_dist = LPIPS_LOSS(container, host)
            lpips_loss = lpips_dist.mean() * self.loss_cfg.wgan.lpips_weight
            self.log('train/lpips_loss', lpips_loss.item())
        else:
            lpips_loss = torch.tensor(0., device=self.device)
        # yuv_container, yuv_host = rgb_to_yuv((container+1)/2.), rgb_to_yuv((host+1)/2.)
        # channel_weight = torch.tensor([1, 100, 100], device=self.device).view(1, -1, 1, 1)
        # pix_loss = F.mse_loss(yuv_container, yuv_host, reduction='none') * channel_weight
        # pix_loss = pix_loss.mean() * self.loss_cfg.wgan.pix_weight
        pix_loss = F.mse_loss(container, host) * self.loss_cfg.wgan.pix_weight
        self.log('train/pix_loss', pix_loss.item())

        # Distortion process
        rnd_idx = random.choice(range(len(self.distortion_gans) + len(self.train_distortions)))
        if rnd_idx < len(self.distortion_gans):
            key = list(self.distortion_gans.keys())[rnd_idx]
            for _ in range(self.distortion_gans[key].loss_cfg.adv_steps):
                self._train_distortion_gan(container.detach(), msg, [key], True)
            # Generate distorted container
            self.set_requires_grad(self.distortion_gans[key].generator, False)
            distorted_container = self.distortion_gans[key].distort(container)
        else:
            key = list(self.train_distortions.keys())[rnd_idx - len(self.distortion_gans)]
            denorm_container = self.image_denorm(container).clamp(0, 1)
            denorm_distorted_container = self.train_distortions[key](denorm_container, self.total_steps)
            distorted_container = self.image_norm(denorm_distorted_container)

        # Decoding process
        self.set_requires_grad(self.decoder, True)
        tmsg_logit = self.decoder(distorted_container)
        tmsg_loss = F.binary_cross_entropy_with_logits(tmsg_logit, msg) * self.loss_cfg.wgan.msg_weight
        self.log('train/tmsg_loss', tmsg_loss.item())

        # Total loss
        loss = (gen_loss + lpips_loss + pix_loss) * min(1., self.total_steps / 1000) + tmsg_loss
        wt_gen_optim.zero_grad()
        self.manual_backward(loss)
        wt_gen_optim.step()

        # Log
        if self.total_steps % 300 == 0:
            show_distorted_container = self.image_denorm(distorted_container[0]).detach().cpu().permute(1, 2, 0)
            show_distorted_container = (show_distorted_container.numpy() * 255).astype('uint8').copy()
            cv2.putText(show_distorted_container, key, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            show_distorted_container = torch.from_numpy(show_distorted_container).permute(2, 0, 1) / 255.
            show_distorted_container = show_distorted_container.to(self.device)

            show_imgs = [self.image_denorm(host[0]), self.image_denorm(container[0]),
                         (torch.abs(container[0] - host[0]) * 5).clamp(0, 1),
                         show_distorted_container]
            self.logger_instance.add_image(f'train/example', torch.cat(show_imgs, dim=-1), self.total_steps)

        # Calculate metrics
        with torch.no_grad():
            self.log('train/psnr', peak_signal_noise_ratio(container, host).item(), prog_bar=True)
            self.log('train/tbit_acc', accuracy(tmsg_logit.sigmoid(), msg, task='binary').item(), prog_bar=True)

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        host, msg = batch
        with torch.no_grad():
            container = self.encoder(host, msg)
            self.valid_psnr.update(container, host)
            if self.train_cfg.simulate_type == 'combine':
                for idx, d_type in enumerate(self.test_distortions.keys()):
                    denorm_container = self.image_denorm(container).clamp(0, 1)
                    distorted_container = self.image_norm(self.test_distortions[d_type](denorm_container))
                    tmsg_logit = self.decoder(distorted_container)
                    self.valid_tbit_acc.update(tmsg_logit.sigmoid(), msg)
            else:
                raise ValueError

            # for idx, d_type in enumerate(self.test_distortions.keys()):
            #     distorted_container = self.test_distortions[d_type](distorted_container)
            #     tmsg_logit = self.decoder(distorted_container)
            #     self.valid_tbit_acc.update(tmsg_logit.sigmoid(), msg)

    def on_validation_epoch_end(self) -> None:
        psnr = self.valid_psnr.compute().item()
        self.valid_psnr.reset()
        self.logger_instance.add_scalar(f'valid/psnr', psnr, self.current_epoch)

        tbit_acc = self.valid_tbit_acc.compute().item()
        self.valid_tbit_acc.reset()
        self.logger_instance.add_scalar(f'valid/tbit_acc', tbit_acc, self.current_epoch)

        # Used to select the best checkpoint in ModelCheckpoint callback
        if psnr < self.train_cfg.target_psnr:
            # Skip the ckpt is the psnr is lower than the target
            tbit_acc = 0.5
        self.log('psnr', psnr, logger=False)
        self.log('tbit_acc', tbit_acc, logger=False, prog_bar=True)

    def configure_optimizers(self):
        enc_dec_optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], lr=self.train_cfg.lr, weight_decay=1e-5)

        discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.train_cfg.lr, weight_decay=1e-5)

        dt_gen_optims = []
        dt_dis_optims = []
        for key in list(self.distortion_gans.keys()):
            dt_gen_optims.append(torch.optim.Adam(
                self.distortion_gans[key].generator.parameters(),
                lr=self.train_cfg.lr, weight_decay=1e-5))
            dt_dis_optims.append(torch.optim.Adam(
                self.distortion_gans[key].discriminator.parameters(),
                lr=self.train_cfg.lr, weight_decay=1e-5))

        return enc_dec_optim, discriminator_optim, *dt_gen_optims, *dt_dis_optims

    def on_test_start(self) -> None:
        seed_everything(42, workers=True)

    def test_step(self, batch: Any, batch_idx: int):
        host, msg = batch
        with torch.no_grad():
            container = self.encoder(host, msg)
            # container = (container - host) * 1.1 + host
            self.test_psnr.update(container, host)
            self.test_ssim.update(container, host)

            if self.train_cfg.simulate_type == 'combine':
                for idx, d_type in enumerate(self.test_distortions.keys()):
                    denorm_container = self.image_denorm(container).clamp(0, 1)
                    distorted_container = (self.test_distortions[d_type](denorm_container) - 0.5) * 2.
                    tmsg_logit = self.decoder(distorted_container)
                    self.test_tbit_acc.update(tmsg_logit.sigmoid(), msg)
                    self.test_tbit_acc_dict[d_type].update(tmsg_logit.sigmoid(), msg)
            else:
                raise ValueError

        # Log example images
        show_imgs = [self.image_denorm(host[0]), self.image_denorm(container[0]),
                     self.image_denorm(container[0] - host[0], mode='min_max')]
        self.logger_instance.add_image(f'test/example', torch.cat(show_imgs, dim=-1), batch_idx)

    def on_test_epoch_end(self) -> None:
        psnr = self.test_psnr.compute().item()
        self.test_psnr.reset()
        self.logger_instance.add_scalar(f'test/psnr', psnr, self.current_epoch)

        ssim = self.test_ssim.compute().item()
        self.test_ssim.reset()
        self.logger_instance.add_scalar(f'test/ssim', ssim, self.current_epoch)

        tbit_acc = self.test_tbit_acc.compute().item()
        self.test_tbit_acc.reset()
        self.logger_instance.add_scalar(f'test/tbit_acc', tbit_acc, self.current_epoch)

        log.info('testing results:')
        log.info(f'psnr: {psnr:.5f}, ssim: {ssim:.5f}, tbit_acc: {tbit_acc:.5f}')

        for idx, d_type in enumerate(self.test_distortions.keys()):
            tbit_acc = self.test_tbit_acc_dict[d_type].compute().item()
            self.test_tbit_acc_dict[d_type].reset()
            self.logger_instance.add_scalar(f'test/{d_type}_tbit_acc', tbit_acc, self.current_epoch)
            log.info(f'{d_type}_tbit_acc: {tbit_acc:.5f}')

    @staticmethod
    def image_denorm(image: torch.Tensor, mode: str = 'default'):
        if mode == 'default':
            return (image + 1.) / 2.
        elif mode == 'min_max':
            return (image - image.min()) / (image.max() - image.min())
        else:
            raise ValueError

    @staticmethod
    def image_norm(image: torch.Tensor):
        return image * 2. - 1.

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
    def clip_container_psnr(container: torch.Tensor, host: torch.Tensor, psnr: float) -> torch.Tensor:
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
        return (host + residual * factor).clamp(-1., 1.)

    @property
    def total_steps(self):
        return self.trainer.fit_loop.total_batch_idx + 1

    @property
    def logger_instance(self):
        return self.logger.experiment


def run():
    pass


if __name__ == '__main__':
    run()
