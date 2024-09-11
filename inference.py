#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : inference2.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/11/15 11:42
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import hydra
import torch
import torchvision.transforms.functional as tvf
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from pytorch_lightning import (
    LightningModule,
    seed_everything,
)
from models.adversarial_distortion_gans import AdversarialDistortionGANs
from models.noiser.lossy_compression import tensor_lossy_compression
from torchmetrics.functional import peak_signal_noise_ratio
from omegaconf import DictConfig

from utils import utils
log = utils.get_logger(__name__)


def encode(model: AdversarialDistortionGANs):
    device = model.device
    host_dir = '/sda1/Datasets/ImageNet_CoCo/128'
    log.info(host_dir)
    host_paths = sorted(path for path in Path(host_dir).glob('*'))
    container_dir = '/sda1/Datasets/AdversarialDistortionTranslation/Wechat/adv_0.01/container'
    log.info(container_dir)
    avg_psnr = 0.
    for path in host_paths:
        host_img = Image.open(path).convert('RGB')
        host = tvf.normalize(tvf.to_tensor(host_img), [0.5], [0.5])
        msg = torch.tensor([0., 1.] * 15)
        with torch.no_grad():
            container = model.encoder(host[None].to(device), msg[None].to(device))[0].cpu()
        psnr = peak_signal_noise_ratio(container, host)
        avg_psnr += psnr
        container_img = tvf.to_pil_image((container + 1.) / 2.)
        container_img.save(f'{container_dir}/{path.stem}.png')
        log.info(f'saving {path.stem}.png: {psnr:.2f}dB')
    avg_psnr = avg_psnr / len(host_paths)
    log.info(f'average psnr: {avg_psnr:.2f}dB')
    # import pdb; pdb.set_trace()
    # print('hello')


def decode(model: AdversarialDistortionGANs):
    device = model.device
    distortion_dir = '/sda1/Datasets/AdversarialDistortionTranslation/Wechat/adv_0.01/distortion_random'
    log.info(distortion_dir)
    distortion_paths = sorted(path for path in Path(distortion_dir).glob('*'))
    avg_tbit_acc = 0.
    for path in distortion_paths:
        distortion_img = Image.open(path).convert('RGB')
        distortion = tvf.normalize(tvf.to_tensor(distortion_img), [0.5], [0.5])
        # distortion = tensor_lossy_compression(distortion, quality_factor=50, ext='.jp2')
        with torch.no_grad():
            msg = model.decoder(distortion[None].to(device))[0].sigmoid().cpu()
        msg = (msg > 0.5).float()
        tbit_acc = (msg == torch.tensor([0., 1.] * 15)).float().mean()
        avg_tbit_acc += tbit_acc
        log.info(f'{path.name}: {tbit_acc:.5f}')
    avg_tbit_acc = avg_tbit_acc / len(distortion_paths)
    log.info(f'average tbit acc: {avg_tbit_acc:.5f}')


@hydra.main(config_path="configs/", config_name="inference.yaml", version_base="1.3")
def main(config: DictConfig) -> None:

    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    device = torch.device('cuda:0')
    model: AdversarialDistortionGANs = hydra.utils.instantiate(config.model)
    state_dict = torch.load(config.ckpt_path)['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    encode(model)
    # decode(model)


if __name__ == '__main__':
    main()
