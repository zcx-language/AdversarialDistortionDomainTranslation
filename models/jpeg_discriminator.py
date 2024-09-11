#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : jpeg_discriminator.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2022/3/28 上午9:54

# Import lib here
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichModelSummary, ModelCheckpoint
from omegaconf import DictConfig
from datamodules import MirflickrPairModule

from fastai.vision.models import resnet18
from fastai.layers import ConvLayer
from typing import Optional, Callable, Type, Union, List, Any


class JPEGDiscriminator(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.net = resnet18(pretrained=False, num_classes=1)

    def forward(self, img):
        logits = self.net(img)
        return logits

    def training_step(self, batch, batch_idx):
        img, label = batch
        pred = self.net(img)
        loss = F.binary_cross_entropy_with_logits(pred, label)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred = self.net(img)
        print('label:', label.detach().cpu().numpy().squeeze())
        print('pred:', (torch.sigmoid(pred).detach().cpu().numpy().squeeze() > 0.5).astype(np.float32))
        print()
        pass

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optim


def run():
    dataset_cfg = DictConfig({
        'data_dir': '/mnt/chengxin/Datasets/Mirflickr/',
        'img_shape': [3, 400, 400],
    })
    datamodule = MirflickrPairModule(dataset_cfg, batch_size=16, num_workers=4)

    model = JPEGDiscriminator()

    callbacks = [ModelCheckpoint('//logs/jpeg_discriminator/checkpoints/',
                                 filename='{epoch:02d}',
                                 monitor='train/loss')]
    # callbacks = []

    logger = TensorBoardLogger(save_dir='//logs/jpeg_discriminator/tensorboard/')
    # logger = []

    trainer = pl.Trainer(logger, callbacks=callbacks, gpus=1, max_epochs=10, enable_model_summary=False)

    trainer.fit(model, datamodule=datamodule)


def valid():
    device = torch.device('cuda')
    dataset_cfg = DictConfig({
        'data_dir': '/mnt/chengxin/Datasets/Mirflickr/',
        'img_shape': [3, 400, 400],
    })
    datamodule = MirflickrPairModule(dataset_cfg, batch_size=16, num_workers=4)
    datamodule.setup()
    valid_dataloader = datamodule.val_dataloader()

    ckpt_path = '//logs/jpeg_discriminator/checkpoints/epoch=9-step=14069.ckpt'
    model = JPEGDiscriminator().to(device)
    ckpt_state_dict = torch.load(ckpt_path)['state_dict']
    model.load_state_dict(ckpt_state_dict)

    losses = []
    model.eval()
    for batch in valid_dataloader:
        img, label = batch
        img = img.to(device)
        label = label.to(device)
        pred = model(img)
        loss = F.binary_cross_entropy_with_logits(pred, label)
        losses.append(loss.detach().cpu().numpy())
        print('label', label.detach().cpu().numpy().squeeze())
        print('pred', (torch.sigmoid(pred).detach().cpu().numpy().squeeze() > 0.5).astype(np.float32))
        print()
    print(np.mean(losses))


if __name__ == '__main__':
    # run()
    valid()
