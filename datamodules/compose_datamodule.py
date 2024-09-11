#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : compose_datamodule.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/12 23:00
#
# Import lib here
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from .mirflickr import Mirflickr
from .image_distortion_pair import ImageDistortionPair

from omegaconf import DictConfig


class ComposeDataModule(LightningDataModule):
    def __init__(self, wm_dataset_cfg: DictConfig,
                 dt_dataset_cfg: DictConfig,
                 dataloader_cfg: DictConfig):
        super().__init__()
        self.wm_dataset_cfg = wm_dataset_cfg
        self.dt_dataset_cfg = dt_dataset_cfg
        self.dataloader_cfg = dataloader_cfg

        self.wm_train_dataset = None
        self.wm_val_dataset = None
        self.wm_test_dataset = None

        self.dt_train_dataset = None
        self.dt_val_dataset = None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: str = None):
        if not self.wm_train_dataset or not self.wm_val_dataset or not self.wm_test_dataset:
            self.wm_train_dataset = Mirflickr(**self.wm_dataset_cfg, stage='train')
            self.wm_val_dataset = Mirflickr(**self.wm_dataset_cfg, stage='val')
            self.wm_test_dataset = Mirflickr(**self.wm_dataset_cfg, stage='test')
        if not self.dt_train_dataset or not self.dt_val_dataset:
            self.dt_train_dataset = ImageDistortionPair(**self.dt_dataset_cfg, stage='train')
            self.wm_val_dataset = ImageDistortionPair(**self.dt_dataset_cfg, stage='val')

    def train_dataloader(self):
        return {'wm_dataset': DataLoader(self.wm_train_dataset, **self.dataloader_cfg, shuffle=True),
                'dt_dataset': DataLoader(self.dt_train_dataset, **self.dataloader_cfg, shuffle=True)}

    def val_dataloader(self):
        return {'wm_dataset': DataLoader(self.wm_val_dataset, **self.dataloader_cfg),
                'wm_dataset2': DataLoader(self.wm_val_dataset, **self.dataloader_cfg)}
        # return {'wm_dataset': DataLoader(self.wm_val_dataset, **self.dataloader_cfg),
        #         'dt_dataset': DataLoader(self.dt_val_dataset, **self.dataloader_cfg)}

    def test_dataloader(self):
        return {'wm_dataset': DataLoader(self.wm_test_dataset, **self.dataloader_cfg)}


def run():
    pass


if __name__ == '__main__':
    run()
