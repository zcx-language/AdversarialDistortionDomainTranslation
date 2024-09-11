#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : host_msg_datamodule.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/2/27 14:28
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig


class HostMsgData(Dataset):
    def __init__(self, host_dir: str,
                 image_shape: Tuple = (3, 128, 128),
                 secret_len: int = 30):
        super().__init__()

        jpg_host_paths = sorted(path for path in Path(host_dir).glob('*.jpg'))
        jpeg_host_paths = sorted(path for path in Path(host_dir).glob('*.JPEG'))
        png_host_paths = sorted(path for path in Path(host_dir).glob('*.png'))
        self.host_paths = jpg_host_paths + jpeg_host_paths + png_host_paths
        self.image_shape = image_shape
        self.secret_len = secret_len

        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        self.random_crop = transforms.RandomCrop(image_shape[-2:], pad_if_needed=True)

    def __getitem__(self, item):
        host_path = self.host_paths[item]
        host = self.to_tensor_norm(Image.open(host_path).convert('RGB'))
        host = self.random_crop(host)
        secret = torch.randint(0, 2, (self.secret_len,)).float()
        return host, secret

    def __len__(self):
        return len(self.host_paths)


class HostMsgDataModule(LightningDataModule):
    def __init__(self,
                 train_data: Dataset,
                 valid_data: Dataset,
                 test_data: Dataset,
                 dataloader_cfg: DictConfig):
        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.dataloader_cfg = dataloader_cfg

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, **self.dataloader_cfg, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_data, **self.dataloader_cfg, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, **self.dataloader_cfg, shuffle=False)


def run():
    pass


if __name__ == '__main__':
    run()
