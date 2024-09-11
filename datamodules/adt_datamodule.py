#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : adt_datamodule.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/11/14 10:34
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


class BlackBoxTrainData(Dataset):
    def __init__(self, host_dir: str,
                 source_dirs: List[str],
                 target_dirs: List[str],
                 image_shape: Tuple = (3, 128, 128),
                 secret_len: int = 30,
                 limited_num: int = 1000):
        super().__init__()

        host_paths = sorted(path for path in Path(host_dir).glob('*.jpg'))
        targets_paths = []
        for target_dir in target_dirs:
            targets_paths.append(sorted(path for path in Path(target_dir).glob('*'))[:limited_num])

        self.host_paths = host_paths
        self.source_dirs = [Path(source_dir) for source_dir in source_dirs]
        self.targets_paths = targets_paths
        self.secret_len = secret_len

        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        self.random_crop = transforms.RandomCrop(image_shape[-2:], pad_if_needed=True)

    def __getitem__(self, item):
        host_path = self.host_paths[item]
        host = self.random_crop(self.to_tensor_norm(Image.open(host_path).convert('RGB')))
        secret = torch.randint(0, 2, (self.secret_len,)).float()

        source_targets = []
        for idx, target_paths in enumerate(self.targets_paths):
            target_path: Path = target_paths[item % len(target_paths)]
            if target_path.suffix == '.npy':
                # BGR -> RGB
                target = self.to_tensor_norm(np.load(target_path)[:, :, ::-1].copy())
            else:
                target = self.to_tensor_norm(Image.open(target_path).convert('RGB'))
            source_path = self.source_dirs[idx] / f'{target_path.stem}.png'
            if not source_path.exists():
                source_path = self.source_dirs[idx] / f'{target_path.stem}.jpg'
            source = self.to_tensor_norm(Image.open(source_path).convert('RGB'))
            if source.shape[-1] != host.shape[-1]:
                source_target = self.random_crop(torch.cat([source, target], dim=0))
            else:
                source_target = torch.cat([source, target], dim=0)
            source_targets.append(source_target)
        return host, secret, *source_targets

    def __len__(self):
        return len(self.host_paths)


class BlackBoxTestData(Dataset):
    def __init__(self, source_dir: str,
                 target_dir: str,
                 image_shape: Tuple = (3, 128, 128),
                 secret_len: int = 30):
        super().__init__()
        target_paths = sorted(path for path in Path(target_dir).glob('*'))

        self.source_dir = Path(source_dir)
        self.target_paths = target_paths
        self.image_shape = image_shape
        self.secret_len = secret_len

        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        self.random_crop = transforms.RandomCrop(image_shape[-2:], pad_if_needed=True)

    def __getitem__(self, item):
        target_path = self.target_paths[item]
        if target_path.suffix == '.npy':
            target = self.to_tensor_norm(np.load(target_path)[:, :, ::-1].copy())
        else:
            target = self.to_tensor_norm(Image.open(target_path).convert('RGB'))
        source_path = self.source_dir / f'{target_path.stem}.jpg'
        if not source_path.exists():
            source_path = self.source_dir / f'{target_path.stem}.png'
        source = self.to_tensor_norm(Image.open(source_path).convert('RGB'))
        if source.shape[-1] != self.image_shape[-1]:
            target_source = self.random_crop(torch.cat([target, source], dim=0))
            target, source = target_source[:3], target_source[3:]
        secret = torch.randint(0, 2, (self.secret_len,)).float()
        return source, target, secret

    def __len__(self):
        return len(self.target_paths)


class BlackBoxDataModule(LightningDataModule):
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
    host_dir = '/sda1/Datasets/Flickr25k/mirflickr'
    source_dir = '/sda1/Datasets/ImageTranslationPair/Crayon/original'
    target_dir = '/sda1/Datasets/ImageTranslationPair/Crayon/distortion'
    train_data = BlackBoxTrainData(host_dir, source_dir, target_dir)
    for batch in train_data:
        import pdb; pdb.set_trace()
        print('hello')
    pass


if __name__ == '__main__':
    run()
