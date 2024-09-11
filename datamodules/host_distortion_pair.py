#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : host_distortion_pair.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/22 14:50
#
# Import lib here
import numpy as np
from typing import Tuple, List, Optional, Callable, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image


class HostDistortionPair(Dataset):
    def __init__(self, host_dir: str,
                 distortion_dataset: Dataset,
                 img_shape: Tuple = (3, 128, 128),
                 secret_len: int = 30,
                 stage: str = 'train'):
        super().__init__()

        host_paths = sorted(path for path in Path(host_dir).glob('*.jpg'))
        num_host = len(host_paths)

        if stage == 'train':
            host_paths = host_paths[:int(num_host*0.9)]
        elif stage == 'val':
            host_paths = host_paths[int(num_host*0.9):int(num_host*0.95)]
        elif stage == 'test':
            host_paths = host_paths[int(num_host*0.95):]
        else:
            raise ValueError

        self.host_paths = host_paths
        self.distortion_dataset = distortion_dataset
        self.secret_len = secret_len
        self.stage = stage

        self.host_transform = transforms.Compose([
            transforms.RandomCrop(img_shape[-2:], pad_if_needed=True, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __getitem__(self, item):
        host_path = self.host_paths[item]
        host = self.host_transform(Image.open(host_path).convert('RGB'))
        secret = torch.randint(0, 2, (self.secret_len,)).float()

        if self.stage == 'train':
            item = np.random.randint(len(self.distortion_dataset))
            img_distortion_pair = self.distortion_dataset.__getitem__(item)
            return host, secret, *img_distortion_pair
        else:
            return host, secret

    def __len__(self):
        return len(self.host_paths)


def run():
    from matplotlib import pyplot as plt
    from distortion_pair import DistortionPair
    dist_pair = DistortionPair()
    dataset = HostDistortionPair('../datasets/mirflickr', dist_pair)
    for batch in dataset:
        cover, secret, orig, dt = batch
        print(cover.shape, secret.shape, orig.shape, dt.shape)
        plt.imshow(cover.permute(1, 2, 0))
        plt.show()
        plt.imshow(orig.permute(1, 2, 0))
        plt.show()
        plt.imshow(dt.permute(1, 2, 0))
        plt.show()
        input()
    pass


if __name__ == '__main__':
    run()
