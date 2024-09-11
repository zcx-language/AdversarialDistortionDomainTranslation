#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : Steganography
# @File         : mirflickr.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2021/8/4 下午5:23

# Import lib here
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union


class Mirflickr(Dataset):
    def __init__(self,
                 data_dir: str = './data/mirflickr/',
                 img_shape: Tuple[int, int, int] = (3, 400, 400),
                 secret_len: int = 100,
                 stage: str = 'train'):
        super().__init__()

        img_paths = sorted(path for path in Path(data_dir).glob('*.jpg'))
        num_img = len(img_paths)

        if stage == 'train':
            img_paths = img_paths[:int(num_img*0.9)]
        elif stage == 'val':
            img_paths = img_paths[int(num_img*0.9):int(num_img*0.95)]
        elif stage == 'test':
            img_paths = img_paths[int(num_img*0.95):]
        else:
            raise ValueError

        self.img_paths = img_paths
        self.secret_len = secret_len

        transform = [
            transforms.RandomCrop(img_shape[-2:], pad_if_needed=True, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
        self.transform = transforms.Compose(transform)

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img = Image.open(img_path).convert('RGB')
        img_tsr = self.transform(img)

        secret = torch.randint(0, 2, (self.secret_len,)).float()
        return img_tsr, secret

    def __len__(self):
        return len(self.img_paths)


def run():
    dataset = Mirflickr()
    for batch in dataset:
        print(batch)
    pass


if __name__ == '__main__':
    run()
