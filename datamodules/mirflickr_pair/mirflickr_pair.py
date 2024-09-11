#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : mirflickr_pair.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2022/3/28 下午5:26

# Import lib here
import numpy as np
import albumentations as A
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union


class MirflickrPair(Dataset):
    def __init__(self,
                 data_dir: str = './data/mirflickr/',
                 stage: str = 'train',
                 img_shape: Tuple[int, int, int] = (3, 400, 400)):
        super().__init__()
        assert stage.lower() in ['train', 'val', 'test']

        channels, height, width = img_shape
        self.img_size = (height, width)
        img_paths = sorted(str(path) for path in Path(data_dir).rglob('*.jpg'))
        num_img = len(img_paths)
        if stage.lower() == 'val':
            self.img_paths = img_paths[int(num_img*0.95):]
        elif stage.lower() == 'test':
            self.img_paths = img_paths[int(num_img*0.9):int(num_img*0.95)]
        else:
            self.img_paths = img_paths[:int(num_img*0.9)]

        self.jpeg_compress = A.ImageCompression(10, 90, always_apply=True)
        self.gaussian_noise = A.GaussNoise(always_apply=True)
        self.gaussian_blur = A.GaussianBlur(always_apply=True)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img = Image.open(img_path).convert('RGB').resize(self.img_size)
        img_np = np.array(img, dtype=np.uint8)
        p = random.random()
        if p < 0.5:
            label = 1
            cp_img_np = self.jpeg_compress(image=img_np)['image']
            pair_img = np.concatenate([img_np, cp_img_np], axis=2)
        else:
            label = 0
            cp_img_np = self.gaussian_noise(image=img_np)['image']
            pair_img = np.concatenate([img_np, cp_img_np], axis=2)

        pair_img_tsr = self.transforms(pair_img)
        return pair_img_tsr, torch.tensor([label], dtype=torch.float32)

    def __len__(self):
        return len(self.img_paths)


def run():
    pass


if __name__ == '__main__':
    run()
