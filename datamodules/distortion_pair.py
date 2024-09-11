#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : distortion_pair.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/12 21:13
#
# Import lib here
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union


class DistortionPair(Dataset):
    def __init__(self, data_root: str = '../datasets/ImageTranslationPair/',
                 distortion_dir: str = 'jpeg/qf=50',
                 img_shape: Tuple[int, int, int] = (3, 128, 128),
                 stage: str = 'train',
                 train_valid_test_nums: Tuple = (1000, 10, 10),
                 target2diff: bool = False):
        super().__init__()
        orig_dir = Path(data_root) / 'original'
        dist_dir = Path(data_root) / distortion_dir
        # orig_paths = sorted(str(path) for path in orig_dir.glob('*.jpg'))
        dist_paths = sorted(path for path in dist_dir.glob('*'))

        if stage == 'train':
            beg_idx, end_idx = 0, train_valid_test_nums[0]
        elif stage == 'val':
            beg_idx, end_idx = train_valid_test_nums[0], train_valid_test_nums[0]+train_valid_test_nums[1]
        elif stage == 'test':
            beg_idx, end_idx = (train_valid_test_nums[0]+train_valid_test_nums[1],
                                train_valid_test_nums[0]+train_valid_test_nums[1]+train_valid_test_nums[2])
        else:
            raise ValueError

        self.orig_dir = orig_dir
        self.dist_paths = dist_paths[beg_idx:end_idx]
        self.target2diff = target2diff

        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        self.random_crop = transforms.RandomCrop(img_shape[-2:], pad_if_needed=True, padding_mode='reflect')

    def __getitem__(self, item):
        dist_path = self.dist_paths[item]
        orig_path = self.orig_dir / f'{dist_path.stem}.jpg'
        orig = self.to_tensor_norm(Image.open(orig_path))
        dist = self.to_tensor_norm(Image.open(dist_path))
        crop_orig_dist = self.random_crop(torch.cat([orig, dist], dim=0))
        orig, dist = crop_orig_dist[:3], crop_orig_dist[3:]
        if self.target2diff:
            # [-1, 1] - [-1, 1] = [-2, 2]
            # [-2, 2] / 2. = [-1, 1]
            dist = (dist - orig) * 0.5
        return orig, dist

    def __len__(self):
        return len(self.dist_paths)


def run():
    dataset = DistortionPair()
    data = dataset.__getitem__(0)
    import pdb; pdb.set_trace()
    print("hello")
    pass


if __name__ == '__main__':
    run()
