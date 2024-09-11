#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : lossy_compression.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/5/15 下午10:29

# Import lib here
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from kornia.utils import tensor_to_image
from utils.image_tools import image_show
from typing import Tuple, List, Any, Optional, Union


def image_lossy_compression(image: np.ndarray, quality_factor: int = 95, ext: str = '.jpg') -> np.ndarray:
    if ext in ['.jpeg', '.jpg']:
        quality_flag = cv2.IMWRITE_JPEG_QUALITY
    elif ext == '.jp2':
        quality_flag = cv2.IMWRITE_JPEG2000_COMPRESSION_X1000
    elif ext == '.webp':
        quality_flag = cv2.IMWRITE_WEBP_QUALITY
    else:
        raise NotImplementedError("Only 'jpeg', 'jpeg2000' and 'webp' compression transforms are implemented. ")

    # `cv2.imwrite` bgr format image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    buffer = cv2.imencode(ext, image, (int(quality_flag), quality_factor))[1]
    img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def tensor_lossy_compression(tensor: torch.Tensor, quality_factor: int = 95, ext: str = '.jpg') -> torch.Tensor:
    tensor_shape = tensor.shape
    ary = (tensor * 255. + 0.5).clamp(0., 255.).detach().cpu().numpy().astype(np.uint8)

    if len(tensor_shape) == 3:
        compressed_img = image_lossy_compression(ary.transpose((1, 2, 0)), quality_factor, ext)
        compressed_tsr = torch.tensor(compressed_img.transpose((2, 0, 1)), dtype=torch.float32, device=tensor.device)
    elif len(tensor_shape) == 4:
        compressed_imgs = []
        for ins in ary:
            compressed_img = image_lossy_compression(ins.transpose((1, 2, 0)), quality_factor, ext)
            compressed_imgs.append(compressed_img)
        compressed_imgs = np.stack(compressed_imgs, axis=0)
        compressed_tsr = torch.tensor(compressed_imgs, dtype=torch.float32, device=tensor.device).permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Cannot process tensor with shape {tensor_shape}")
    return compressed_tsr / 255.


class JpegOpenCV:
    def __init__(self, quality_factor: Union[int, List]):
        super().__init__()
        self.quality_factor = quality_factor

    def __call__(self, image: torch.tensor) -> torch.tensor:
        if isinstance(self.quality_factor, int):
            quality_factor = self.quality_factor
        else:
            quality_factor = np.random.randint(self.quality_factor[0], self.quality_factor[1])
        return tensor_lossy_compression(image, quality_factor, ext='.jpg')


class WebpOpenCV:
    def __init__(self, quality_factor: Union[int, List]):
        super().__init__()
        self.quality_factor = quality_factor

    def __call__(self, image: torch.tensor) -> torch.tensor:
        if isinstance(self.quality_factor, int):
            quality_factor = self.quality_factor
        else:
            quality_factor = np.random.randint(self.quality_factor[0], self.quality_factor[1])
        return tensor_lossy_compression(image, quality_factor, ext='.webp')


class Jpeg2000OpenCV:
    def __init__(self, compression_ratio: Union[int, List]):
        super().__init__()
        self.compression_ratio = compression_ratio

    def __call__(self, image: torch.tensor) -> torch.tensor:
        if isinstance(self.compression_ratio, int):
            compression_ratio = self.compression_ratio
        else:
            compression_ratio = np.random.randint(self.compression_ratio[0], self.compression_ratio[1])
        return tensor_lossy_compression(image, compression_ratio, ext='.jp2')


def run():
    image_path = '/sda1/Datasets/Flickr25k/mirflickr/im90.jpg'
    image_ary = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_ary = cv2.resize(image_ary, (512, 512))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tsr = transform(image_ary)
    image_show(image_tsr[:, 100:300, 100:300])
    image_tsr = torch.stack([image_tsr, image_tsr], dim=0)
    compressed_tsr = tensor_lossy_compression(image_tsr, quality_factor=50, ext='.jpg')
    image_show(compressed_tsr[1][:, 100:300, 100:300])
    compressed_tsr = tensor_lossy_compression(image_tsr, quality_factor=50, ext='.webp')
    image_show(compressed_tsr[1][:, 100:300, 100:300])
    compressed_tsr = tensor_lossy_compression(image_tsr, quality_factor=50, ext='.jp2')
    image_show(compressed_tsr[1][:, 100:300, 100:300])
    pass


if __name__ == '__main__':
    run()
