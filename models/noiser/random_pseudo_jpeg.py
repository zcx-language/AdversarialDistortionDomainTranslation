#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : CXTools
# @File         : jpeg.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2024/1/12 14:46
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import os
import random
import string
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf
from PIL import Image
from .ramp_base import RampBase
from functools import partial

Image.MAX_IMAGE_PIXELS = None


class JpegBasic(nn.Module):
    """JPEG basic block"""

    def __init__(self):
        super(JpegBasic, self).__init__()

    def std_quantization(self, image_yuv_dct, scale_factor, round_func=torch.round):

        luminance_quant_tbl = (torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
            image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

        chrominance_quant_tbl = (torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
            image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

        q_image_yuv_dct = image_yuv_dct.clone()
        q_image_yuv_dct[:, :1, :, :] = image_yuv_dct[:, :1, :, :] / luminance_quant_tbl
        q_image_yuv_dct[:, 1:, :, :] = image_yuv_dct[:, 1:, :, :] / chrominance_quant_tbl
        q_image_yuv_dct_round = round_func(q_image_yuv_dct)
        return q_image_yuv_dct_round

    def std_reverse_quantization(self, q_image_yuv_dct, scale_factor):

        luminance_quant_tbl = (torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
            q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

        chrominance_quant_tbl = (torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
            q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

        image_yuv_dct = q_image_yuv_dct.clone()
        image_yuv_dct[:, :1, :, :] = q_image_yuv_dct[:, :1, :, :] * luminance_quant_tbl
        image_yuv_dct[:, 1:, :, :] = q_image_yuv_dct[:, 1:, :, :] * chrominance_quant_tbl
        return image_yuv_dct

    def dct(self, image):
        coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num1 = image.shape[2] // 8
        split_num2 = image.shape[3] // 8
        image_dct = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
        image_dct = torch.matmul(coff, image_dct)
        image_dct = torch.matmul(image_dct, coff.permute(1, 0))
        image_dct = torch.cat(torch.cat(image_dct.chunk(split_num2, 0), 3).chunk(split_num1, 0), 2)
        return image_dct

    def idct(self, image_dct):
        coff = torch.zeros((8, 8), dtype=torch.float).to(image_dct.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num1 = image_dct.shape[2] // 8
        split_num2 = image_dct.shape[3] // 8
        image = torch.cat(torch.cat(image_dct.split(8, 2), 0).split(8, 3), 0)
        image = torch.matmul(coff.permute(1, 0), image)
        image = torch.matmul(image, coff)
        image = torch.cat(torch.cat(image.chunk(split_num2, 0), 3).chunk(split_num1, 0), 2)
        return image

    def rgb2yuv(self, image_rgb):
        image_yuv = torch.empty_like(image_rgb)
        image_yuv[:, 0:1, :, :] = 0.299 * image_rgb[:, 0:1, :, :] \
                                  + 0.587 * image_rgb[:, 1:2, :, :] + 0.114 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 1:2, :, :] = -0.1687 * image_rgb[:, 0:1, :, :] \
                                  - 0.3313 * image_rgb[:, 1:2, :, :] + 0.5 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 2:3, :, :] = 0.5 * image_rgb[:, 0:1, :, :] \
                                  - 0.4187 * image_rgb[:, 1:2, :, :] - 0.0813 * image_rgb[:, 2:3, :, :]
        return image_yuv

    def yuv2rgb(self, image_yuv):
        image_rgb = torch.empty_like(image_yuv)
        image_rgb[:, 0:1, :, :] = image_yuv[:, 0:1, :, :] + 1.40198758 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 1:2, :, :] = image_yuv[:, 0:1, :, :] - 0.344113281 * image_yuv[:, 1:2, :, :] \
                                  - 0.714103821 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 2:3, :, :] = image_yuv[:, 0:1, :, :] + 1.77197812 * image_yuv[:, 1:2, :, :]
        return image_rgb

    def yuv_dct(self, image, subsample):
        image = (image * 255.).clamp(0, 255)

        # pad the image so that we can do dct on 8x8 blocks
        pad_height = (8 - image.shape[2] % 8) % 8
        pad_width = (8 - image.shape[3] % 8) % 8
        image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(image)

        # convert to yuv
        image_yuv = self.rgb2yuv(image)

        assert image_yuv.shape[2] % 8 == 0
        assert image_yuv.shape[3] % 8 == 0

        # subsample
        image_subsample = self.subsampling(image_yuv, subsample)

        # apply dct
        image_dct = self.dct(image_subsample)

        return image_dct, pad_width, pad_height

    def idct_rgb(self, image_quantization, pad_width, pad_height):
        # apply inverse dct (idct)
        image_idct = self.idct(image_quantization)

        # transform from yuv to rgb
        image_ret_padded = self.yuv2rgb(image_idct)

        # un-pad
        image_rgb = image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height,
                    :image_ret_padded.shape[3] - pad_width].clone()

        return (image_rgb / 255.).clamp(0, 1)

    def subsampling(self, image, subsample):
        if subsample == 2:
            split_num1 = image.shape[2] // 8
            split_num2 = image.shape[3] // 8
            image_block = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
            for i in range(8):
                if i % 2 == 1: image_block[:, 1:3, i, :] = image_block[:, 1:3, i - 1, :]
            for j in range(8):
                if j % 2 == 1: image_block[:, 1:3, :, j] = image_block[:, 1:3, :, j - 1]
            image = torch.cat(torch.cat(image_block.chunk(split_num2, 0), 3).chunk(split_num1, 0), 2)
        return image


class _RandomJpeg(JpegBasic):
    """JPEG with random compression quality parameter"""

    def __init__(self, min_quality=50, max_quality=100, subsample=2, padding=False):
        super().__init__()
        self.max_quality = max_quality
        self.min_quality = min_quality
        # subsample
        self.subsample = subsample
        self.padding = padding

    def forward(self, image):
        # quantization table
        self.Q = np.random.randint(self.min_quality, self.max_quality)
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

        if self.padding:
            h = np.random.randint(0, 8)
            w = np.random.randint(0, 8)
            image = nn.ReflectionPad2d((w, w, h, h))(image)

        # [0, 1] to [0,255], rgb2yuv, dct
        image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

        # quantization
        image_quantization = self.std_quantization(image_dct, self.scale_factor)

        # reverse quantization
        image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

        # idct, yuv2rgb, [0,255] to [0, 1]
        noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)

        if self.padding:
            noised_image = noised_image[:, :, h:noised_image.shape[2] - h, w:noised_image.shape[3] - w]

        return noised_image


class RandomJpeg(RampBase):
    def __init__(self,
                 quality: Tuple[int, int] = (50, 100),
                 p: float = 1.,
                 step0: int = 0,
                 ramp: int = 10000):
        super().__init__(step0=step0, ramp=ramp)
        self.quality = quality
        self.p = p

        self.jpeg = partial(_RandomJpeg, max_quality=quality[1])

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return inputs

        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        degrade = self.quality[1] - self.quality[0]
        min_quality = int(self.quality[1] - degrade * ramp_ratio)
        jpeg = self.jpeg(min_quality=min_quality)
        return jpeg(inputs)


class _RandomJpegSS(JpegBasic):
    """JPEG-SS with random compression quality parameter"""

    def __init__(self, min_quality=50, max_quality=100, subsample=2, padding=False):
        super().__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality
        # subsample
        self.subsample = subsample
        self.padding = padding

    def round_ss(self, x):
        cond = (torch.abs(x) < 0.5)
        cond = cond.float()
        cond = cond.to(x.device)
        return cond * (x ** 3) + (1 - cond) * x

    def forward(self, image):

        # quantization table
        self.Q = np.random.randint(self.min_quality, self.max_quality)
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

        if self.padding:
            h = np.random.randint(0, 8)
            w = np.random.randint(0, 8)
            image = nn.ReflectionPad2d((w, w, h, h))(image)

        # self.subsample = np.random.choice(a=[0,2],p=[0.5,0.5])
        # [0,1] to [0,255], rgb2yuv, dct
        image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

        # quantization
        image_quantization = self.std_quantization(image_dct, self.scale_factor, self.round_ss)

        # reverse quantization
        image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

        # idct, yuv2rgb, [0,255] to [0,1]
        noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)

        if self.padding:
            noised_image = noised_image[:, :, h:noised_image.shape[2] - h, w:noised_image.shape[3] - w]
        return noised_image


class RandomJpegSS(RampBase):
    def __init__(self,
                 quality: Tuple[int, int] = (50, 100),
                 p: float = 1.,
                 step0: int = 0,
                 ramp: int = 10000):
        super().__init__(step0=step0, ramp=ramp)
        self.quality = quality
        self.p = p

        self.jpeg_ss = partial(_RandomJpegSS, max_quality=quality[1])

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return inputs

        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        degrade = self.quality[1] - self.quality[0]
        min_quality = int(self.quality[1] - degrade * ramp_ratio)
        jpeg_ss = self.jpeg_ss(min_quality=min_quality)
        return jpeg_ss(inputs)


class _RandomJpegMask(JpegBasic):
    """JPEGMask with random compression quality parameter"""

    def __init__(self, min_quality=50, max_quality=100, subsample=2, padding=False):
        super().__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.padding = padding

        # subsample
        self.subsample = subsample

    def round_mask(self, x):
        mask = torch.zeros(1, 3, 8, 8).to(x.device)
        mask[:, 0:1, :5, :5] = 1
        mask[:, 1:3, :3, :3] = 1
        mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
        return x * mask

    def forward(self, image):
        # quantization table
        self.Q = np.random.randint(self.min_quality, self.max_quality)
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

        if self.padding:
            h = np.random.randint(0, 9)
            w = np.random.randint(0, 9)
            image = nn.ReflectionPad2d((w, w, h, h))(image)

        # self.subsample = np.random.choice(a=[0,2],p=[0.5,0.5])
        # [0,1] to [0,255], rgb2yuv, dct
        image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

        # mask
        image_mask = self.round_mask(image_dct)

        # idct, yuv2rgb, [0,255] to [0,1]
        noised_image = self.idct_rgb(image_mask, pad_width, pad_height)

        if self.padding:
            noised_image = noised_image[:, :, h:noised_image.shape[2] - h, w:noised_image.shape[3] - w]

        return noised_image


class RandomJpegMask(RampBase):
    def __init__(self,
                 quality: Tuple[int, int] = (50, 100),
                 p: float = 1.,
                 step0: int = 0,
                 ramp: int = 10000):
        super().__init__(step0=step0, ramp=ramp)
        self.quality = quality
        self.p = p

        self.jpeg_mask = partial(_RandomJpegMask, max_quality=quality[1])

    def __call__(self, inputs: torch.Tensor, step: int = None) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return inputs

        ramp_ratio = self.ramp_fn(step) if step is not None else 1.
        degrade = self.quality[1] - self.quality[0]
        min_quality = int(self.quality[1] - degrade * ramp_ratio)
        jpeg_mask = self.jpeg_mask(min_quality=min_quality)
        return jpeg_mask(inputs)


class RandomJpegTest(nn.Module):
    """Real JPEG with random compression quality parameter"""

    def __init__(self, min_quality=50, max_quality=100, subsample=2, path="temp/"):
        super(RandomJpegTest, self).__init__()
        self.subsample = subsample
        self.path = path
        self.max_quality = max_quality
        self.min_quality = min_quality
        if not os.path.exists(path):
            os.mkdir(path)

    def get_path(self):
        return self.path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".jpg"

    def forward(self, image):
        self.Q = np.random.randint(self.min_quality, self.max_quality)

        jpeg_image = []
        image_ary = (image * 255.).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        for i in range(image.shape[0]):
            single_image = image_ary[i]
            im = Image.fromarray(single_image)
            file = self.get_path()
            while os.path.exists(file):
                file = self.get_path()
            im.save(file, format="JPEG", quality=self.Q, subsampling=self.subsample)

            jpeg = Image.open(file).convert('RGB')
            jpeg_image.append(tvf.to_tensor(jpeg))
            os.remove(file)

        return torch.stack(jpeg_image, dim=0).to(image.device)


def run():
    from PIL import Image
    from matplotlib import pyplot as plt
    import numpy as np
    import torchvision.transforms.functional as tvf

    img = Image.open('/home/chengxin/Desktop/my.png').convert('RGB')
    img = img.resize((256, 256))
    img = tvf.to_tensor(img).unsqueeze(0)

    d_img_list = []
    ramp_distortion = RandomJpegSS()
    for i in list([0, 2500, 5000, 7500, 10000]):
        d_img_list.append(ramp_distortion(img, i).squeeze(0).permute(1, 2, 0).numpy())

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, ax in enumerate(axes):
        ax.imshow(d_img_list[i])
        ax.set_title(f'step: {i * 2500}')
    plt.show()
    pass


if __name__ == '__main__':
    run()
