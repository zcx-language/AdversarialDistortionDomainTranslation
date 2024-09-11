#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : KPTG2
# @File         : image_tools.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2021/3/26 下午3:02

# Import lib here
import numpy as np
import torch
import math
import cv2
import matplotlib as mpl
from pathlib import Path

from torchvision import transforms
from matplotlib import pyplot as plt
from typing import Union, Optional, List, Tuple


def image_tensor2numpy(tensor: torch.Tensor, normalize: bool = False, value_range: Optional[Tuple[int, int]] = None,
                       scale_each: bool = True, keep_dims: bool = False) -> np.ndarray:
    """Converts a PyTorch tensor to a numpy image.
    In case the tensor is in the GPU, it will be copied back to CPU.

    Args:
        tensor (torch.Tensor): image of the form :math:`(H, W)`, :math:`(C, H, W)` or :math:`(B, C, H, W)`.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool): If ``True``, scale each image in the batch of
            images separately rather than over all images. Default: ``True``.
        keep_dims (bool): if ``False``, squeeze the input tensor.

    Returns:
        numpy.ndarray: image of the form :math:`(H, W)`, :math:`(H, W, C)` or :math:`(B, H, W, C)`.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    input_shape = tensor.shape

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        pass
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        if input_shape[0] == 1:
            tensor = tensor.squeeze()
        else:
            tensor = tensor.permute(1, 2, 0)
    elif len(input_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        tensor = tensor.permute(0, 2, 3, 1)
        if input_shape[0] == 1 and not keep_dims:
            tensor = tensor.squeeze(0)
        if input_shape[1] == 1:
            tensor = tensor.squeeze(-1)

    else:
        raise ValueError("Cannot process tensor with shape {}".format(input_shape))

    if normalize is True:
        tensor = tensor.clone()
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if len(input_shape) == 4 and scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    assert tensor.min() >= 0. and tensor.max() <= 1.
    images = tensor.mul(255.).add_(0.5).clamp_(0, 255).byte()
    return images.numpy()


def images_save(images: Union[torch.Tensor, np.ndarray], folder: str, gray: bool = False) -> None:
    """Save torch.Tensor or np.ndarray format images

    Args:
        images (Union[torch.Tensor, np.ndarray]):
            if image is torch.Tensor, need :math:`(H, W)`, :math:`(C, H, W)` or :math:`(B, C, H, W)`.
            if image is np.ndarray, need :math:`(H, W)`, :math:`(H, W, C)` or :math:`(B, H, W, C)`.

        folder (str): the folder that images to save
        gray (bool): indicate the images are gray or not
    """

    if isinstance(images, torch.Tensor):
        images = image_tensor2numpy(images)
    if isinstance(images, np.ndarray):
        images = images.squeeze()
    # here images shape may be (B, H, W, C) or (H, W, C) if gray is False
    # else images shape may be (B, H, W) or (H, W)

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    images_shape = images.shape
    if gray is False:
        if len(images_shape) == 4:
            bits = int(math.log10(images_shape[0])) + 1
            for idx, image in enumerate(images):
                image_path = folder / f'{idx:0{bits}d}.jpg'
                plt.imsave(image_path, image)
        elif len(images_shape) == 3:
            image = images
            image_path = folder / '0.jpg'
            plt.imsave(image_path, image)
        else:
            raise ValueError("Cannot process images with shape {}".format(images_shape))

    else:
        if len(images_shape) == 3:
            bits = int(math.log10(images_shape[0])) + 1
            for idx, image in enumerate(images):
                image_path = folder / f'{idx:0{bits}d}.jpg'
                plt.imsave(image_path, image, cmap='gray')
        elif len(images_shape) == 2:
            image = images
            image_path = folder / '0.jpg'
            plt.imsave(image_path, image, cmap='gray')
        else:
            raise ValueError("Cannot process images with shape {}".format(images_shape))
    return


def image_show(image: Union[torch.Tensor, np.ndarray], figsize: Tuple = (0, 0)) -> None:
    """Show image of torch.Tensor or numpy.array format

    Args:
        image (Union[torch.Tensor, np.ndarray]): the image to show
        figsize (Tuple): the figure size to show

    Returns:
        None
    """

    if isinstance(image, torch.Tensor):
        image = image_tensor2numpy(image)

    # What size does the figure need to be in inches to fit the image?
    if figsize == (0, 0):
        dpi = mpl.rcParams['figure.dpi']
        height, width = image.shape[:2]
        figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(image, cmap='gray')
    plt.show()


def image_color2gray(image: np.ndarray, color: str = 'rgb') -> np.ndarray:
    """ Convert image to gray schema

    Args:
        image (np.ndarray):
        color (str):

    Returns:
        np.ndarray
    """

    color = color.lower()
    assert color in {'rgb', 'bgr'}, f'Error, not support color: {color}'
    ndim = image.ndim
    if ndim == 3:
        if color == 'rgb':
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif ndim == 2:
        gray = image.copy()
    else:
        raise ValueError(f'Error, not support image with dimension: {ndim}')

    return gray


def image_split(image: np.ndarray, row: int, col: int, resize=True) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """

    Args:
        image ():
        row ():
        col ():
        resize ():

    Returns:

    """

    height, width = image.shape[:2]
    if resize:
        patch_height, patch_width = height // row, width // col
        height -= height % patch_height
        width -= width % patch_width
        image = cv2.resize(image, (width, height))
    else:
        assert height % row == 0 and width % col == 0
        patch_height, patch_width = height // row, width // col

    patches = []
    offsets = []
    for i in range(row):
        for j in range(col):
            patch = image[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width, ...]
            patches.append(patch)
            offsets.append((i*patch_height, j*patch_width))
    return patches, offsets


def image_assemble(patches: List[np.ndarray], row: int, col: int) -> np.ndarray:
    """

    Args:
        patches ():
        row ():
        col():

    Returns:

    """

    image = cv2.vconcat([cv2.hconcat([patches[i*col+j] for j in range(col)]) for i in range(row)])
    return image


def image_compression_by_cv2(image: torch.Tensor, quality: int) -> torch.Tensor:
    """Compress images by opencv (Obsoleted)

    Args:
        image (torch.Tensor): N, C, H, W
        quality (int):
    """
    assert image.ndim in {3, 4}, f'Error, need shape of 3(3, H, W) or 4(N, 3, H, W) but get {image.ndim}'
    ary_image = image_tensor2numpy(image)
    if ary_image.ndim == 3:
        _, encoded_img = cv2.imencode('.jpg', ary_image, (int(cv2.IMWRITE_JPEG_QUALITY), quality))
        ary_image = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
        tsr_image = torch.tensor(ary_image, dtype=torch.float32, device=image.device) / 255.
    elif ary_image.ndim == 4:
        for i in range(ary_image.shape[0]):
            _, encoded_img = cv2.imencode('.jpg', ary_image[i], (int(cv2.IMWRITE_JPEG_QUALITY), quality))
            ary_image[i] = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
        tsr_image = torch.tensor(ary_image.transpose((0, 3, 1, 2)), dtype=torch.float32, device=image.device) / 255.
    else:
        raise NotImplementedError
    return tsr_image


def run():
    image_path = '/home/chengxin/Desktop/lena.png'
    image_ary = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_ary = cv2.resize(image_ary, (512, 512))
    image_tsr = transforms.ToTensor()(image_ary)
    image_show(image_tsr)
    image_tsr = image_compression_by_cv2(image_tsr.unsqueeze(0), quality=10)
    image_show(image_tsr[0])
    pass


if __name__ == '__main__':
    run()
