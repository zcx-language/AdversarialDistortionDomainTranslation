#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : __init__.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2022/3/30 下午4:35

# Import lib here
from .hidden_decoder import HiDDeNDecoder
from .stegastamp_decoder import StegaStampDecoder
from .arwgan_decoder import Decoder as ARWGANDecoder
from .pimog_decoder import Decoder as PIMoGDecoder


def run():
    pass


if __name__ == '__main__':
    run()
