#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : ssim.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/4/28 下午4:29

# Import lib here
import torch
from torchmetrics import Metric
from torchmetrics.functional.image import structural_similarity_index_measure
from typing import Any, Optional


class SSIM(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state('ssim', torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('num', torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target) -> None:
        self.ssim += structural_similarity_index_measure(preds, target)
        self.num += 1

    def compute(self):
        return 1.0 * self.ssim / self.num


def run():
    pass


if __name__ == '__main__':
    run()
