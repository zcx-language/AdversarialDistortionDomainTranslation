#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : bit_accuracy.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2022/3/30 下午4:58

# Import lib here
import torch
from torchmetrics import Metric
from typing import Optional, Any


class BitAccuracy(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state('identical_bits', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('bits', torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target) -> None:
        self.identical_bits += torch.sum(preds.eq(target))
        self.bits += target.numel()

    def compute(self):
        return 1.0 * self.identical_bits / self.bits


def get_bit_accuracy(msg: torch.Tensor, msg_hat: torch.Tensor):
    batch_size, msg_len = msg.shape
    correct_bits = torch.sum((msg_hat - msg) == 0, dim=1)
    str_acc = torch.sum((correct_bits - msg_len) == 0) / batch_size
    bit_acc = torch.sum(correct_bits) / msg_hat.numel()
    return bit_acc, str_acc


def run():
    from torchmetrics.classification import MultilabelAccuracy
    tm_bit_acc = MultilabelAccuracy(num_labels=30)
    bit_acc = BitAccuracy()
    pred = torch.randn(10, 30)
    target = torch.randint(0, 2, (10, 30))
    print(tm_bit_acc(pred, target))
    print(bit_acc(pred.sigmoid().ge(0.5), target))
    pass


if __name__ == '__main__':
    run()
