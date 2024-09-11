#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : AdversarialDistortionTranslation
# @File         : datamodule.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/10/14 19:20
#
# Import lib here
from typing import Optional
from functools import partial

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig


class DataModule(LightningDataModule):
    def __init__(self, dataset: partial,
                 dataloader_cfg: DictConfig):
        super().__init__()

        self.dataset = dataset
        self.dataloader_cfg = dataloader_cfg

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`,
        so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.train_data:
            self.train_data = self.dataset(stage='train')
        if not self.val_data:
            self.val_data = self.dataset(stage='val')
        if not self.test_data:
            self.test_data = self.dataset(stage='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, **self.dataloader_cfg, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, **self.dataloader_cfg, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, **self.dataloader_cfg, shuffle=False)


def run():
    pass


if __name__ == '__main__':
    run()
