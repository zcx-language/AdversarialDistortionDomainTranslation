#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : Steganography
# @File         : mirflickr_module.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2021/8/4 下午4:32

# Import lib here
from typing import Optional

from pytorch_lightning import LightningDataModule
from omegaconf import dictconfig
from torch.utils.data import Dataset, DataLoader
from datamodules.mirflickr.mirflickr import Mirflickr


class MirflickrModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(self,
                 dataset_cfg: dictconfig,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = False):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

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
        if not self.train_data and not self.val_data and not self.test_data:
            self.train_data = Mirflickr(**self.hparams.dataset_cfg)
            self.val_data = Mirflickr(**self.hparams.dataset_cfg, stage='val')
            self.test_data = Mirflickr(**self.hparams.dataset_cfg, stage='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory, drop_last=self.hparams.drop_last)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory, drop_last=self.hparams.drop_last)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory, drop_last=self.hparams.drop_last)


def run():
    pass


if __name__ == '__main__':
    run()
