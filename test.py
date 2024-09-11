#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : test.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/5/12 下午2:40

# Import lib here
import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

import hydra
import torch
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger

from omegaconf import DictConfig
from utils import utils

from typing import List

log = utils.get_logger(__name__)


@hydra.main(config_path="configs/", config_name="test.yaml", version_base="1.3")
def main(config: DictConfig) -> None:
    utils.extras(config)

    torch.set_float32_matmul_precision('medium')

    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )


if __name__ == '__main__':
    main()