#!/usr/bin/bash

python main.py \
task_name=adt \
tags=gans_adv \
model.train_cfg.target_psnr=33 \
model.loss_cfg.wgan.lpips_weight=3. \
model.loss_cfg.wgan.pix_weight=1. \
model.loss_cfg.wgan.msg_weight=5. \
model.loss_cfg.wgan.gan_weight=0.1 \
callbacks=for_adt \
trainer.max_epochs=100 \
test_after_training=true \
