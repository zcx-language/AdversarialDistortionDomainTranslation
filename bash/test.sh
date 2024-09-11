#!/usr/bin/bash

CUDA_VISIBLE_DEVICES="$1" \
python /home/chengxin/Project/AdversarialDistortionTranslation/test.py \
model=adversarial_distortion_gans \
model.distortion_types="\[jpeg,jpeg2000,webp,rainy,emboss,fisheye,gaussian_blur,gaussian_noise,resize,dropout,cropout\]" \
ckpt_path="$2" \
