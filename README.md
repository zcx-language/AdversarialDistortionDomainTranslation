---

<div align="center">

# Adversarial Distortion Domain Translation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Abstract

Deep watermarking models optimize robustness by incorporating distortions between the encoder and decoder. To tackle non-differentiable distortions, current methods only train the decoder with distorted images, which breaks the joint optimization of the encoder-decoder, resulting in suboptimal performance. To address this problem, we propose an Adversarial Distortion Domain Translation (AD$^2$T) method by treating the distortion as an image-to-image translation task. AD$^2$T adopts conditional GANs to learn the non-differentiable distortion mappings. It employs generators to transform the encoded image into the distorted one to bridge the encoder-decoder for joint optimization. We also supervise the GANs to generate challenging distorted samples to augment the watermarking model via adversarial training. This further improves the model robustness by minimizing the maximum decoding loss. Extensive experiments demonstrate the superiority of our method when tested on non-differentiable distortions, including lossy compression and style transfers.


## Visualizations

![Visuallizations](https://chengxin-imagebed.oss-cn-wuhan-lr.aliyuncs.com/images/diff_ndiff2.png)

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/zcx-language/AdversarialDistortionTranslation.git
cd AdversarialDistortionTranslation

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model

```bash
# train AdversarialDistortionGANS
python main.py model=adversarial_distortion_gans.yaml
```

Test model

```bash
# test saved checkpoints
python test.py model=adversarial_distortion_gans.yaml ckpt_path=$YOURPATH
```

You can override any parameter from command line like this

```bash
python main.py trainer.max_epochs=20 datamodule.batch_size=64
```
