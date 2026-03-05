## Diffusion Transformer

![DiT CIFAR10 Samples](assets/step199999_cfg5.0_ema.png)

This repo contains the Diffusion Transformer from the paper **Scalable Diffusion Models with Transformers (DiT)**. [[arxiv](https://arxiv.org/abs/2212.09748)] [[code](https://github.com/facebookresearch/DiT)]. It is a repo created with interest in the combination of diffusion model and transformer model. The code for the network is mostly based on the official implementation from MetaAI. I made several changes to the model with new techniques and tricks.

## Setup
You can recreate the conda environment using the provided environment.yml.
```
conda env create -f environment.yml
conda activate dit
```

## Training
```
python train.py --use_repa --proj_coeff 0.5 --encoder_type dinov2 --encoder_size s --encoder_depth 6
```

## Sampling
```
python evaluate_fid.py --checkpoint checkpoints/20260219_152042/step_100000.pth

python evaluate_cknna.py --checkpoint checkpoints/20260219_152042/step_100000.pth --num_samples 20000

python evaluate_linear_probe.py --checkpoint checkpoints/20260219_152042/step_100000.pth --num_epochs 100
```