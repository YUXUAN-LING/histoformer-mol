#!/usr/bin/env bash

# 选 GPU
export CUDA_VISIBLE_DEVICES=6

python -m lora_adapters.train_lora_mol \
  --domain blur \
  --domains rain,snow,fog,blur \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/blur/train_list_blur.txt \
  --val_list   data/blur/val_list_blur.txt \
  --rank 16 \
  --alpha 16 \
  --epochs 30 \
  --batch 2 \
  --lr 4e-5 \
  --patch 256 \
  --save_dir weights/lora \
  --device cuda




