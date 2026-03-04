#!/usr/bin/env bash

# 选 GPU
export CUDA_VISIBLE_DEVICES=6

python -m lora_adapters.train_lora_mol \
  --domain snow \
  --domains snow \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/snow/train_list_snow.txt \
  --val_list   data/snow/val_list_snow.txt \
  --rank 16 \
  --alpha 16 \
  --epochs 20 \
  --batch 2 \
  --lr 5e-5 \
  --patch 256 \
  --save_dir weights/lora \
  --device cuda


export CUDA_VISIBLE_DEVICES=6

python -m lora_adapters.train_lora_mol \
  --domain snow1 \
  --domains snow1 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/CDD-11_train/train_list_snow1.txt \
  --val_list   data/CDD-11_train/val_list_snow1.txt \
  --rank 16 \
  --alpha 16 \
  --epochs 20 \
  --batch 2 \
  --lr 5e-5 \
  --patch 256 \
  --save_dir weights/lora \
  --device cuda

  