#!/usr/bin/env bash

# 选 GPU
export CUDA_VISIBLE_DEVICES=6

python -m lora_adapters.train_lora_mol \
  --domain haze \
  --domains haze \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/haze/train_list_haze.txt \
  --val_list   data/haze/val_list_haze.txt \
  --rank 16 \
  --alpha 16 \
  --epochs 30 \
  --batch 2 \
  --lr 5e-5 \
  --patch 256 \
  --save_dir weights/lora \
  --device cuda

export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.train_lora_mol \
  --domain hazy \
  --domains hazy \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/hazy/train_list_hazy.txt \
  --val_list   data/hazy/val_list_hazy.txt \
  --rank 16 --alpha 16 \
  --epochs 30 --batch 1 \
  --lr 5e-5 \
  --patch 256 \
  --val_patch 256 \
  --save_dir weights/lora \
  --device cuda

  export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.train_lora_mol \
  --domain haze1\
  --domains haze1 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/haze1/train_list_haze1.txt \
  --val_list   data/haze1/val_list_haze1.txt \
  --rank 16 \
  --alpha 16 \
  --epochs 30 \
  --batch 2 \
  --lr 5e-5 \
  --patch 256 \
  --save_dir weights/lora \
  --device cuda