#!/usr/bin/env bash

# 选 GPU，例如只用第 6 张
export CUDA_VISIBLE_DEVICES=6

python -m lora_adapters.train_lora_mol \
  --domain rain \
  --domains rain \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/rain/train_list_rain.txt \
  --val_list   data/rain/val_list_rain.txt \
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
  --domain rain2 \
  --domains rain2 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/rain2/train_list_rain2.txt \
  --val_list   data/rain2/val_list_rain2.txt \
  --rank 16 \
  --alpha 16 \
  --epochs 20 \
  --batch 2 \
  --lr 5e-5 \
  --patch 256 \
  --save_dir weights/lora \
  --device cuda

  python -m lora_adapters.train_lora_mol \
  --domain rainy \
  --domains rainy \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/rainy/train_list_rainy.txt \
  --val_list   data/rainy/val_list_rainy.txt \
  --rank 16 \
  --alpha 16 \
  --epochs 30 \
  --batch 2 \
  --lr 5e-5 \
  --patch 256 \
  --save_dir weights/lora \
  --device cuda