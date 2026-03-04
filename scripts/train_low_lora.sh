export CUDA_VISIBLE_DEVICES=6

python -m lora_adapters.train_lora_mol \
  --domain low \
  --domains low \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/low/train_list_low.txt \
  --val_list   data/low/val_list_low.txt \
  --rank 16 \
  --alpha 16 \
  --epochs 30 \
  --batch 2 \
  --lr 4e-5 \
  --patch 256 \
  --save_dir weights/lora \
  --device cuda