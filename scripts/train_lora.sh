export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.train_lora_mol \
  --domain haze2\
  --domains haze2 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/CDD-11_train/train_list_haze2.txt \
  --val_list   data/CDD-11_train/val_list_haze2.txt \
  --rank 16 \
  --alpha 16 \
  --epochs 30 \
  --batch 2 \
  --lr 5e-5 \
  --patch 256 \
  --save_dir weights/lora \
  --device cuda

export CUDA_VISIBLE_DEVICES=5
python -m lora_adapters.train_lora_mol \
  --domain rain3\
  --domains rain3 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/CDD-11_train/train_list_rain3.txt \
  --val_list   data/CDD-11_train/val_list_rain3.txt \
  --rank 16 \
  --alpha 16 \
  --epochs 30 \
  --batch 2 \
  --lr 5e-5 \
  --patch 256 \
  --save_dir weights/lora \
  --device cuda

export CUDA_VISIBLE_DEVICES=4
python -m lora_adapters.train_lora_mol \
  --domain haze_snow\
  --domains haze_snow \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/CDD-11_train/train_list_haze_snow.txt \
  --val_list   data/CDD-11_train/val_list_haze_snow.txt \
  --rank 16 \
  --alpha 16 \
  --epochs 20 \
  --batch 2 \
  --lr 5e-5 \
  --patch 256 \
  --save_dir weights/lora \
  --device cuda



  export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=4
python -m lora_adapters.train_lora_regular \
  --domain snow1 \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --train_list data/CDD-11_train/train_list_snow1.txt \
  --val_list data/CDD-11_train/val_list_snow1.txt \
  --rank 16 --alpha 16 \
  --epochs 10 --lr 5e-5 \
  --batch 1 --patch 256 \
  --save_dir weights/lora_ortho \
  --init_lora_ckpt weights/lora/snow1/lora.pth \
  --ortho_lambda 3e-4 \
  --ortho_mode both \
  --ortho_domains haze2 \
  --ortho_loradb weights/lora \
  --seed 42 --deterministic


  export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=4
python -m lora_adapters.train_lora_layerwise \
  --domain snow1_local \
  --domains snow1_local \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --train_list data/CDD-11_train/train_list_snow1.txt \
  --val_list data/CDD-11_train/val_list_snow1.txt \
  --rank 16 --alpha 16 \
  --epochs 5 --lr 5e-5 --batch 1 \
  --patch 256  \
  --enable_prefixes encoder_level1,encoder_level2,decoder_level1,refinement \
  --init_lora_ckpt weights/lora/snow1/lora.pth \
  --zero_disabled_up \
  --save_dir weights/lora_lw \
  --seed 42 --deterministic

python -m lora_adapters.train_lora_layerwise \
  --domain haze2_global \
  --domains haze2_global \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --train_list data/CDD-11_train/train_list_haze2.txt \
  --val_list data/CDD-11_train/val_list_haze2.txt \
  --rank 16 --alpha 16 \
  --epochs 5 --lr 5e-5 --batch 1 \
  --patch 256 \
  --enable_prefixes encoder_level3,latent,decoder_level2,decoder_level3 \
  --init_lora_ckpt weights/lora/haze2/lora.pth \
  --zero_disabled_up \
  --save_dir weights/lora_lw \
  --seed 42 --deterministic

