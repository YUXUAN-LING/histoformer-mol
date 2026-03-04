python -m lora_adapters.build_prototypes \
  --domains rain2,low,rain,snow,haze,haze1,rainy \
  --data_root data \
  --txt_root data/txt_lists \
  --txt_suffix _train.txt \
  --out weights/prototypes/dinov2_vitb14.pt \
  --loradb_root weights/lora \
  --max_images 1000



###dino v2 prototypes
python -m lora_adapters.build_prototypes \
  --embedder dino_v2 \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --data_root data \
  --txt_root data/txt_lists \
  --out weights/prototypes/dinov2_vitb14.pt \
  --loradb_root weights/lora \
  --max_images 1000 \
  --dino_ckpt weights/dinov2/vit_base_patch14_dinov2.lvd142m-384.pth\
  --k_proto 3 \

###clip prototypes
python -m lora_adapters.build_prototypes \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --data_root data \
  --txt_root data/txt_lists \
  --out weights/prototypes/clip_vitb16.pt \
  --loradb_root weights/lora \
  --max_images 1000

###fft prototypes
python -m lora_adapters.build_prototypes \
  --embedder fft \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --data_root data \
  --txt_root data/txt_lists \
  --out weights/prototypes/fft.pt \
  --loradb_root weights/lora \
  --max_images 1000 \



python -m lora_adapters.build_fft_clean_proto \
  --txt_root data/txt_lists \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --data_root data \
  --out weights/fft_clean_proto.npy \
  --max_images 5000 \
  --fft_resize 256 \
  --radial_bins 32 \
  --angle_bins 16 \
  --patch_size 32 \
  --gt_keywords "gt,hq,clean,norain,clear"
###fft enhanced prototypes