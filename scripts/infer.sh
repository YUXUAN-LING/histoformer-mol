python -m lora_adapters.infer_retrieval \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/haze \
  --output mix_results/mix_haze \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda:0 \
  --gt_root samples/clear \
  --metrics_csv mix_results/mix_haze/metrics.csv

python -m lora_adapters.infer_retrieval \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/rain \
  --output mix_results/mix_rain \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,rainy,haze1 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda:0 \
  --gt_root samples/clear \
  --metrics_csv mix_results/mix_rain/metrics.csv



  python -m lora_adapters.infer_retrieval \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input data/rainy/val/lq \
  --output mix_results/mix_rain \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,rainy,haze1 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda:0 \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07   \
  --gt_root data/rainy/val/gt \
  --metrics_csv mix_results/mix_rain/metrics.csv
  
  # --embedder dino_v2 \
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_retrieval \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/haze \
  --output mix_results/mix_haze \
  --loradb weights/lora \
  --domains low,haze,haze1,snow,rain,rainy,rain2,rain3,low1,haze2 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda:0 \
  --embedder dino_v2\
  --sim_metric euclidean \
  --temperature 0.07   \
  --gt_root samples/clear \
  --metrics_csv mix_results/mix_haze/metrics.csv

export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_retrieval \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/rain \
  --output mix_results/mix_rain \
  --loradb weights/lora \
  --domains low,haze,haze1,snow,rain,rainy,rain2,rain3,low1,haze2 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda:0 \
  --embedder fft\
  --sim_metric euclidean \
  --temperature 0.07   \
  --gt_root samples/clear \
  --metrics_csv mix_results/mix_rain/metrics.csv







###########rain

# 1) DINO, cosine, tau=0.07
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/rain \
  --output results/rain_dino_cos_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder dino_v2 \
  --sim_metric cosine \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/rain_dino_cos_007/metrics.csv \
  --summary_csv results/rain_summary.csv \
  --run_name rain_dino_cos_007

# 2) Clip, cosine, tau=0.07
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/rain \
  --output results/rain_clip_cos_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric cosine \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/rain_clip_cos_007/metrics.csv \
  --summary_csv results/rain_summary.csv \
  --run_name rain_clip_cos_007


# 3)fft, cosine, tau=0.07
export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/rain \
  --output results/rain_fft_cos_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder fft \
  --sim_metric cosine \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/rain_fft_cos_007/metrics.csv \
  --summary_csv results/rain_summary.csv \
  --run_name rain_fft_cos_007



# 1) DINO, euclidean, tau=0.07
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/rain \
  --output results/rain_dino_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder dino_v2 \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/rain_dino_euclidean_007/metrics.csv \
  --summary_csv results/rain_summary.csv \
  --run_name rain_dino_euclidean_007

# 2) Clip, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/rain \
  --output results/rain_clip_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/rain_clip_euclidean_007/metrics.csv \
  --summary_csv results/rain_summary.csv \
  --run_name rain_clip_euclidean_007


# 3)fft, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/rain \
  --output results/rain_fft_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder fft \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/rain_fft_euclidean_007/metrics.csv \
  --summary_csv results/rain_summary.csv \
  --run_name rain_fft_euclidean_007


#########low


  # 1) DINO, euclidean, tau=0.07
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/low \
  --output results/low_dino_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder dino_v2 \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/low_dino_euclidean_007/metrics.csv \
  --summary_csv results/low_summary.csv \
  --run_name low_dino_euclidean_007

# 2) Clip, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/low \
  --output results/low_clip_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/low_clip_euclidean_007/metrics.csv \
  --summary_csv results/low_summary.csv \
  --run_name low_clip_euclidean_007


# 3)fft, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/low \
  --output results/low_fft_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder fft \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/low_fft_euclidean_007/metrics.csv \
  --summary_csv results/low_summary.csv \
  --run_name low_fft_euclidean_007




  # 1) DINO, cosine, tau=0.07
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/low \
  --output results/low_dino_cos_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder dino_v2 \
  --sim_metric cosine \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/low_dino_cos_007/metrics.csv \
  --summary_csv results/low_summary.csv \
  --run_name low_dino_cos_007

# 2) Clip, cosine, tau=0.07
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/low \
  --output results/rain_clip_cos_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric cosine \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/low_clip_cos_007/metrics.csv \
  --summary_csv results/low_summary.csv \
  --run_name low_clip_cos_007


# 3)fft, cosine, tau=0.07
export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/low \
  --output results/low_fft_cos_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder fft \
  --sim_metric cosine \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/low_fft_cos_007/metrics.csv \
  --summary_csv results/low_summary.csv \
  --run_name low_fft_cos_007



  #########haze_rain
  # 1) DINO, euclidean, tau=0.07
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/haze_rain \
  --output results/haze_rain_dino_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder dino_v2 \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/haze_rain_dino_euclidean_007/metrics.csv \
  --summary_csv results/haze_rain_summary.csv \
  --run_name haze_rain_dino_euclidean_007

# 2) Clip, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/haze_rain \
  --output results/haze_rain_clip_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/haze_rain_clip_euclidean_007/metrics.csv \
  --summary_csv results/haze_rain_summary.csv \
  --run_name haze_rain_clip_euclidean_007


# 3)fft, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/haze_rain \
  --output results/haze_rain_fft_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder fft \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/haze_rain_fft_euclidean_007/metrics.csv \
  --summary_csv results/haze_rain_summary.csv \
  --run_name haze_rain_fft_euclidean_007

####enhanced fft
export CUDA_VISIBLE_DEVICES=7

  python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/rain \
  --output results/rain_fft_enh \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2 \
  --embedder fft_enhanced \
  --fft_resize 256 \
  --fft_out_size 32 \
  --fft_clean_proto weights/fft_clean_proto.npy \
  --sim_metric cosine \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/rain_fft_enh/metrics.csv \
  --summary_csv results/rain_summary.csv \
  --run_name rain_fft_enh_cos_007



#########snow
  # 1) DINO, euclidean, tau=0.07
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/snow \
  --output results/snow_dino_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --embedder dino_v2 \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/snow_dino_euclidean_007/metrics.csv \
  --summary_csv results/snow_summary.csv \
  --run_name snow_dino_euclidean_007

# 2) Clip, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=3
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/snow \
  --output results/snow_clip_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1\
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/snow_clip_euclidean_007/metrics.csv \
  --summary_csv results/snow_summary.csv \
  --run_name snow_clip_euclidean_007


# 3)fft, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/snow \
  --output results/snow_fft_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --embedder fft \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/snow_fft_euclidean_007/metrics.csv \
  --summary_csv results/snow_summary.csv \
  --run_name snow_fft_euclidean_007




#########low_rain
  # 1) DINO, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=5
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/low_rain \
  --output results/low_rain_dino_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --embedder dino_v2 \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/low_rain_dino_euclidean_007/metrics.csv \
  --summary_csv results/low_rain_summary.csv \
  --run_name low_rain_dino_euclidean_007

# 2) Clip, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/low_rain \
  --output results/low_rain_clip_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1\
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/low_rain_clip_euclidean_007/metrics.csv \
  --summary_csv results/low_rain_summary.csv \
  --run_name low_rain_clip_euclidean_007


# 3)fft, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/low_rain \
  --output results/low_rain_fft_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --embedder fft \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/low_rain_fft_euclidean_007/metrics.csv \
  --summary_csv results/low_rain_summary.csv \
  --run_name low_rain_fft_euclidean_007




#########haze_snow
  # 1) DINO, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=5
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/haze_snow \
  --output results/haze_snow_dino_cosine_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --embedder dino_v2 \
  --sim_metric cosine \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/haze_snow_dino_euclidean_007/metrics.csv \
  --summary_csv results/haze_snow_summary.csv \
  --run_name haze_snow_dino_cosine_007

# 2) Clip, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=2
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/haze_snow \
  --output results/haze_snow_clip_cosine_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1\
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric cosine \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --gt_root samples/clear \
  --metrics_csv results/haze_snow_clip_cosine_007/metrics.csv \
  --summary_csv results/haze_snow_summary.csv \
  --run_name haze_snow_clip_cosine_007


# haze,clip, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --pair_list data/haze/train_list_haze.txt \
  --input data/haze/train/lq \
  --output results/haze_clip_euclidean_007 \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --metrics_csv results/haze_clip_euclidean_007/metrics.csv \
  --summary_csv results/haze_snow_summary.csv \
  --run_name hhaze_clip_euclidean_007

  # haze,clip, euclidean, tau=0.07
export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_data \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --pair_list data/txt_lists/val_list_haze_snow.txt \
  --input data/rain/train/lq \
  --output mix_results/haze_snow_clip_euclidean_007 \
  --loradb weights/lora \
  --domains haze_snow,rain \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --device cuda \
  --metrics_csv mix_results/haze_snow_clip_euclidean_007/metrics.csv \
  --summary_csv mix_results/haze_snow_summary.csv \
  --run_name haze_snow_clip_euclidean_007



#############_ramp_full
export PYTHONPATH=.
python -m lora_adapters.infer_data_ramp \
  --input data/haze_snow/test_lq \
  --output test_results/haze_snow_ramp_full \
  --loradb weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --single_tau 0.60 \
  --single_margin 0.20 \
  --mix_topk 3 \
  --ramp_p0 0.45 \
  --ramp_k 6 \
  --ramp_eps 0.0 \
  --gt_root samples/clear \
  --metrics_csv test_results/haze_snow_ramp_full/metrics.csv \
  --summary_csv test_results/ramp_full_summary.csv \
  --run_name haze_snow_ramp_full \
  --save_images --concat \
  --seed 42 --deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_data_ramp \
  --input samples/haze_snow \
  --pair_list data/txt_lists/val_list_haze_snow.txt \
  --output test_results/haze_snow_ramp_full \
  --loradb weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --single_tau 0.70 \
  --single_margin 0.38 \
  --mix_topk 3 \
  --ramp_p0 0.45 \
  --ramp_k 6 \
  --ramp_eps 0.0 \
  --metrics_csv test_results/haze_snow_ramp_full/metrics.csv \
  --summary_csv test_results/ramp_full_summary.csv \
  --run_name haze_snow_ramp_full \
  --save_images --concat \
  --seed 42 --deterministic

 [SUMMARY]
  BASE: PSNR=17.0261, SSIM=0.8117
  MIX : PSNR=18.3617, SSIM=0.8558

  export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_data_ramp \
  --input samples/haze_rain \
  --pair_list data/txt_lists/val_list_haze_rain.txt \
  --output test_results/haze_rain_ramp_full \
  --loradb weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --single_tau 0.70 \
  --single_margin 0.35 \
  --mix_topk 3 \
  --ramp_p0 0.45 \
  --ramp_k 8 \
  --ramp_eps 0.0 \
  --metrics_csv test_results/haze_rain_ramp_full/metrics.csv \
  --summary_csv test_results/ramp_full_summary.csv \
  --run_name haze_rain_ramp_full \
  --save_images --concat \
  --seed 42 --deterministic
[SUMMARY]
  BASE: PSNR=20.3443, SSIM=0.9078
  MIX : PSNR=19.1056, SSIM=0.8754
[SUMMARY]
  BASE: PSNR=20.3443, SSIM=0.9078
  MIX : PSNR=18.9130, SSIM=0.8719


export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_data_ramp \
  --input samples/haze_snow \
  --pair_list data/txt_lists/val_list_low_rain.txt \
  --output test_results/haze_snow_ramp_full \
  --loradb weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --single_tau 0.70 \
  --single_margin 0.35 \
  --mix_topk 3 \
  --ramp_p0 0.45 \
  --ramp_k 6 \
  --ramp_eps 0.0 \
  --metrics_csv test_results/low_rain_ramp_full/metrics.csv \
  --summary_csv test_results/ramp_full_summary.csv \
  --run_name low_rain_ramp_full \
  --save_images --concat \
  --seed 42 --deterministic
[SUMMARY]
  BASE: PSNR=18.9668, SSIM=0.6790
  MIX : PSNR=31.5864, SSIM=0.9538


  export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_data_ramp \
  --input samples/low_snow \
  --pair_list data/txt_lists/val_list_low_snow.txt \
  --output test_results/low_snow_ramp_full \
  --loradb weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --single_tau 0.70 \
  --single_margin 0.35 \
  --mix_topk 3 \
  --ramp_p0 0.40 \
  --ramp_k 12 \
  --ramp_eps 0.0 \
  --metrics_csv test_results/low_snow_ramp_full/metrics.csv \
  --summary_csv test_results/ramp_full_summary.csv \
  --run_name low_snow_ramp_full \
  --save_images --concat \
  --seed 42 --deterministic
  [SUMMARY]
  BASE: PSNR=12.1928, SSIM=0.4723
  MIX : PSNR=16.7826, SSIM=0.5694
[SUMMARY]
  BASE: PSNR=12.1928, SSIM=0.4723
  MIX : PSNR=16.8414, SSIM=0.5762


  export CUDA_VISIBLE_DEVICES=1
python -m lora_adapters.infer_data_ramp_new \
  --input samples/snow \
  --pair_list data/txt_lists/val_list_snow.txt \
  --output test_results/snow_ramp_full \
  --loradb weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --topk 3 \
  --rank 16 \
  --alpha 16 \
  --single_tau 0.70 \
  --single_margin 0.35 \
  --mix_topk 3 \
  --ramp_p0 0.45 \
  --ramp_k 6 \
  --ramp_eps 0.0 \
  --metrics_csv test_results/snow_ramp_full/metrics.csv \
  --summary_csv test_results/ramp_full_summary.csv \
  --run_name snow_ramp_full \
  --save_images --concat \
  --seed 42 --deterministic
[SUMMARY]
  BASE: PSNR=33.9220, SSIM=0.9451
  MIX : PSNR=32.7488, SSIM=0.9333

##########haze_snow_cascade_fixed
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_cascade_data_v2 \
  --input samples/snow \
  --pair_list test_data/Rain100L/val_list_Rain100L.txt \
  --output_dir test_results/cascade_v2/Rain100L_fixed \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora \
  --local_domains rain,rain2,rainy,snow,snow1,rain3 \
  --global_domains haze,haze1,haze2,low,low1 \
  --sim_metric euclidean \
  --topk1 3  --topk2 3 \
  --temperature1 0.07 --temperature2 0.07 \
  --rank 16 \
  --alpha 16 \
  --single_tau 0.7 \
  --single_margin 0.35 \
  --mix_topk 3 \
  --order local2global \
  --stage2_mode fixed \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --run_name Rain100L_cascade_fixed \
  --save_base --save_stage_images --save_concat




  export PYTHONPATH=.
  export CUDA_VISIBLE_DEVICES=2
  python -m lora_adapters.infer_data_kselect \
    --loradb weights/lora \
    --base_ckpt pretrained_models/histoformer_base.pth \
    --yaml lora_adapters/configs/histoformer_mol.yaml \
    --input samples/haze_snow \
    --pair_list data/txt_lists/val_list_haze_rain.txt \
    --output_dir test_results/haze_rain_kselect \
    --loradb weights/lora \
    --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
    --local_domains rain,rain2,rain3,rainy,snow,snow1 \
    --global_domains haze,haze1,haze2,low,low1 \
    --embedder clip --clip_model ViT-B-16 --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
    --sim_metric euclidean --temperature 0.07 \
    --topk 3  --single_tau 0.75 --single_margin 0.40 \
    --k_topk 80 --k_alpha 0.5 --k_beta 1.0 --use_gamma 1 --k_ramp_mode sigmoid \
    --rank 16 --alpha 16 \
    --gt_root samples/clear \
    --metrics_csv test_results/haze_rain_kselect/metrics.csv \
    --routing_jsonl test_results/haze_rain_kselect/routing.jsonl \
    --run_name haze_rain_kselect \
    --save_images --concat \
    --seed 42 --deterministic --verbose



python scripts/analyze_kselect_grid.py \
  --root test_results/haze_rain_kselect_grid \
  --pattern "*_metrics.csv" \
  --sort_by dpsnr_mean \
  --topk 20


./run_kselect_grid.sh


python lora_adapters/infer_data_kselect.py \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/haze_snow \
  --pair_list data/txt_lists/val_list_haze_snow.txt \
  --output_dir test_results/fixedpair_snow1_haze2_kselect \
  --loradb weights/lora \
  --domains snow1,haze2 \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean \
  --temperature 0.07 \
  --rank 16 --alpha 16 \
  --topk 3 \
  --mix_topk 11 \
  --single_tau 2.0 \
  --single_margin 999 \
  --local_domains snow1 \
  --global_domains haze2 \
  --k_topk 80 \
  --k_alpha 1.0 --k_beta 1.0 \
  --use_gamma 1 \
  --k_ramp_mode sigmoid \
  --gt_root samples/clear \
  --metrics_csv test_results/fixedpair_snow1_haze2_kselect/metrics.csv \
  --summary_csv test_results/fixedpair_snow1_haze2_kselect/summary.csv \
  --run_name fixed_snow1_haze2_kselect \
  --save_images --concat \
  --seed 42



#########kselect_v2_static_triplet
export PYTHONPATH=.
python -m lora_adapters.infer_data_kselect_v2 \
  --input samples/haze_rain \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --pair_list data/txt_lists/val_list_haze_rain.txt \
  --output_dir test_results/haze_rain_kselect_v2_static \
  --gt_root samples/clear \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --local_domains rain,rain2,rain3,rainy,snow,snow1 \
  --global_domains haze,haze1,haze2,low,low1 \
  --embedder clip --clip_model ViT-B-16  \
  --sim_metric euclidean --temperature 0.07 \
  --topk 3 --mix_topk 5 --single_tau 0.78 --single_margin 0.40 \
  --rank 16 --alpha 16 \
  --kselect_mode static \
  --k_score_mode topk_ratio \
  --gamma_mode per_layer --gamma_clip 3 \
  --k_topk 40 \
  --k_alpha 1.0 --k_beta 1.0 --k_ramp_mode sigmoid \
  --save_images --concat \
  --metrics_csv test_results/haze_rain_kselect_v2_static/metrics.csv \
  --routing_jsonl test_results/haze_rain_kselect_v2_static/routing.jsonl \
  --run_name haze_rain_kselect_v2_static \
  --seed 42 --deterministic --verbose



export PYTHONPATH=.
python -m lora_adapters.infer_data_kselect_v2 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --pair_list data/txt_lists/val_list_haze_rain.txt \
  --output_dir test_results/haze_rain_kselect_v2_act \
  --gt_root samples/clear \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora \
  --domains rain3,haze2 \
  --local_domains rain3 \
  --global_domains haze2 \
  --embedder clip --clip_model ViT-B-16  \
  --sim_metric euclidean --temperature 0.07 \
  --topk 3 --mix_topk 3 --single_tau 1.78 --single_margin 1.40 \
  --rank 16 --alpha 16 \
  --kselect_mode activation \
  --act_score_mode mean_abs \
  --gamma_mode per_layer --gamma_clip 3 \
  --k_topk 8 --k_alpha 0.8 --k_beta 1.5 --k_ramp_mode sigmoid \
  --save_images --concat \
  --metrics_csv test_results/haze_rain_kselect_v2_act/metrics.csv \
  --routing_jsonl test_results/haze_rain_kselect_v2_act/routing.jsonl \
  --run_name haze_rain_kselect_v2_act \
  --seed 42 --deterministic --verbose


python -m lora_adapters.infer_data_kselect \
  --pair_list data/txt_lists/val_list_haze_rain.txt \
  --output_dir test_results/haze_rain_act_dy_ch \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --local_domains rain,rain2,rain3,rainy,snow,snow1 \
  --global_domains haze,haze1,haze2,low,low1 \
  --embedder clip --clip_model ViT-B-16 --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean --temperature 0.07 \
  --rank 16 --alpha 16 \
  --single_tau 0.60 --single_margin 0.20 --mix_topk 3 \
  --mix_mode act_kselect_dy_ch \
  --act_layer_topk 30 \
  --act_layer_tau 0.05 \
  --act_ch_topk 32 \
  --act_enable_both 1 \
  --act_both_tau 0.05 \
  --act_both_ratio 0.6 \
  --act_print_top 30 \
  --gt_root samples/clear \
  --metrics_csv test_results/haze_rain_act_dy_ch/metrics.csv \
  --summary_csv test_results/haze_rain_act_dy_ch/summary.csv \
  --run_name haze_rain_act_dy_ch \
  --save_images --concat \
  --verbose