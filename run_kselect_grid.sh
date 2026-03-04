#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
mkdir -p test_results/low_rain_kselect_grid

BASE_CMD=(
  python -m lora_adapters.infer_data_kselect
  --base_ckpt pretrained_models/histoformer_base.pth 
--yaml lora_adapters/configs/histoformer_mol.yaml 
  --input samples/low_rain
  --pair_list data/txt_lists/val_list_low_rain.txt
  --output test_results/low_rain_kselect_grid
  --loradb weights/lora
  --embedder clip
  --clip_model ViT-B-16
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin
  --sim_metric euclidean
  --temperature 0.07
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --local_domains rain,rain2,rain3,rainy,snow,snow1
  --global_domains haze,haze1,haze2,low,low1
  --rank 16 --alpha 16
  --single_tau 0.80
  --single_margin 0.40
  --save_images --concat
)

for ktopk in 48; do
  for mode in sigmoid; do
    for alpha in 0.5 1.0 1.5; do
      for beta in 0.5 1.0; do
        run="ktopk${ktopk}_m${mode}_g1_a${alpha}_b${beta}"

        "${BASE_CMD[@]}" \
          --k_topk $ktopk \
          --k_ramp_mode $mode \
          --use_gamma 1 \
          --k_alpha $alpha \
          --k_beta $beta \
          --metrics_csv test_results/low_rain_kselect_grid/${run}_metrics.csv \
          --summary_csv test_results/low_rain_kselect_grid/summary_all.csv \
          --run_name $run
      done
    done
  done
done
