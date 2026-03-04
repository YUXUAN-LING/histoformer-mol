CUDA_VISIBLE_DEVICES=6 \
python -m lora_adapters.infer_folder_lora_compare \
  --input_dir data/blur/train/lq \
  --output_dir results/blur_lora_compare \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --lora_ckpt weights/lora/blur/lora.pth \
  --domain blur \
  --domains rain,snow,fog,blur \
  --rank 32 \
  --alpha 16 \
  --device cuda

CUDA_VISIBLE_DEVICES=6 \
python -m lora_adapters.infer_folder_lora_compare \
  --input_dir data/rain2/test_a/test/data \
  --output_dir results/rain2_lora_compare \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --lora_ckpt weights/lora/rain2/lora.pth \
  --domain blur \
  --domains rain,snow,fog,blur \
  --rank 8 \
  --alpha 8 \
  --device cuda

  CUDA_VISIBLE_DEVICES=6 \
python -m lora_adapters.infer_folder_lora_compare \
  --input_dir data/hazy/Test/hazy/C005 \
  --output_dir results/hazy_lora_compare \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --lora_ckpt weights/lora/hazy/lora.pth \
  --domain hazy \
  --domains hazy \
  --rank 16 \
  --alpha 16 \
  --device cuda

  python -m lora_adapters.infer_folder_lora_compare \
  --input_dir data/blur/train/lq \
  --gt_dir data/blur/train/gt \
  --output_dir results/blur_lora_compare \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --lora_ckpt weights/lora/blur/lora.pth \
  --domain blur \
  --domains rain,snow,fog,blur \
  --rank 32 \
  --alpha 16 \
  --tile 640 \
  --overlap 32 \
  --device cuda

  python -m lora_adapters.infer_folder_lora_compare \
  --input_dir data/rain/val/lq \
  --gt_dir data/rain/val/gt \
  --output_dir results/rain_compare \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --lora_ckpt weights/lora/rain3/lora.pth \
  --domain rain3 \
  --domains rain3,snow,fog,blur \
  --rank 16 \
  --alpha 16 \
  --device cuda

  python -m lora_adapters.infer_folder_lora_compare \
  --pair_list data/hazy/val_list_hazy.txt \
  --output_dir results/hazy_lora_compare \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --lora_ckpt weights/lora/hazy/lora.pth \
  --domain hazy \
  --domains hazy \
  --rank 16 --alpha 16 \
  --tile 1600 --overlap 32 \
  --device cuda

python -m lora_adapters.infer_folder_lora \
  --pair_list data/rainy/val_list_rainy.txt \
  --output_dir results/rainy_lora_compare \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --lora_ckpt weights/lora/rainy/lora.pth \
  --domain rainy \
  --domains rainy,snow,haze1,low \
  --rank 16 \
  --alpha 16 \
  --device cuda


    python -m lora_adapters.infer_folder_lora \
  --input_dir samples/haze \
  --gt_dir samples/clear \
  --output_dir mix_results/haze \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --lora_ckpt weights/lora/haze2/lora.pth \
  --domain haze2 \
  --domains haze2 \
  --rank 16 \
  --alpha 16 \
  --device cuda


####比较结果
python analyze_mol_results.py \
  --summary_csv results/rain_summary.csv \
  --metrics_glob "results/rain_*/metrics.csv" \
  --out_dir results/rain_analysis

###级联运行
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.cascade \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --input samples/haze_rain\
  --output mix_results/cascade_rain3_haze2 \
  --loradb_root weights/lora \
  --gt_root samples/clear \
  --metrics_csv mix_results/cascade_rain3_haze2/metrics.csv \
  --summary_csv mix_results/cascade_summary.csv \
  --run_name rain3_then_haze2 \
  --domains rain3,haze2 \
  --reverse

####同时激活权重
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=.
python -m lora_adapters.infer_dual_lora_compare \
  --pair_list data/txt_lists/val_list_haze_snow.txt \
  --output_dir mix_results/haze_snow_dual_oracle \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --rank 16 --alpha 16 \
  --domain_a haze2\
  --domain_b snow1 \
  --wa 0.5 --wb 0.5 \
  --device cuda \
  --deterministic\
  --save_images


export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_dual_lora_compare \
  --pair_list data/txt_lists/val_list_haze_rain.txt \
  --output_dir mix_results/haze_rain_dual_oracle \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora_ortho \
  --domains rain3,haze2\
  --rank 16 --alpha 16 \
  --domain_a haze2 \
  --domain_b rain3 \
  --wa 1.0 --wb 1.0 \
  --device cuda \
  --deterministic\
  --save_images
####
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_dual_lora_compare \
  --pair_list data/txt_lists/val_list_haze_snow.txt \
  --output_dir mix_results/haze_snow_lw_only_snow \
  --save_images \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora_lw \
  --domains snow1_local,haze2_global \
  --rank 16 --alpha 16 \
  --domain_a haze2_global \
  --domain_b snow1_local \
  --wa 0.0 --wb 1.0 \
  --dual_variant both \
  --global_domain haze2_global \
  --local_domain snow1_local \
  --default_policy none \
  --seed 42 --deterministic

## 
export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_dual_lora_compare \
  --pair_list data/txt_lists/val_list_haze_snow.txt \
  --output_dir mix_results/haze_snow_oldloras_lw_vs_all \
  --save_images \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora \
  --domains haze2,snow1 \
  --rank 16 --alpha 16 \
  --domain_a haze2 \
  --domain_b snow1 \
  --wa 1.0 --wb 1.0 \
  --dual_variant both \
  --global_domain haze2 \
  --local_domain snow1 \
  --default_policy none \
  --seed 42 --deterministic

#### layerwise ramp 比较
export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_dual_lora_compare \
  --pair_list data/txt_lists/val_list_low_snow.txt \
  --output_dir mix_results/low_snow_all_lw_ramp \
  --save_images \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora \
  --domains low1,snow1\
  --rank 16 --alpha 16 \
  --domain_a low1 --domain_b snow1 \
  --wa 1.0 --wb 1.0 \
  --dual_variant all_layerwise_ramp \
  --ramp_p0 0.4 --ramp_k 12 --ramp_eps 0 \
  --seed 42 --deterministic

export CUDA_VISIBLE_DEVICES=7
python -m lora_adapters.infer_dual_lora_compare \
  --pair_list data/txt_lists/val_list_haze_snow.txt \
  --output_dir mix_results/haze_snow_all_lw_ramp \
  --save_images \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora \
  --domains haze2,snow1 \
  --rank 16 --alpha 16 \
  --domain_a haze2 --domain_b snow1 \
  --wa 1.0 --wb 1.0 \
  --dual_variant all_layerwise_ramp \
  --ramp_p0 0.65 --ramp_k 6 --ramp_eps 0 \
  --seed 42 --deterministic

export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_dual_lora_compare \
  --pair_list data/txt_lists/val_list_haze_rain.txt \
  --output_dir mix_results/haze_rain_all_lw_ramp \
  --save_images \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora \
  --domains haze2,rain3 \
  --rank 16 --alpha 16 \
  --domain_a haze2 --domain_b rain3 \
  --wa 1.0 --wb 1.0 \
  --dual_variant all_layerwise_ramp \
  --ramp_p0 0.45 --ramp_k 6 --ramp_eps 0 \
  --seed 42 --deterministic


  ####大批量比较
  export CUDA_VISIBLE_DEVICES=3
  python -m lora_adapters.infer_dual_lora_amp \
  --pair_list data/txt_lists/val_list_low_snow.txt \
  --output_dir results/low_snow_ramp_grid \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --rank 16 --alpha 16 \
  --domain_a low1 --domain_b snow1 \
  --wa 1.0 --wb 1.0 \
  --grid_ramp \
  --grid_ramp_ks "4,6,8,12" \
  --grid_ramp_p0s "0.40,0.45,0.50,0.55,0.65" \
  --grid_ramp_eps 0 \
  --seed 42 --deterministic


###检索评估
export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.run_retrieval_eval_all \
  --eval_sets eval_sets.json \
  --out_dir results/retrieval_eval/clip_eu_007 \
  --cache_emb_dir results/emb_cache \
  --loradb_root weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric euclidean --euclidean_mode inv \
  --temperature 0.07 --topk 3 \
  --normalize \
  --proto_reduce sum \
  --seed 42

python -m lora_adapters.run_retrieval_eval_all \
  --eval_sets eval_sets.json \
  --out_dir results/retrieval_eval/dino_eu_007 \
  --cache_emb_dir results/emb_cache \
  --loradb_root weights/lora \
  --domains rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1 \
  --embedder dino_v2 \
  --dino_ckpt weights/dinov2/vit_base_patch14_dinov2.lvd142m-384.pth\
  --sim_metric euclidean --euclidean_mode inv \
  --temperature 0.07 --topk 3 \
  --normalize \
  --proto_reduce sum \
  --seed 42
###画图分析
python -m lora_adapters.plot_retrieval_results \
  --in_dir results/retrieval_eval/clip_eu_007/haze_snow \
  --out_dir results/retrieval_eval/clip_eu_007/haze_snow/plots
