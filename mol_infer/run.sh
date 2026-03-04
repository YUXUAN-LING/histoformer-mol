python -m mol_infer.scripts.infer \
  --input samples/haze_snow \
  --output_dir test_results/mol_stage1_smoke \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --embedder clip \
  --embedder_tag clip_vit-b-16\
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --loradb_root weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --local_domains rain,rain2,rain3,rainy,snow,snow1 \
  --global_domains haze,haze1,haze2,low,low1 \
  --embedder clip --clip_model ViT-B-16 \
  --sim_metric cosine \
  --topk 5 --mix_topk 5 \
  --single_tau 0.72 --single_margin 0.10 \
  --device cuda





python -m mol_infer.scripts.infer \
  --input samples/haze_snow \
  --output_dir test_results/mol_kselect_none \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb_root weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --local_domains rain,rain2,rain3,rainy,snow,snow1 \
  --global_domains haze,haze1,haze2,low,low1 \
  --embedder clip --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric cosine \
  --topk 5 --mix_topk 5 \
  --single_tau 0.72 --single_margin 0.10 \
  --enable_lora \
  --enable_fusion \
  --fusion kselect_none \
  --kselect_mode static \
  --k_score_mode topk_ratio \
  --k_alpha 1.0 --k_beta 1.0 \
  --k_ramp_mode sigmoid --use_gamma --gamma_mode per_layer \
  --none_mode const --none_beta 0.15 --none_alpha 0.0 --none_metric abs \
  --save_images --concat \
  --device cuda

python -m mol_infer.scripts.infer_clean \
  --input samples/haze_snow \
  --output_dir test_results/mol_clean_smoke \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb_root weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --local_domains rain,rain2,rain3,rainy,snow,snow1 \
  --global_domains haze,haze1,haze2,low,low1 \
  --embedder clip --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --sim_metric cosine --topk 5 --mix_topk 5 \
  --single_tau 0.72 --single_margin 0.10 \
  --device cuda


python -m cli.infer_mol \
  --input_dir data/haze_snow/val/lq \
  --pair_list data/txt_lists/val_list_low_rain.txt \
  --gt_root data/low_rain/val/gt \
  --output_dir results/kselect_dy_demo_low_rain \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --loradb_root weights/lora \
  --domains rain,rain2,rain3,rainy,snow,snow1,haze,haze1,haze2,low,low1 \
  --local_domains rain,rain2,rain3,rainy,snow,snow1 \
  --global_domains haze,haze1,haze2,low,low1 \
  --embedder clip \
  --clip_model ViT-B-16 \
  --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --temperature 0.07 --topk 3 --mix_topk 10 \
  --single_tau 1 --single_margin 1 \
  --k_ramp_mode sigmoid --k_alpha 1.0 --k_beta 1.0 \
  --act_every_n 4 --act_score_mode mean_abs \
  --device cuda --embed_device cpu \
  --save_triplet 1 --save_out 1 --routes_jsonl 1 --log_every 10 \
  --mix_mode act_kselect_dy \
  --dy_topk_layers 30 \
  --dy_tau 0.05 \
  --dy_enable_both 1 \
  --dy_both_tau 0.5 \
  --dy_both_ratio 0.6 \
  --dy_score_mode rms \
  --dy_verbose 1 \
  --dy_debug_topn 30 \
  --routes_jsonl 1