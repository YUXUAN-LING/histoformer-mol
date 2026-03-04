export CUDA_VISIBLE_DEVICES=6
python -m lora_adapters.infer_cascade_data \
  --input samples/haze_snow \
  --gt_root samples/clear \
  --output_dir mix_results/cascade_mix_both \
  --base_ckpt pretrained_models/histoformer_base.pth \
  --yaml lora_adapters/configs/histoformer_mol.yaml \
  --rank 16 --alpha 16 \
  --loradb weights/lora \
  --local_domains rain,rain2,rainy,snow,snow1 \
  --global_domains haze,haze1,haze2,low,low1 \
  --order local2global \
  --stage2_source both \
  --embedder clip \
  --clip_model ViT-B-16 --clip_pretrained weights/clip/open_clip_pytorch_model.bin \
  --topk1 1 --topk2 1 \
  --temperature1 0.05 --temperature2 0.05 \
  --stage2_min_margin 0.02 \
  --save_base --save_stage_images --save_concat \
  --run_name exp_cascade_both \
  --embedder_tag clip_vit-b-16

rain2,low,rain,snow,haze,haze1,rainy,rain3,low1,haze2,snow1