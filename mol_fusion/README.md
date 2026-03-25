# mol_fusion (new mainline)

This package is the clean mainline for:

`retrieval -> routing -> pairwise fusion policy -> executor -> evaluation/probe`

## Migration notes

- **Use** `mol_fusion.pipelines.infer_retrieval` instead of old retrieval snippets in `lora_adapters/infer_data_kselect.py`.
- **Use** `mol_fusion.pipelines.infer_pairwise` for fixed `(dom1, dom2)` pairwise study.
- **Use** `mol_fusion.pipelines.infer_full` for full retrieval+routing+fusion.
- `probe_weights.py` reuses **the same fusion policy implementation** as inference.

## Top-k semantics (strict)

1. `topk < 0` => full-layer softmix.
2. `topk = 0` => 0 key layers, all layers use non-key baseline.
3. `topk > 0` => only top-k layers use key-layer softmix.
4. `topk = 0` and `nonkey_mode=ramp` => exactly pure-ramp.
