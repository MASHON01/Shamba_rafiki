# Swahili fine-tune (stretch, optional)

This directory is the **optional upside** on top of prompt-side Swahili
(`backend/app/language/swahili/`). Per the build plan, the fine-tune is
**the first thing to cut if behind**. Prompt-side Swahili is the safe
core and ships regardless; a fine-tune only ships if it clearly earns its
place.

## The decision gate

Ship the fine-tuned model **only if all** of these hold:

1. **It beats prompt-only.** Run `python -m scripts.eval_swahili --compare`
   with the base model first. Only fine-tune if enrichment-ON Swahili is
   still too weak. After fine-tuning, the fine-tuned model must score
   higher on the same eval set than prompt-only did.
2. **It stays within the RAM budget.** After merge + re-quantize to
   Q4_K_M GGUF, `scripts/profile_memory.py` on the 8 GB target must still
   sit comfortably under the 7 GB ceiling (Output 10). A fine-tune that
   erodes the headroom that wins the Budget Profile bonus is not worth it.
3. **It doesn't regress English.** English is the primary evaluation
   language; the fine-tune must not lower English accuracy (Output 7).

If any fail, keep the base model + prompt-side Swahili and drop the
fine-tune. That is a good outcome, not a failure.

## Pipeline

```
python prepare_dataset.py            # -> training/data/swahili_agri.jsonl
python finetune_swahili.py \         # -> training/adapters/<run>/  (LoRA)
    --base <hf-model-or-path> \
    --dataset training/data/swahili_agri.jsonl
bash merge_and_quantize.sh <base> training/adapters/<run> Q4_K_M
                                     # -> training/merged/*.gguf, then re-benchmark
```

Runs on a capable machine (a GPU makes QLoRA practical); it is **not**
part of the app, CI, or the kiosk runtime. All produced weights/datasets
are gitignored - only the scripts are committed.

## Requirements (install only when you actually run this)

`pip install transformers peft trl datasets bitsandbytes accelerate`
plus a built `llama.cpp` for the convert + quantize step.
