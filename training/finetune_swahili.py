#!/usr/bin/env python3
"""
LoRA / QLoRA fine-tune for Swahili agricultural advice (stretch).

    python finetune_swahili.py \
        --base bartowski-or-hf/Llama-3.2-3B-Instruct \
        --dataset training/data/swahili_agri.jsonl \
        --out training/adapters/run1

Trains a small LoRA adapter (4-bit QLoRA by default) on the Swahili
instruction set from prepare_dataset.py. It deliberately does NOT ship
anything: it produces an adapter in training/adapters/ (gitignored) that
merge_and_quantize.sh then folds into the base model and converts to
Q4_K_M GGUF for evaluation against the decision gate (see README.md).

Heavy ML dependencies (transformers, peft, trl, datasets, bitsandbytes)
are imported lazily and only needed when this actually runs - the app,
CI, and the kiosk never import this file. A GPU makes QLoRA practical;
on CPU this is impractically slow.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune for Swahili.")
    parser.add_argument("--base", required=True, help="Base model (HF id or path).")
    parser.add_argument("--dataset", required=True, help="JSONL from prepare_dataset.py.")
    parser.add_argument("--out", default="training/adapters/run1", help="Adapter output dir.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-4bit", action="store_true", help="LoRA in 16-bit instead of QLoRA.")
    args = parser.parse_args()

    # Lazy, guarded imports so this file is inert unless actually run.
    try:
        import torch  # noqa: F401
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer

        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - only when run for real.
        raise SystemExit(
            "Missing training deps. Install with: pip install transformers "
            "peft trl datasets bitsandbytes accelerate"
        ) from exc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    quant_config = None
    if not args.no_4bit:
        import torch

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant_config, device_map="auto"
    )

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    trainer = SFTTrainer(
        model=model,
        peft_config=lora,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(out),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=8,
            logging_steps=5,
            save_strategy="epoch",
            bf16=True,
        ),
    )
    trainer.train()
    trainer.save_model(str(out))
    print(f"Saved LoRA adapter -> {out}")
    print(
        "Next: bash merge_and_quantize.sh <base> "
        f"{out} Q4_K_M # then re-run the eval + RAM gate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main)
