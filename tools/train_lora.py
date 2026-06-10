#!/usr/bin/env python3
"""
QLoRA fine-tune of the chat base model on the server transcript dataset.

Loads the base model NF4-quantized (fits an RTX 3080 12 GB: ~2.5 GB weights for
a 3B model plus adapter gradients/optimizer and activations), attaches LoRA
adapters to all linear layers, and runs completion-style SFT on the transcript
chunks from tools/build_dataset.py. Only the adapter (a few tens of MB) is
saved; the bot loads it on top of the same base model at inference time.

Usage:
  python tools/train_lora.py                                # defaults
  python tools/train_lora.py --base meta-llama/Llama-3.2-3B --epochs 2
  python tools/train_lora.py --batch-size 2 --grad-accum 8  # if you have headroom
  python tools/train_lora.py --batch-size 2 --grad-accum 8 --resume
                             # continue an interrupted run (same flags as before!)

Memory notes for 12 GB cards: Llama-3's 128k vocab makes the logits tensor the
peak allocation (~250 MB per sample at 1024 tokens, more in backward), so
per-device batch size is the main VRAM lever — gradient accumulation keeps the
effective batch size without the memory cost. Defaults here (batch 1 x accum 16)
fit alongside a desktop session.

Gated models (Llama, Gemma) need a one-time `hf auth login` after accepting
the license on huggingface.co. Free VRAM first — stop the image bot.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Reduces VRAM fragmentation; must be set before torch is imported (same as
# flux/generate.py). Without it, "reserved but unallocated" memory piles up.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--base", default="meta-llama/Llama-3.2-3B",
                        help="Base model id (should match FRIENDBOT_LLM_BASE)")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "sft")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "models" / "adapters" / "friendbot-lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Training sequence length in tokens")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--resume", nargs="?", const=True, default=None, metavar="CHECKPOINT",
        help="Resume an interrupted run: bare --resume picks the latest "
             "checkpoint-* in --out, or pass an explicit checkpoint directory. "
             "Use the same hyperparameters as the original run, or the "
             "skipped-step bookkeeping will be wrong.",
    )
    args = parser.parse_args()

    if args.resume is True and not list(args.out.glob("checkpoint-*")):
        raise SystemExit(f"--resume given but no checkpoint-* found in {args.out}")
    if isinstance(args.resume, str) and not Path(args.resume).is_dir():
        raise SystemExit(f"--resume checkpoint not found: {args.resume}")

    # Heavy imports after argparse so --help stays instant.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for QLoRA training.")
    if not (args.data / "train.jsonl").exists():
        raise SystemExit(f"No dataset at {args.data} — run tools/build_dataset.py first.")

    print(f"Base model: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.data / "train.jsonl"),
            "val": str(args.data / "val.jsonl"),
        },
    )
    print(f"Dataset: {len(dataset['train'])} train / {len(dataset['val'])} val samples")

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        # Eval would otherwise default to batch 8 and OOM at the epoch boundary.
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        max_length=args.max_length,
        # Samples from build_dataset.py are already ~max_length tokens, so
        # packing gains almost nothing — and without flash-attn it risks
        # cross-attention between packed samples (TRL warns about this).
        packing=False,
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        peft_config=lora,
    )

    if args.resume:
        where = args.resume if isinstance(args.resume, str) else "latest checkpoint"
        print(f"Resuming from {where} (optimizer/scheduler/RNG state restored)")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"\nAdapter saved to {args.out}")
    print("Run the bot with:  python -m friendbot2 chat")


if __name__ == "__main__":
    main()
