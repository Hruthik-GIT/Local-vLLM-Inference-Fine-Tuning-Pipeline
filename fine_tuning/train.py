"""
Unsloth + LoRA fine-tuning for JSON tool-calling / SQL generation.

Usage:
    python train.py --model unsloth/Meta-Llama-3.1-8B-Instruct --epochs 3
    python train.py --model unsloth/Qwen2.5-7B-Instruct --epochs 3

Produces:
    ./outputs/lora_adapters/       - LoRA adapter weights (small, portable)
    ./outputs/merged_16bit/        - merged fp16 model ready for vLLM serving
    ./outputs/vram_report.json     - peak VRAM usage during training
"""
import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an 8B model with Unsloth + LoRA")
    parser.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct",
                         help="Base model to fine-tune (e.g. unsloth/Qwen2.5-7B-Instruct)")
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "outputs"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--skip-merge", action="store_true",
                         help="Skip exporting the merged 16-bit model (adapters only)")
    return parser.parse_args()


def format_example(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}


def track_vram_start() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def track_vram_report(output_dir: str, base_model: str, elapsed_seconds: float) -> None:
    report = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        report.update({
            "device_name": torch.cuda.get_device_name(0),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
        })
    report["base_model"] = base_model
    report["training_seconds"] = round(elapsed_seconds, 1)
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "vram_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[train.py] VRAM report written to {report_path}: {report}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[train.py] Loading base model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect (bf16 if supported, else fp16)
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print(f"[train.py] Loading dataset: {args.dataset}")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    dataset = dataset.map(lambda ex: format_example(ex, tokenizer))

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=training_args,
    )

    track_vram_start()
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    track_vram_report(args.output_dir, args.model, elapsed)

    adapter_dir = os.path.join(args.output_dir, "lora_adapters")
    print(f"[train.py] Saving LoRA adapters to {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    if not args.skip_merge:
        merged_dir = os.path.join(args.output_dir, "merged_16bit")
        print(f"[train.py] Merging LoRA into base weights (fp16) -> {merged_dir}")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"[train.py] Merged model ready for vLLM serving at {merged_dir}")
        print("[train.py] To also produce a GGUF artifact, run:")
        print(f'    model.save_pretrained_gguf("{os.path.join(args.output_dir, "gguf")}", tokenizer, quantization_method="q8_0")')

    print("[train.py] Done.")


if __name__ == "__main__":
    main()
