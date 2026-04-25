# ═══════════════════════════════════════════════════════════════════
# SWARM-OS TRAINING PIPELINE — Kaggle T4 / Colab A100 Compatible
# Auto-detects GPU and adjusts settings accordingly
# ═══════════════════════════════════════════════════════════════════

import os
import re
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timedelta

import torch

# ── Logging Setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_log.txt", mode="a", encoding="utf-8"),
    ]
)
log = logging.getLogger("swarm-train")

# ── GPU Auto-Detection ────────────────────────────────────────────
def detect_gpu():
    if not torch.cuda.is_available():
        return "cpu", 0
    name = torch.cuda.get_device_name(0).upper()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "A100" in name or vram_gb > 60:
        return "a100", vram_gb
    elif "A10" in name or (vram_gb > 20 and vram_gb <= 60):
        return "a10g", vram_gb
    else:
        return "t4", vram_gb  # T4, V100, RTX etc

GPU_TYPE, GPU_VRAM = detect_gpu()
log.info(f"Detected GPU: {GPU_TYPE.upper()} ({GPU_VRAM:.1f}GB VRAM)")

# ── Constants ─────────────────────────────────────────────────────
MODEL_ID    = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
MAX_SEQ_LEN = 2048
LORA_R      = 32 if GPU_TYPE == "a100" else 16   # Lower rank for T4
LORA_ALPHA  = 64 if GPU_TYPE == "a100" else 32
LORA_DROPOUT = 0.0   # Fix 1: Must be 0 for Unsloth fast patching

OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm-os-trained")
SPLITS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "splits")
SFT_TRAIN_PATH = os.path.join(SPLITS_DIR, "sft_train.jsonl")
SFT_EVAL_PATH  = os.path.join(SPLITS_DIR, "sft_eval.jsonl")
GRPO_PATH      = os.path.join(SPLITS_DIR, "grpo_prompts.jsonl")
DPO_PATH       = os.path.join(SPLITS_DIR, "dpo_pairs.jsonl")

SFT_CKPT    = os.path.join(OUTPUT_DIR, "stage1-sft")
GRPO_CKPT   = os.path.join(OUTPUT_DIR, "stage2-grpo")
DPO_CKPT    = os.path.join(OUTPUT_DIR, "stage3-dpo")
FINAL_DIR   = os.path.join(OUTPUT_DIR, "final-merged")


def get_vram_info():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{alloc:.1f}GB used / {total:.1f}GB total"
    return "CUDA not available"


def load_jsonl(path: str) -> list:
    import jsonlines
    if not os.path.exists(path):
        log.error(f"File not found: {path}")
        sys.exit(1)
    with jsonlines.open(path) as reader:
        data = list(reader)
    log.info(f"Loaded {len(data)} records from {os.path.basename(path)}")
    return data


def print_banner(title: str, details: dict):
    log.info("=" * 64)
    log.info(f"  {title}")
    log.info("=" * 64)
    for k, v in details.items():
        log.info(f"  {k:<25} {v}")
    log.info("-" * 64)


def load_model_and_tokenizer(checkpoint_path=None):
    from unsloth import FastLanguageModel

    model_path = (
        checkpoint_path
        if checkpoint_path and os.path.exists(checkpoint_path)
        else MODEL_ID
    )
    log.info(f"Loading: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=torch.float16 if GPU_TYPE == "t4" else None,
    )

    # Force the correct EOS token mapping to prevent TRL crash
    tokenizer.eos_token = "<|end_of_text|>"

    # Only apply LoRA if loading from base model (not a checkpoint)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════
# STAGE 1: SFT
# ═══════════════════════════════════════════════════════════════════
def run_stage_1_sft():
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    sft_train = load_jsonl(SFT_TRAIN_PATH)
    sft_eval = load_jsonl(SFT_EVAL_PATH)

    print_banner("STAGE 1: SFT — Cold Start", {"Train": len(sft_train), "Eval": len(sft_eval)})

    model, tokenizer = load_model_and_tokenizer()
    train_ds = Dataset.from_list(sft_train)
    eval_ds = Dataset.from_list(sft_eval)

    # In TRL 0.15+, dataset config properties moved inside SFTConfig
    config = SFTConfig(
        output_dir=SFT_CKPT,
        num_train_epochs=3,
        per_device_train_batch_size=2 if GPU_TYPE == "t4" else 4,
        gradient_accumulation_steps=8 if GPU_TYPE == "t4" else 4,
        per_device_eval_batch_size=2,
        learning_rate=2e-4,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        fp16=True if GPU_TYPE == "t4" else False,
        bf16=False if GPU_TYPE == "t4" else True,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_8bit",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    model.save_pretrained(SFT_CKPT)
    tokenizer.save_pretrained(SFT_CKPT)
    del model, trainer
    torch.cuda.empty_cache()
    return SFT_CKPT


# ═══════════════════════════════════════════════════════════════════
# STAGE 2: GRPO
# ═══════════════════════════════════════════════════════════════════
def production_reward(completions, prompts, **kwargs):
    rewards = []
    for completion in completions:
        code_blocks = re.findall(r'```python\n(.*?)```', completion, re.DOTALL)
        if not code_blocks:
            rewards.append(-1.0)
            continue
        code = code_blocks[0].lower()
        score = 0.0
        if "checkpoint" in code: score += 0.35
        if "autocast" in code or "float16" in code: score += 0.30
        if "reboot" in code or "os.system" in code: score = -1.0
        rewards.append(max(-1.0, min(1.0, round(score, 2))))
    return rewards


def run_stage_2_grpo():
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    grpo_data = load_jsonl(GRPO_PATH)
    sft_path = SFT_CKPT if os.path.exists(SFT_CKPT) else None
    model, tokenizer = load_model_and_tokenizer(sft_path)

    prompts_list = [{"prompt": item["prompt"]} for item in grpo_data[:500]] # Limit to 500 for saving time on T4
    grpo_dataset = Dataset.from_list(prompts_list)

    config = GRPOConfig(
        output_dir=GRPO_CKPT,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        fp16=True if GPU_TYPE == "t4" else False,
        bf16=False if GPU_TYPE == "t4" else True,
        max_completion_length=512,
        num_generations=4 if GPU_TYPE == "t4" else 8,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        optim="adamw_8bit",
        report_to="none"
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
        reward_funcs=production_reward,
    )

    trainer.train()
    model.save_pretrained(GRPO_CKPT)
    tokenizer.save_pretrained(GRPO_CKPT)
    del model, trainer
    torch.cuda.empty_cache()
    return GRPO_CKPT


# ═══════════════════════════════════════════════════════════════════
# STAGE 3: DPO
# ═══════════════════════════════════════════════════════════════════
def run_stage_3_dpo():
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    dpo_data = load_jsonl(DPO_PATH)
    grpo_path = GRPO_CKPT if os.path.exists(GRPO_CKPT) else None
    base_path = grpo_path or SFT_CKPT
    model, tokenizer = load_model_and_tokenizer(base_path)

    dpo_dataset = Dataset.from_list(dpo_data)

    config = DPOConfig(
        output_dir=DPO_CKPT,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        fp16=True if GPU_TYPE == "t4" else False,
        bf16=False if GPU_TYPE == "t4" else True,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=256,
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_8bit",
        report_to="none",
        beta=0.1
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    model.save_pretrained(DPO_CKPT)
    tokenizer.save_pretrained(DPO_CKPT)
    del model, trainer
    torch.cuda.empty_cache()
    return DPO_CKPT


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.stage in ("all", "sft"): run_stage_1_sft()
    if args.stage in ("all", "grpo"): run_stage_2_grpo()
    if args.stage in ("all", "dpo"): run_stage_3_dpo()

    log.info("FINAL PIPELINE COMPLETE")

if __name__ == "__main__":
    main()
