# ═══════════════════════════════════════════════════════════
# SWARM-OS TRAINING PIPELINE — 3-Stage Best-of-Best
# Run on Colab A100. Total time: ~7-8 hours. Total cost: ~$80
# ═══════════════════════════════════════════════════════════

import torch
import jsonlines
import requests
import re
from datasets import Dataset

# Lazy import so we don't crash the standard FastAPI runtime
try:
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer, DPOConfig, DPOTrainer
except ImportError:
    pass

# ── Constants ─────────────────────────────────────────────
MODEL_ID    = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
MAX_SEQ_LEN = 2048
OUTPUT_DIR  = "./swarm-os-final"


def run_stage_1_sft():
    print("\n" + "=" * 60)
    print("STAGE 1: SFT — Cold Start")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    SFT_EXAMPLES = [
        # ... Golden examples ...
    ]

    all_sft_examples = SFT_EXAMPLES
    try:
        with jsonlines.open("swarm_dataset.jsonl") as reader:
            for row in reader:
                if row["label"] == "positive" and row["reward"] > 0.3:
                    all_sft_examples.append({
                        "messages": [
                            {"role": "system", "content": "You are a Swarm-OS agent. Fix PyTorch cluster failures using memory-efficient code."},
                            {"role": "user", "content": str(row["agent_action"])},
                            {"role": "assistant", "content": f"SANDBOX_PASS | VRAM_{row['vram_peak_gb']}GB | reward={row['reward']}"}
                        ]
                    })
    except FileNotFoundError:
        pass

    def format_sft(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

    sft_dataset = Dataset.from_list([{"text": format_sft(ex)} for ex in all_sft_examples])

    sft_config = SFTConfig(
        output_dir="./stage1-sft",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        report_to="none",
    )

    sft_trainer = SFTTrainer(model=model, args=sft_config, train_dataset=sft_dataset, tokenizer=tokenizer)
    sft_trainer.train()
    
    model.save_pretrained("./stage1-sft-checkpoint")
    tokenizer.save_pretrained("./stage1-sft-checkpoint")
    print("✓ Stage 1 complete. SFT checkpoint saved.")


def production_reward(completions, prompts, **kwargs):
    """
    Physical Docker Oracle Reward
    Submits outputs directly to Swarm-OS Backend for binary, unfakeable physics validation.
    """
    rewards = []
    for completion in completions:
        code_blocks = re.findall(r'```python\n(.*?)```', completion, re.DOTALL)
        if not code_blocks:
            rewards.append(-1.0)
            continue
        
        try:
            response = requests.post("http://localhost:8000/api/code/submit", json={
                "code": code_blocks[0],
                "filename": "grpo_eval.py",
                "agent_role": "CODER",
                "mock_mode": False,
                "challenge_tier": 1
            }, timeout=45)
            result = response.json()
            rewards.append(float(result.get("reward", -1.0)))
        except requests.exceptions.RequestException:
            rewards.append(-1.0)
            
    return rewards

# Main execution trigger
if __name__ == "__main__":
    print("WARNING: This script manages the full 8-hour A100 training pipeline.")
    print("If you are running this, ensure 80GB VRAM and Swarm-OS Backend is actively running on :8000 for Physical Evaluation!")
    # run_stage_1_sft()
