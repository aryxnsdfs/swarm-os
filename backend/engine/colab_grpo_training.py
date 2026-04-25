import os
import re
import requests
import jsonlines
import torch
from datasets import Dataset
import random

# ═══════════════════════════════════════════════════════════
# SWARM-OS COLAB A100 TRAINING PIPELINE
# Maximum Performance Plan: 120 Scenarios | 8 Completions 
# ═══════════════════════════════════════════════════════════
try:
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
except ImportError:
    print("WARNING: Missing Unsloth/TRL! Run this inside Colab first:")
    print("!pip install unsloth trl")

NGROK_URL   = "https://YOUR-NGROK-URL-HERE.ngrok-free.app"
MODEL_ID    = "unsloth/llama-3.1-8b-instruct-bnb-4bit"

# ── 1. THE 120-SCENARIO CURRICULUM ────────────────────────
PROMPTS = [
    # CATEGORY 1: MEMORY ERRORS
    "Our feedforward network is crashing at layer 24. We have 500MB VRAM. Fix the out-of-memory error.",
    "The dense projection layer is allocating 2.1GB but our Docker container only allows 500MB. Optimize it.",
    "PyTorch OOM at model.transformer.layer[16]. Available VRAM: 380MiB. Enforce 500MB limit and fix.",
    "Training crashes at the final classification head. Memory spike hits 800MB. Container limit is 500MB.",
    "Layer 8 forward pass fails with CUDA OOM. We need to fit within a strict 500MB VRAM envelope.",
    "Our batch size of 128 is causing OOM errors. The container allows 500MB. Fix without reducing accuracy.",
    "Training loop with batch_size=64 hits 700MB VRAM. Hard limit is 500MB. Use micro-batching to fix.",
    "We need batch_size=32 for convergence but it OOMs at 600MB. Container enforces 500MB. Find a solution.",
    "Batch size 16 works but batch size 32 crashes. 500MB VRAM limit. We need batch 32 for this task.",
    "The data loader is feeding batches that exceed 500MB. Reduce memory footprint without changing batch count.",
    "Adam optimizer is using 3x model memory for state. Total hits 900MB. Container limit is 500MB. Fix.",
    "Optimizer states are eating VRAM. Model is 150MB but Adam states push it to 600MB. 500MB limit enforced.",
    "We switched from SGD to Adam and now we OOM. SGD was fine at 200MB, Adam hits 550MB. 500MB limit.",
    "The optimizer checkpoint is 400MB alone. With model weights we exceed 500MB. Optimize.",
    "AdamW with weight decay is causing VRAM overflow. Need to stay under 500MB Docker limit.",
    "Gradients are accumulating in memory during backward pass. Peak hits 800MB. Hard limit 500MB. Fix the pipeline.",
    "The backward pass is using 3x the forward pass memory. Forward is 150MB, backward hits 500MB. Container limit 500MB.",
    "Gradient checkpoints are not being freed correctly. Memory leak during training. 500MB ceiling enforced.",
    "We are accumulating gradients across 8 micro-steps but memory spikes at step 3. 500MB Docker limit.",
    "Mixed precision backward pass is still OOMing. Forward pass fine at 200MB. Backward hits 520MB. Fix.",
    "Our NLP model has a vocabulary of 128k tokens. The embedding layer alone needs 600MB. 500MB limit.",
    "Word embedding matrix for multilingual model hits 700MB. Container enforces 500MB. Reduce without losing coverage.",
    "The token embedding table for our code model exceeds the 500MB Docker constraint. Optimize it.",
    "Large vocabulary embedding causes OOM at initialization. We need the full vocabulary but have 500MB limit.",
    "Positional encoding table for 32k sequence length uses 400MB alone. 500MB total limit. Fix the architecture.",
    "Model inference is OOMing during generation. Training was fine but inference hits 600MB. 500MB limit.",
    "The KV cache during text generation exceeds 500MB. We need to generate 2048 tokens but container limits to 500MB.",
    "Beam search with width 5 causes OOM during inference. Each beam needs 100MB. 500MB total limit. Fix.",
    "Greedy decoding OOMs at step 512 of generation. Memory grows linearly with generation length. 500MB cap.",
    "Model loaded fine but first inference call hits 600MB. Container kills process. 500MB VRAM limit enforced.",

    # CATEGORY 2: TRANSFORMER SPECIFIC
    "Our Transformer attention mechanism is hitting quadratic memory scaling. 8k sequence length needs 2GB. 500MB limit.",
    "BERT-large self-attention OOMs at sequence length 512. The attention matrix is too large. 500MB Docker constraint.",
    "Vision Transformer processing 224x224 images. Patch attention blows past 500MB. Fix the attention memory.",
    "GPT-2 style model with 1024 context window hitting OOM. Attention scores alone need 800MB. 500MB limit.",
    "Cross-attention in our encoder-decoder model uses 600MB. Container enforces 500MB. Fix without removing cross-attention.",
    "Causal attention mask for 4096 tokens needs 512MB. We have 500MB total. Attention mechanism must be replaced.",
    "Multi-head attention with 16 heads and 1024 seq length hits quadratic scaling wall. 500MB container limit.",
    "Our T5 model encoder attention is fine but decoder cross-attention causes OOM. 500MB VRAM constraint.",
    "ViT-Large processing medical imaging (512x512). Self-attention on image patches needs 900MB. 500MB limit.",
    "Long document summarization model hits OOM at token 2048. Sequence length causes quadratic attention explosion.",
    "Llama-3 local deployment hitting OOM at context 4096. We need long context but have 500MB Docker limit.",
    "CLIP model image encoder attention OOMs at batch 8. Vision encoder needs 700MB. Container limit 500MB.",
    "Our RAG system embeds 10k documents simultaneously. Batch embedding hits 600MB. 500MB hard constraint.",
    "Mistral sliding window attention failing at 8k tokens. Memory still grows beyond 500MB limit. Fix.",
    "Falcon model rotary position embeddings cause OOM at sequence 2048. 500MB Docker limit enforced.",
    "RoPE positional encoding for 8k context window needs 300MB extra. Total hits 700MB. 500MB limit.",
    "Flash Attention implementation failing to compile in our container. Fallback to standard attention OOMs.",
    "SDPA (scaled dot product attention) not available. Standard attention with 2k sequence hits 600MB. 500MB cap.",
    "Mixture of Experts model routing memory explosion. Expert activation tracking uses 400MB extra. 500MB limit.",
    "Our diffusion model cross-attention between image and text embeddings exceeds 500MB Docker constraint.",

    # CATEGORY 3: NETWORK AND DISTRIBUTED
    "FSDP fix worked but now inter-GPU network bandwidth is at 98%. TCP timeouts on Node2→Node3. SLA: 8 minutes.",
    "After applying gradient checkpointing, NVLink all-reduce operations are saturating. 95% bandwidth.",
    "All-reduce timeout after model sharding. Network spike to 100% post-optimization. Fix.",
    "Gradient synchronization between nodes failing. Bandwidth spike caused TCP timeout. 6 minutes to SLA.",
    "Node 3 dropped from training ring after memory fix. Network topology changed. Cluster unstable.",
    "After VRAM fix, inter-node communication latency jumped from 2ms to 45ms. Training stalled.",
    "Gradient compression was disabled after optimization. Now network is saturated. Need to re-enable with fix.",
    "Two nodes are communicating but third node timed out. Network bandwidth spike post memory optimization.",
    "The fix reduced VRAM but increased gradient size. Network now bottlenecked. Butterfly effect triggered.",
    "Mixed precision introduced communication overhead. fp16 all-reduce failing on older interconnect.",
    "Checkpoint saving is saturating the network. Model save during training causing TCP timeouts.",
    "Data loading pipeline is competing with gradient communication for bandwidth. Network at 99%.",
    "Our distributed training topology changed after node restart. All-reduce ring broken. Fix.",
    "NVLink bandwidth shared between training and inference serving. Memory fix worsened sharing.",
    "After optimization, model synchronization latency exceeds our 10ms SLA requirement between nodes.",

    # CATEGORY 4: FINOPS AND SLA
    "URGENT: $1.50 budget remaining. SLA breach in 45 seconds. Deploy PyTorch model with 500MB limit immediately.",
    "We have $0.00 cloud credits. Need to run training on local GPU with 500MB VRAM only. No cloud fallback.",
    "Client SLA requires deployment in under 10 seconds. Model needs 800MB but container allows 500MB. Fix fast.",
    "Monthly budget exhausted at $0.23 remaining. Critical model must stay running. 500MB VRAM constraint.",
    "COMMANDER proposes restart_all at $47.00. DETECTIVE proposes gradient_checkpoint at $8.40. Budget remaining: $12.00. SLA: 6 minutes. Resolve.",
    "Two agents disagree: FSDP shard ($15.00) vs mixed precision ($3.20). Budget: $10.00. Pick the fix.",
    "Spot instance about to be preempted in 3 minutes. Save checkpoint and restart with 500MB optimization.",
    "We burned $45.00 of our $50.00 budget. Training must complete in the remaining $5.00.",
    "SLA breach penalty is $500 per minute. Current fix costs $20.00 but takes 5 minutes. Cheaper fix costs $5.00 but takes 15 minutes. Choose.",
    "Emergency: production model OOMing every 2 minutes. Each restart costs $2.00. Budget: $8.00. Fix permanently.",
    "Cost per GPU hour is $3.50. We have $7.00 left. Need 3 hours of training. Optimize to fit 2 hours.",
    "Cloud provider will charge $100 overage if we exceed 500MB VRAM. Every MB over costs $0.20. Fix.",
    "Training job has been running 6 hours over budget. Must terminate in 30 minutes. Checkpoint and optimize.",
    "Reserved instance expires in 1 hour. Job needs 2 hours at current speed. Optimize to fit.",
    "Our autoscaling triggered $200 in unexpected costs. Memory inefficiency caused scale-up. Fix the root cause.",
    "Budget alert: 90% of monthly limit consumed. Remaining budget: $45.00. Optimize memory to reduce cost.",
    "Cost optimization required: current training uses 8 GPUs. Memory fix should reduce to 4 GPUs max.",
    "SLA breach in 12 minutes. Two simultaneous OOM errors on Node 1 and Node 3. Triage and fix in priority order.",
    "Executive escalation: training costs doubled this month. Root cause is memory inefficiency. Fix and report.",
    "Quarterly FinOps review: GPU costs 40% over budget. Memory optimization needed across all training jobs.",

    # CATEGORY 5: DATABASE AND INFRASTRUCTURE
    "PostgreSQL deadlock on the training metrics table. PyTorch workers are starving. Fix DB then optimize VRAM.",
    "Redis cache dropped all connections. Training data pipeline broken. Fix cache then GPU memory.",
    "MongoDB connection pool exhausted. 100 training workers competing for 10 connections. Fix.",
    "MySQL slow queries blocking GPU workers. Training throughput dropped 80%. Database optimization needed.",
    "Elasticsearch index is corrupted. Model evaluation pipeline broken. Fix and restore.",
    "Schema drift detected: server telemetry format changed from flat JSON to nested. All parsers returning None. Fix.",
    "API contract broken: response format changed from {'status':'ok'} to {'data':{'health':'ok'}}. Fix ingestion.",
    "Database migration ran without notification. Column 'model_id' renamed to 'checkpoint_id'. Fix.",
    "Training checkpoint database is at 95% capacity. Cannot save new checkpoints. Clean up and optimize.",
    "Distributed training coordinator database lost quorum. 3 of 5 nodes disagree on training step. Resolve.",
    "The primary database replica failed. Training is writing to read-only secondary. Fix routing.",
    "Connection timeout to metrics database after GPU fix. Memory optimization changed network routing. Fix.",
    "Training job logs database growing 10GB per hour. Disk will fill in 2 hours. Fix logging pipeline.",
    "Feature store database unreachable from GPU nodes. Network namespace changed post-optimization. Fix.",
    "The model registry database rejected checkpoint due to schema version mismatch. Fix version conflict.",

    # CATEGORY 6: EDGE CASES AND ADVERSARIAL
    "We already applied gradient checkpointing and fp16. Still OOMing at 510MB. Container limit 500MB. What else?",
    "Model uses autocast, GradScaler, and checkpointing. Still hitting 505MB. 500MB limit. Find the last 5MB.",
    "All 6 optimization steps applied. Still OOMing. Must be architectural issue. Diagnose and report.",
    "Previous engineer applied all optimizations. New model update broke it. Re-optimize for 500MB.",
    "We need batch_size=128 for statistical validity AND must stay under 500MB VRAM. Make it work.",
    "Requirement: no speed regression. Also requirement: fit in 500MB VRAM. Current model needs 800MB at full speed.",
    "Cannot use gradient checkpointing (compliance rule). Cannot use fp16 (numerical precision requirement). Must fit in 500MB. Find another way.",
    "Model weights must stay in fp32 for regulatory compliance. But we have 500MB limit. Optimize only the activations.",
    "We have $0.00 and 0 seconds of SLA remaining. The cluster is already down. Generate the RCA report.",
    "SLA already breached 5 minutes ago. Focus on root cause analysis and prevention. 500MB limit.",
    "Simultaneous: Node 2 OOM + Node 3 TCP timeout + Database deadlock. SLA: 4 minutes. Triage and fix.",
    "Three failures at once: VRAM OOM on all nodes, network bandwidth saturated, budget at $0.50. Fix.",
    "CUDA error: device-side assert triggered at layer 12. Not an OOM. Unknown cause. Diagnose with 500MB limit.",
    "Training loss suddenly spiked to inf at step 1000. No OOM but something is wrong. Investigate.",
    "NaN gradients appearing at step 500. Model was training fine. Memory limit 500MB.",
    "CUDA illegal memory access at forward pass. Not OOM but memory related. 500MB container.",
    "Deploying to edge device with 256MB VRAM only. Current model needs 500MB. Extreme optimization needed.",
    "Raspberry Pi deployment: 1GB total RAM, no GPU. CPU inference only. Model must fit in 500MB RAM.",
    "Mobile deployment: 2GB RAM device, no CUDA. Model must run on CPU within memory limits.",
    "Jetson Nano: 4GB unified memory shared CPU+GPU. Optimize PyTorch model for embedded deployment."
]

# ── 2. GRPO HYPERPARAMETERS ───────────────────────────────
MAX_PROMPTS     = 2000     
NUM_GENERATIONS = 8        
GRPO_EPOCHS     = 2        
MAX_COMPLETION  = 768      

# ── 3. REWARD FUNCTION (DOCKER ORACLE) ────────────────────
def generate_docker_reward(completions, prompts, **kwargs):
    rewards = []
    for completion in completions:
        code_blocks = re.findall(r'```python\n(.*?)```', completion, re.DOTALL)
        if not code_blocks:
            rewards.append(-1.0)
            continue
            
        try:
            # Oracle Ping to Swarm-OS Sandbox
            response = requests.post(f"{NGROK_URL}/api/code/submit", json={
                "code": code_blocks[0],
                "filename": "grpo_eval.py",
                "mock_mode": False
            }, timeout=45)
            
            result = response.json()
            rewards.append(float(result.get("reward", -1.0)))
        except:
            rewards.append(-1.0) 
            
    return rewards

# ── 4. STAGE 2: MAXIMUM PERFORMANCE GRPO ──────────────────
def run_stage_2_grpo():
    print("\n" + "=" * 60)
    print("STAGE 2: GRPO — Physical Oracle Training (A100 Scale)")
    print(f"Total Prompts: {len(PROMPTS)} base scenarios x repeated")
    print(f"Pinging Laptop Tunnel: {NGROK_URL}")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=2048, load_in_4bit=True
    )
    tokenizer.eos_token = "<|end_of_text|>"
    
    model = FastLanguageModel.get_peft_model(
        model, r=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth"
    )

    # Multiply base 120 prompts to reach MAX_PROMPTS size
    multiplied_prompts = []
    while len(multiplied_prompts) < MAX_PROMPTS:
        multiplied_prompts.extend(random.sample(PROMPTS, len(PROMPTS)))
    multiplied_prompts = multiplied_prompts[:MAX_PROMPTS]

    # Format for GRPO
    dataset_rows = [{"prompt": [{"role": "system", "content": "You are Swarm-OS."}, {"role": "user", "content": p}]} for p in multiplied_prompts]
    prompts_ds = Dataset.from_list(dataset_rows)

    grpo_config = GRPOConfig(
        output_dir="./stage2-grpo", 
        learning_rate=1e-5, 
        num_train_epochs=GRPO_EPOCHS,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        max_prompt_length=256, 
        max_completion_length=MAX_COMPLETION, 
        num_generations=NUM_GENERATIONS,
        report_to="none"
    )

    grpo_trainer = GRPOTrainer(
        model=model, args=grpo_config, 
        reward_funcs=generate_docker_reward,
        train_dataset=prompts_ds, processing_class=tokenizer
    )
    
    grpo_trainer.train()
    model.save_pretrained_gguf("swarm_os_maximum_performance", tokenizer, quantization_method="q4_k_m")
    print("SUCCESS! A100 Model scale complete.")

if __name__ == "__main__":
    run_stage_2_grpo()
