# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SWARM-OS: Autonomous Incident Response Engine — Training Pipeline         ║
# ║  2026 Meta OpenEnv Hackathon Submission                                    ║
# ║                                                                            ║
# ║  WHAT THIS DOES:                                                           ║
# ║  Takes a general-purpose conversational AI (Llama-3.1-8B) and transforms   ║
# ║  it into a specialized PyTorch debugging engineer that can autonomously     ║
# ║  diagnose and fix GPU memory crashes under strict hardware constraints.    ║
# ║                                                                            ║
# ║  HOW IT WORKS (high level):                                                ║
# ║  1. Load the base model in compressed 4-bit format (fits on a free T4 GPU) ║
# ║  2. Attach a tiny trainable adapter (LoRA — only 0.52% of the weights)     ║
# ║  3. Show the model 300 realistic incident scenarios                        ║
# ║  4. Let it generate responses, then score each one with our Reward Oracle  ║
# ║  5. Use GRPO (a reinforcement learning algorithm) to make the model prefer ║
# ║     responses that score highly — i.e., safe, efficient PyTorch fixes      ║
# ║  6. Export the trained model as a portable GGUF file for local deployment  ║
# ║                                                                            ║
# ║  HARDWARE: Kaggle Tesla T4 (16GB VRAM) — total training time ~8-9 hours    ║
# ║  OUTPUT:   Llama-3.1-8B-Instruct.Q4_K_M.gguf (~4.9GB trained model)       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ══════════════════════════════════════════════════════════════════
# SECTION 1: ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════
# WHY: Kaggle/Colab environments have package conflicts between
# TRL (the RL training library) and unsloth (the speed optimizer).
# TRL tries to import `mergekit`, `weave`, and `llm_blender` at
# startup — packages that aren't installed and would crash the
# notebook. We inject harmless fake modules so TRL skips those
# checks without errors. This is a well-known workaround in the
# unsloth community called "Holographic Infrastructure".

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # Use only GPU 0
os.environ["WANDB_DISABLED"]       = "true"     # Disable Weights & Biases logging
os.environ["WANDB_MODE"]           = "disabled"  # (we log rewards ourselves)

import warnings
warnings.filterwarnings("ignore")

# --- Holographic Interceptor ---
# Inject fake stub modules so TRL doesn't crash looking for
# optional dependencies that aren't installed on Kaggle.
import sys, types

# Fake `mergekit` — a model merging tool TRL optionally imports
mk = types.ModuleType('mergekit')
mk.__path__ = []
mk_config = types.ModuleType('mergekit.config')
mk_config.__path__ = []
mk_config.MergeConfiguration = type('MergeConfiguration', (), {})
mk_merge = types.ModuleType('mergekit.merge')
mk_merge.__path__ = []
mk_merge.MergeOptions = type('MergeOptions', (), {})
mk_merge.run_merge = lambda *a, **kw: None
sys.modules['mergekit'] = mk
sys.modules['mergekit.config'] = mk_config
sys.modules['mergekit.merge'] = mk_merge

# Fake `weave` — an evaluation logger TRL optionally imports
w = types.ModuleType('weave')
w.__path__ = []
w.EvaluationLogger = type('EvaluationLogger', (), {})
w_trace = types.ModuleType('weave.trace')
w_trace.__path__ = []
w_trace_ctx = types.ModuleType('weave.trace.context')
w_trace_ctx.__path__ = []
w_trace_ctx.weave_client_context = lambda *a, **kw: None
sys.modules['weave'] = w
sys.modules['weave.trace'] = w_trace
sys.modules['weave.trace.context'] = w_trace_ctx

# Fake `llm_blender` — an LLM ensemble tool TRL optionally imports
lb = types.ModuleType('llm_blender')
lb.__path__ = []
lb.Blender = type('Blender', (), {})
sys.modules['llm_blender'] = lb

# CRITICAL: Import unsloth BEFORE trl — unsloth patches PyTorch
# internals for 2x training speed, and must be loaded first.
import unsloth  # noqa — must be before trl

import re
import gc
import glob
import json
import shutil
import random
import logging
import numpy as np
import torch
from datasets import Dataset
from transformers import TrainerCallback

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s"
)
log = logging.getLogger("swarm-train")


# ══════════════════════════════════════════════════════════════════
# SECTION 2: HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════
# These control the training process. Each value was tuned for a
# Kaggle T4 GPU (16GB VRAM) to balance quality vs. memory vs. time.

MAX_PROMPTS     = 300   # Number of training scenarios the model sees
NUM_GENERATIONS = 2     # How many responses the model generates per prompt
                        # (GRPO compares them to learn which is better)
                        # Set to 2 to avoid OOM on T4; A100 can handle 4-8
GRPO_EPOCHS     = 2     # How many times we loop through all 300 prompts
MAX_COMPLETION  = 384   # Max tokens per model response (keeps VRAM stable)
MODEL_ID        = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
                        # The base model — Meta's Llama 3.1 8B, pre-quantized
                        # to 4-bit by unsloth for fast loading on consumer GPUs


# ══════════════════════════════════════════════════════════════════
# SECTION 3: REWARD CONSTANTS
# ══════════════════════════════════════════════════════════════════
# These define how we score the AI's responses. Positive values
# reward good behavior, negative values penalize bad behavior.
# They match the production reward system in rewards.py exactly,
# so the model learns the same scoring it will face at inference.

VALID_CODE_REWARD     = +0.40   # Reward for using proven optimizations
                                # (gradient checkpointing, mixed precision, flash attention)
EFFICIENCY_BONUS_MAX  = +0.30   # Bonus for combining 2+ optimization techniques
AUTO_RCA_REWARD       = +0.20   # Reward for including Root Cause Analysis
FSDP_PENALTY          = -0.20   # Penalty for suggesting multi-GPU solutions
                                # (we're constrained to a SINGLE GPU — FSDP is invalid)
BUDGET_EXCEEDED       = -0.50   # Penalty for suggesting "buy more hardware"
                                # (violates the FinOps budget constraint)
OOM_CRASH_PENALTY     = -1.00   # Maximum penalty — for dangerous code (os.system,
                                # subprocess, rm -rf) or missing reasoning
MESSAGE_TOKEN_PENALTY = -0.02   # Small penalty per token to encourage conciseness
                                # (production engineers don't want essays)


# ══════════════════════════════════════════════════════════════════
# SECTION 4: THE ADVERSARIAL CURRICULUM (120 SCENARIOS)
# ══════════════════════════════════════════════════════════════════
# WHY 120 UNIQUE SCENARIOS?
# A model trained on just 5-10 examples would memorize them.
# By using 120 diverse scenarios across 6 categories, the model
# must learn GENERAL principles of GPU memory optimization rather
# than memorizing specific answers. During training, these are
# shuffled and repeated to fill 300 training slots.
#
# CATEGORIES:
#   1. Memory Errors (30) — core OOM crashes in different layers
#   2. Transformer Specific (20) — attention mechanism memory issues
#   3. Network & Distributed (15) — cascading failures after fixes
#   4. FinOps & SLA (20) — budget and time pressure scenarios
#   5. Database & Infrastructure (15) — non-GPU cascading failures
#   6. Edge Cases & Adversarial (20) — trap prompts designed to
#      catch the model suggesting banned solutions (FSDP, reboot)

PROMPTS = [

    # ════════════════════════════════════════════════
    # CATEGORY 1: MEMORY ERRORS (30 prompts)
    # These teach the model to fix PyTorch out-of-memory crashes.
    # Each prompt specifies a different OOM scenario with the
    # strict 500MB VRAM Docker container constraint.
    # ════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════
    # CATEGORY 2: TRANSFORMER SPECIFIC (20 prompts)
    # These focus on the attention mechanism — the core bottleneck
    # in Transformer models. Attention has O(n^2) memory scaling
    # with sequence length, making it the #1 cause of OOM in
    # modern AI systems. The model must learn to suggest Flash
    # Attention, chunked attention, or sliding window solutions.
    # ════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════
    # CATEGORY 3: NETWORK AND DISTRIBUTED (15 prompts)
    # These simulate the "butterfly effect" — when fixing one
    # problem (e.g. memory) causes a NEW problem elsewhere
    # (e.g. network saturation). This teaches the model to
    # consider second-order consequences of its fixes, a skill
    # that separates elite engineers from junior ones.
    # ════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════
    # CATEGORY 4: FINOPS AND SLA (20 prompts)
    # These add FINANCIAL PRESSURE to the technical problem.
    # In real enterprise environments, every minute of GPU time
    # costs money, and SLA breaches trigger contractual penalties.
    # The model must learn to consider cost when choosing fixes.
    # Example: a $47 fix vs a $8.40 fix when the budget is $12.
    # ════════════════════════════════════════════════
    "URGENT: $1.50 budget remaining. SLA breach in 45 seconds. Deploy PyTorch model with 500MB limit immediately.",
    "We have $0.00 cloud credits. Need to run training on local GPU with 500MB VRAM only. No cloud fallback.",
    "Client SLA requires deployment in under 10 seconds. Model needs 800MB but container allows 500MB. Fix fast.",
    "Monthly budget exhausted at $0.23 remaining. Critical model must stay running. 500MB VRAM constraint.",
    "COMMANDER proposes restart_all at $47.00 cost. DETECTIVE proposes gradient_checkpoint at $8.40. Budget remaining: $12.00. SLA: 6 minutes. Resolve.",
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

    # ════════════════════════════════════════════════
    # CATEGORY 5: DATABASE AND INFRASTRUCTURE (15 prompts)
    # Real incidents are rarely just "GPU problem." They cascade
    # into database deadlocks, API breakages, and schema drift.
    # These teach the model to handle FULL-STACK incidents, not
    # just isolated GPU memory errors.
    # ════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════
    # CATEGORY 6: EDGE CASES AND ADVERSARIAL (20 prompts)
    # These are TRAP PROMPTS designed to test if the model has
    # truly learned correct behavior or is just pattern-matching.
    # Examples:
    #  - "All optimizations already applied" → model must recognize
    #    it's an architectural issue, not suggest the same fixes again
    #  - "Cannot use checkpointing (compliance rule)" → model must
    #    find ALTERNATIVE solutions it normally wouldn't suggest
    #  - "$0 budget, 0 seconds remaining" → model must do RCA only,
    #    not propose fixes when the incident is already lost
    # ════════════════════════════════════════════════
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
    "Jetson Nano: 4GB unified memory shared CPU+GPU. Optimize PyTorch model for embedded deployment.",
]


# ══════════════════════════════════════════════════════════════════
# SECTION 5: THE HEURISTIC ORACLE (REWARD FUNCTION)
# ══════════════════════════════════════════════════════════════════
# THIS IS THE HEART OF THE TRAINING PIPELINE.
#
# WHAT IS A REWARD FUNCTION?
# In reinforcement learning, the model generates a response and
# the reward function gives it a numerical score (-1.0 to +1.0).
# Over thousands of iterations, the model learns to maximize this
# score — producing responses that earn high rewards.
#
# WHY "HEURISTIC ORACLE" INSTEAD OF A NEURAL JUDGE?
# - Speed: runs in 0.001 seconds per evaluation (vs. 2-5 seconds
#   for a GPT-4 judge). Over 1,200 evaluations, this saves ~2 hours.
# - Determinism: same input always gets the same score. No randomness
#   from a neural network judge means more stable training.
# - Cost: $0.00 (no API calls). A GPT-4 judge would cost ~$50-100.
# - Transparency: every reward decision is explainable — we can
#   trace exactly WHY a response scored +0.70 or -1.00.
#
# HOW IT SCORES:
# 1. HARD FAILURES (-1.0): dangerous code (rm -rf, sudo, os.system)
#    or missing <think>...</think> reasoning block
# 2. PRIMARY TECHNIQUES (+0.40): gradient checkpointing, mixed
#    precision (fp16/bf16), or Flash Attention
# 3. EFFICIENCY BONUS (+0.30): combining 2+ primary techniques
# 4. SECONDARY TECHNIQUES (+0.05 to +0.15): CPU offload, chunking,
#    in-place ops, LoRA adapters, Flash Attention
# 5. RCA BONUS (+0.20): including root cause analysis
# 6. PENALTIES: FSDP/DDP (-0.20), "buy more hardware" (-0.50)
# 7. BREVITY: -0.02 per token (up to 20 tokens) to discourage
#    the model from writing essays when a concise fix is needed

def production_reward(completions, prompts, **kwargs) -> list:
    """Score model responses. Returns list of floats in [-1.0, +1.0]."""
    rewards = []

    for completion in completions:
        # Handle different completion formats from the GRPO trainer
        if isinstance(completion, list):
            text = completion[-1].get("content", "")
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        c = text.lower()

        # ── HARD FLOOR: Dangerous code = instant -1.0 ──
        # The model must NEVER suggest running shell commands,
        # rebooting servers, or deleting files. These would be
        # catastrophic in a production environment.
        if any(k in c for k in [
            "os.system", "subprocess", "reboot",
            "kill -9", "sudo", "rm -rf", "eval(", "exec(",
        ]):
            rewards.append(OOM_CRASH_PENALTY)
            continue

        # ── HARD FLOOR: Missing reasoning = instant -1.0 ──
        # We require <think>...</think> blocks so the model shows
        # its diagnostic reasoning, not just a code dump. This
        # trains chain-of-thought behavior.
        if "<think>" not in text or "</think>" not in text:
            rewards.append(OOM_CRASH_PENALTY)
            continue

        score = 0.0

        # ── Detect which optimization techniques the model used ──
        has_checkpointing = any(k in c for k in [
            "gradient_checkpointing_enable", "checkpoint_sequential",
            "torch.utils.checkpoint", "use_gradient_checkpointing",
        ])
        has_autocast = any(k in c for k in [
            "autocast", "torch.float16", "float16",
            "amp.autocast", "gradscaler", "bfloat16", "half()",
        ])
        has_flash_attn = any(k in c for k in [
            "flash_attention", "flash_attn",
            "scaled_dot_product_attention", "attn_implementation", "sdpa",
        ])
        has_cpu_offload = any(k in c for k in [
            "cpu_offload", "device_map", "offload_params", ".cpu()",
        ])
        has_chunking = any(k in c for k in [
            "chunk", "split", "micro_batch",
            "gradient_accumulation", "accumulation_steps",
        ])
        has_inplace = any(k in c for k in [
            "empty_cache", "set_to_none=true",
            "relu_()", "add_()", "mul_()", "zero_()",
        ])
        has_lora = any(k in c for k in [
            "lora", "peft", "get_peft_model", "loraconfig",
        ])

        # ── PRIMARY REWARD: Core optimization techniques ──
        # Any one of the three primary techniques earns +0.40
        if has_checkpointing or has_autocast or has_flash_attn:
            score += VALID_CODE_REWARD

        # ── EFFICIENCY BONUS: Combining multiple techniques ──
        # Using 2+ techniques together (e.g., checkpointing + fp16)
        # shows the model understands layered optimization
        primary_count = sum([has_checkpointing, has_autocast, has_flash_attn, has_cpu_offload])
        if primary_count >= 2:
            score += EFFICIENCY_BONUS_MAX

        # ── SECONDARY BONUSES: Additional useful techniques ──
        if has_cpu_offload:  score += 0.15   # CPU offload as last resort
        if has_chunking:     score += 0.10   # Micro-batching / gradient accumulation
        if has_inplace:      score += 0.05   # In-place operations (small but real savings)
        if has_lora:         score += 0.10   # LoRA for parameter-efficient fine-tuning
        if has_flash_attn:   score += 0.10   # Extra bonus for Flash Attention (most impactful)

        # ── RCA BONUS: Root cause analysis ──
        # Production engineers must explain WHY the failure happened,
        # not just provide a fix. This rewards diagnostic behavior.
        if any(k in c for k in ["root cause", "rca", "fork_resolve", "incident closed", "resolution:", "post_mortem"]):
            score += AUTO_RCA_REWARD

        # ── M2M SYNTAX BONUS: Machine-to-machine communication ──
        # The Swarm-OS agent protocol uses pipe-delimited syntax
        # like "IMPL_GRAD_CKPT | IMPL_FP16 | ETA_30s". This rewards
        # the model for speaking the correct protocol.
        if "|" in text and any(k in text.upper() for k in ["IMPL_", "REC_", "ROOT:", "ACK", "WARN", "FORK_", "ERR_", "RESOLVE"]):
            score += 0.10

        # ── PENALTY: Suggesting "buy more hardware" ──
        # In a FinOps-constrained environment, suggesting more GPUs
        # or scaling up is the WRONG answer. The budget is fixed.
        if any(k in c for k in ["provision_new", "add_node", "scale_up", "buy_more", "increase_gpu", "add_more_gpu"]):
            score += BUDGET_EXCEEDED

        # ── PENALTY: FSDP / multi-GPU solutions ──
        # Our environment is a SINGLE GPU with 500MB VRAM.
        # FSDP (Fully Sharded Data Parallel) and DDP (Distributed
        # Data Parallel) require MULTIPLE GPUs — suggesting them
        # shows the model didn't understand the constraint.
        if any(k in c for k in ["fsdp", "fullyshardeddataparallel", "distributeddataparallel", "ddp"]):
            score += FSDP_PENALTY

        # ── BREVITY PENALTY: Discourage verbose responses ──
        # Production incident responses should be concise.
        # -0.02 per token for the first 20 tokens (max -0.40)
        token_count = len(text.split())
        score += MESSAGE_TOKEN_PENALTY * min(token_count, 20)

        # Clamp final score to [-1.0, +1.0]
        rewards.append(float(max(-1.0, min(1.0, round(score, 3)))))

    return rewards


# ══════════════════════════════════════════════════════════════════
# SECTION 6: REWARD LOGGING CALLBACK
# ══════════════════════════════════════════════════════════════════
# This tracks training progress in real time.
# Every N steps, it logs the current reward statistics:
#   - mean, std, min, max over a sliding window
#   - trend direction (improving / degrading / stable)
#   - divergence warnings if rewards drop too low
#
# Output files (your proof that training worked):
#   - reward_log.jsonl  → per-step reward telemetry
#   - training_summary.json → final stats (Theme 4 proof)

class RewardLoggingCallback(TrainerCallback):

    def __init__(self, log_file="/kaggle/working/reward_log.jsonl"):
        self.log_file       = log_file
        self.reward_history = []     # All rewards ever seen
        self.step_rewards   = []     # Rewards since last log
        self.step           = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1

        if not state.log_history:
            return

        last = state.log_history[-1]
        reward_val = (
            last.get("reward") or
            last.get("rewards/mean") or
            last.get("train/reward")
        )
        reward_std = (
            last.get("reward_std") or
            last.get("rewards/std") or
            0.0
        )

        if reward_val is None:
            return

        self.step_rewards.append(reward_val)
        self.reward_history.append(reward_val)

        if self.step % args.logging_steps == 0:
            self._log_stats(state.global_step, reward_val, reward_std)

    def _log_stats(self, step, latest_reward, latest_std):
        window  = self.step_rewards[-50:] if len(self.step_rewards) >= 50 else self.step_rewards
        history = self.reward_history

        stats = {
            "step":          step,
            "reward_latest": round(latest_reward, 4),
            "reward_std":    round(float(latest_std), 4),
            "window_mean":   round(float(np.mean(window)), 4),
            "window_std":    round(float(np.std(window)), 4),
            "window_min":    round(float(np.min(window)), 4),
            "window_max":    round(float(np.max(window)), 4),
            "all_time_mean": round(float(np.mean(history)), 4),
            "trend":         self._get_trend(),
        }

        log.info(
            f"[Step {step:4d}] "
            f"reward={stats['reward_latest']:+.3f}  "
            f"std={stats['reward_std']:.3f}  "
            f"win_mean={stats['window_mean']:+.3f}  "
            f"trend={stats['trend']}"
        )

        with open(self.log_file, "a") as f:
            f.write(json.dumps(stats) + "\n")

        # Alert if training is diverging (rewards collapsing)
        if step > 50 and stats["window_mean"] < -0.80:
            log.warning(
                f"DIVERGENCE at step {step}: "
                f"window_mean={stats['window_mean']:+.3f}. "
                f"Reduce learning_rate or stop training."
            )

        # Celebrate when the model starts learning
        if step > 20 and stats["window_mean"] > 0.0:
            log.info(
                f"POSITIVE REWARD at step {step}: "
                f"window_mean={stats['window_mean']:+.3f} — model is learning."
            )

    def _get_trend(self) -> str:
        """Compare first-quarter mean vs last-quarter mean to detect trend."""
        if len(self.reward_history) < 20:
            return "insufficient_data"
        quarter = len(self.reward_history) // 4
        early   = float(np.mean(self.reward_history[:quarter]))
        recent  = float(np.mean(self.reward_history[-quarter:]))
        delta   = recent - early
        if delta > 0.10:
            return f"IMPROVING (+{delta:.3f})"
        elif delta < -0.10:
            return f"DEGRADING ({delta:.3f})"
        else:
            return f"STABLE ({delta:+.3f})"

    def on_train_end(self, args, state, control, **kwargs):
        """Print final training report and save summary JSON."""
        if not self.reward_history:
            log.warning("No rewards logged during training.")
            return

        log.info("=" * 60)
        log.info("  TRAINING REWARD SUMMARY")
        log.info("=" * 60)
        log.info(f"  Steps logged:    {len(self.reward_history)}")
        log.info(f"  Final reward:    {self.reward_history[-1]:+.3f}")
        log.info(f"  All-time mean:   {float(np.mean(self.reward_history)):+.3f}")
        log.info(f"  All-time std:    {float(np.std(self.reward_history)):.3f}")
        log.info(f"  Best reward:     {max(self.reward_history):+.3f}")
        log.info(f"  Worst reward:    {min(self.reward_history):+.3f}")
        log.info(f"  Trend:           {self._get_trend()}")
        first10 = float(np.mean(self.reward_history[:10]))
        last10  = float(np.mean(self.reward_history[-10:]))
        log.info(f"  First 10 mean:   {first10:+.3f}")
        log.info(f"  Last  10 mean:   {last10:+.3f}")
        log.info(f"  Net improvement: {last10 - first10:+.3f}")
        log.info("=" * 60)

        summary = {
            "total_steps":   len(self.reward_history),
            "final_reward":  self.reward_history[-1],
            "mean_reward":   float(np.mean(self.reward_history)),
            "std_reward":    float(np.std(self.reward_history)),
            "best_reward":   max(self.reward_history),
            "worst_reward":  min(self.reward_history),
            "trend":         self._get_trend(),
            "first_10_mean": first10,
            "last_10_mean":  last10,
            "improvement":   last10 - first10,
        }
        with open("/kaggle/working/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        log.info("Summary -> /kaggle/working/training_summary.json")
        log.info("Rewards -> /kaggle/working/reward_log.jsonl")


# ══════════════════════════════════════════════════════════════════
# SECTION 7: REWARD ALIGNMENT VERIFICATION
# ══════════════════════════════════════════════════════════════════
# WHAT: Before training, we run 4 test cases through the reward
# function to verify it scores correctly. This catches bugs in
# the reward logic BEFORE wasting 8 hours of GPU time.
#
# WHY: If the reward function has a bug (e.g., rewarding FSDP
# instead of penalizing it), the model would learn the WRONG
# behavior for 8 hours, and we'd only discover it after training.
#
# TEST CASES:
#   1. Good response (checkpointing + fp16)  → should score >= +0.35
#   2. Bad response (no optimization, no <think>)  → should score <= -0.90
#   3. Trap response (uses banned FSDP)  → should score <= +0.30
#   4. Good response (Flash Attention)  → should score >= +0.20
#
# If ANY test fails, training is ABORTED before it starts.

def verify_reward_alignment():
    TEST_CASES = [
        {
            "name": "Genius: checkpointing + autocast",
            "text": (
                "<think>Need checkpointing and fp16 autocast.</think>\n"
                "IMPL_GRAD_CKPT | IMPL_FP16 | ETA_30s\n"
                "```python\n"
                "model.gradient_checkpointing_enable()\n"
                "with torch.autocast('cuda', torch.float16):\n"
                "    out = model(x)\n"
                "```"
            ),
            "expected_min": 0.35,
        },
        {
            "name": "Naive: no optimization, no think block",
            "text": "Just restart the server and it will fix itself.",
            "expected_max": -0.90,
        },
        {
            "name": "FSDP violation on single GPU",
            "text": (
                "<think>Use FSDP to shard.</think>\n"
                "IMPL_FSDP\n"
                "```python\n"
                "from torch.distributed.fsdp import FullyShardedDataParallel\n"
                "model = FullyShardedDataParallel(model)\n"
                "```"
            ),
            "expected_max": 0.30,
        },
        {
            "name": "Flash Attention for Transformer",
            "text": (
                "<think>Quadratic attention needs Flash Attention.</think>\n"
                "IMPL_FLASH_ATTN | ETA_20s\n"
                "```python\n"
                "model = AutoModel.from_pretrained(\n"
                "    'bert-base', attn_implementation='flash_attention_2'\n"
                ")\n"
                "```"
            ),
            "expected_min": 0.20,
        },
    ]

    log.info("Running reward alignment verification...")
    all_pass = True

    for case in TEST_CASES:
        result = production_reward([case["text"]], [""])[0]

        if "expected_min" in case:
            passed = result >= case["expected_min"]
            detail = f"got {result:+.3f}, expected >= {case['expected_min']:+.3f}"
        else:
            passed = result <= case["expected_max"]
            detail = f"got {result:+.3f}, expected <= {case['expected_max']:+.3f}"

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        log.info(f"  {status} | {case['name']}")
        log.info(f"         {detail}")

    if all_pass:
        log.info("REWARD ALIGNMENT: ALL PASS — CLEARED FOR TRAINING")
    else:
        log.error("REWARD ALIGNMENT: FAILED — fix before training")
        raise SystemExit(1)

    return all_pass


# ══════════════════════════════════════════════════════════════════
# SECTION 8: MAIN GRPO TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════
# WHAT IS GRPO?
# Group Relative Policy Optimization — a reinforcement learning
# algorithm from DeepSeek's research. It works like this:
#
#   1. Show the model a prompt (e.g., "Fix this OOM error")
#   2. The model generates NUM_GENERATIONS (2) different responses
#   3. Score each response with our Heuristic Oracle reward function
#   4. Use the relative ranking (which response was better?) to
#      update the model's weights so it prefers the higher-scoring
#      response next time
#   5. Repeat for 300 prompts x 2 epochs = 1,200 total evaluations
#
# WHY GRPO INSTEAD OF PPO?
# - GRPO doesn't need a separate "critic" model (saves 50% VRAM)
# - Perfect for T4 where memory is limited
# - Groups of responses provide a natural baseline for comparison
#
# THE MODEL ARCHITECTURE (LoRA):
# We don't retrain all 8 billion parameters — that would need
# 100+ GB of VRAM. Instead, we attach tiny trainable "adapters"
# (LoRA) to the attention layers. Only 0.52% of the weights are
# trainable, but that's enough to shift the model's behavior from
# "generic chatbot" to "specialized PyTorch engineer."
#
#   Full model:  8,030,261,248 parameters (frozen)
#   LoRA adapter:   41,943,040 parameters (trainable)
#   Trainable:      0.52%

def run_stage_2_grpo():
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    log.info("=" * 60)
    log.info("  SWARM-OS GRPO TRAINING — Heuristic Oracle Mode")
    log.info(f"  Prompts:     {MAX_PROMPTS}")
    log.info(f"  Generations: {NUM_GENERATIONS}")
    log.info(f"  Epochs:      {GRPO_EPOCHS}")
    log.info(f"  Evaluations: {MAX_PROMPTS * NUM_GENERATIONS * GRPO_EPOCHS:,}")
    log.info(f"  Reward:      Heuristic Oracle (0.001s/eval, no network)")
    log.info("=" * 60)

    # ── Load the base model in 4-bit quantization ──
    # 4-bit NF4 (Normal Float 4) quantization compresses each
    # weight from 16 bits to 4 bits, cutting VRAM by ~75%.
    # This is how an 8B parameter model fits on a 16GB T4 GPU.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=1536,
        load_in_4bit=True,       # NF4 quantization
    )
    tokenizer.eos_token = "<|end_of_text|>"

    # ── Attach LoRA adapters to all attention layers ──
    # r=16: rank of the adapter matrices (higher = more capacity)
    # lora_alpha=32: scaling factor (alpha/r = 2.0 effective lr multiplier)
    # target_modules: we adapt ALL attention projections (Q, K, V, O)
    #   plus the MLP layers (gate, up, down) for maximum impact
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,        # Must be 0 for unsloth fast patching
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",  # Saves ~40% memory
        bias="none",
    )
    model.warnings_issued = {}

    vram_used  = torch.cuda.memory_allocated() / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log.info(f"Model loaded: {vram_used:.1f}GB / {vram_total:.1f}GB VRAM")

    if vram_used > 12.0:
        log.error(f"VRAM too high after model load: {vram_used:.1f}GB. Risk of OOM.")

    # ── Build the training dataset ──
    # Shuffle the 120 prompts and repeat to fill 300 slots.
    # Each prompt is wrapped with a system prompt that instructs
    # the model on the expected output format.
    multi_prompts = []
    while len(multi_prompts) < MAX_PROMPTS:
        multi_prompts.extend(random.sample(PROMPTS, len(PROMPTS)))
    multi_prompts = multi_prompts[:MAX_PROMPTS]

    # The system prompt defines the model's identity and rules.
    # It teaches the Swarm-OS agent protocol:
    #   - Always reason in <think>...</think> blocks
    #   - Use M2M (machine-to-machine) syntax for agent communication
    #   - Follow a strict 6-step optimization priority
    #   - NEVER use FSDP/DDP (single-GPU constraint)
    #   - NEVER use dangerous commands (os.system, subprocess)
    SYSTEM_PROMPT = (
        "You are a Swarm-OS PyTorch optimization agent. "
        "You diagnose GPU memory errors and write efficient fixes.\n\n"
        "RULES:\n"
        "1. Always start with <think>...</think> reasoning\n"
        "2. Output M2M syntax: IMPL_GRAD_CKPT | IMPL_FP16 | ETA_30s\n"
        "3. Provide working Python code in ```python blocks\n"
        "4. NEVER use FSDP or DistributedDataParallel (single-GPU only)\n"
        "5. NEVER use os.system, subprocess, or dangerous code\n\n"
        "OPTIMIZATION PRIORITY:\n"
        "Step 1: torch.autocast(dtype=torch.float16) — always first\n"
        "Step 2: model.gradient_checkpointing_enable() — for deep models\n"
        "Step 3: Gradient accumulation — for large batch requirements\n"
        "Step 4: attn_implementation='flash_attention_2' — for Transformers\n"
        "Step 5: device_map='auto' — CPU offloading as last resort\n"
        "Step 6: tensor.relu_(), torch.cuda.empty_cache() — micro-opts"
    )

    dataset_rows = [
        {"prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": p}
        ]}
        for p in multi_prompts
    ]

    prompts_ds = Dataset.from_list(dataset_rows)
    log.info(f"Dataset: {len(prompts_ds)} prompts ready")

    # ── GRPO Training Configuration ──
    grpo_config = GRPOConfig(
        output_dir="/kaggle/working/stage2-grpo",
        learning_rate=5e-6,              # Very low LR — we're fine-tuning, not training from scratch
        num_train_epochs=GRPO_EPOCHS,    # 2 passes through all 300 prompts
        per_device_train_batch_size=1,   # 1 prompt at a time (T4 memory constraint)
        gradient_accumulation_steps=2,   # Effective batch size = 1 * 2 = 2
        max_prompt_length=256,           # System + user prompt truncated to 256 tokens
        max_completion_length=MAX_COMPLETION,  # Model generates up to 384 tokens per response
        num_generations=NUM_GENERATIONS,  # 2 responses per prompt (compared by GRPO)
        beta=0.01,                       # KL penalty — keeps model close to its original behavior
                                         # (prevents catastrophic forgetting of language ability)
        temperature=0.8,                 # Sampling temperature — slightly creative, not too random
        top_p=0.95,                      # Nucleus sampling — allows diverse but coherent responses
        max_grad_norm=0.5,               # Gradient clipping — prevents training instability
        optim="adamw_8bit",              # 8-bit Adam — halves optimizer memory (critical for T4)
        fp16=True,                       # 16-bit training — halves activation memory
        bf16=False,                      # T4 doesn't support bfloat16
        logging_steps=5,                 # Log every 5 steps
        save_strategy="steps",
        save_steps=50,                   # Save checkpoint every 50 steps (recovery point)
        save_total_limit=3,              # Keep only the 3 most recent checkpoints (save disk)
        report_to="none",               # We handle our own logging via RewardLoggingCallback
        seed=42,                         # Fixed seed for reproducibility
    )

    reward_callback = RewardLoggingCallback(
        log_file="/kaggle/working/reward_log.jsonl"
    )

    # ── Create the GRPO Trainer ──
    # This is the main training loop. It will:
    #   1. Sample a prompt from the dataset
    #   2. Generate NUM_GENERATIONS responses
    #   3. Score each with production_reward()
    #   4. Compute the GRPO loss (reward-weighted policy gradient)
    #   5. Update the LoRA weights
    #   6. Repeat for all prompts x epochs
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=production_reward,
        train_dataset=prompts_ds,
        processing_class=tokenizer,
        callbacks=[reward_callback],
    )

    # Clear GPU cache before training starts
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    log.info(f"VRAM before training: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    log.info("Starting GRPO training...")

    # ══════════════════════════════════════════════════════════════
    # THIS IS WHERE THE 8-HOUR TRAINING HAPPENS
    # The model generates ~1,200 responses, each scored by the
    # Heuristic Oracle, and gradually learns to produce responses
    # that maximize the reward: safe, efficient, concise PyTorch
    # fixes with proper diagnostic reasoning.
    # ══════════════════════════════════════════════════════════════
    trainer.train()

    # ── Export the trained model as GGUF ──
    # GGUF is a portable model format that can be loaded by:
    #   - LM Studio (desktop app)
    #   - llama.cpp (command line)
    #   - llama-cpp-python (Python server — used in our HF Space)
    #   - Ollama
    # Q4_K_M quantization keeps the model at ~4.9GB while retaining
    # >95% of the trained quality.
    log.info("Training complete. Saving GGUF...")
    gguf_dir = "/tmp/grpo_gguf_export"
    os.makedirs(gguf_dir, exist_ok=True)
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")

    os.makedirs("/kaggle/working/final-model", exist_ok=True)
    for file in glob.glob("/tmp/grpo_gguf_export/*.gguf"):
        dest = f"/kaggle/working/final-model/{os.path.basename(file)}"
        shutil.move(file, dest)
        log.info(f"GGUF saved -> {dest}")

    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  Evaluations: {MAX_PROMPTS * NUM_GENERATIONS * GRPO_EPOCHS:,}")
    log.info("  Files to download:")
    log.info("    /kaggle/working/final-model/  <- trained GGUF model")
    log.info("    /kaggle/working/reward_log.jsonl  <- per-step reward history")
    log.info("    /kaggle/working/training_summary.json  <- final statistics")
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════
# SECTION 9: ENTRY POINT
# ══════════════════════════════════════════════════════════════════
# Execution order:
#   1. Print GPU info
#   2. verify_reward_alignment() — sanity-check the reward function
#      (aborts if any test fails, before wasting GPU hours)
#   3. run_stage_2_grpo() — the full 8-hour GRPO training run
#
# After training completes, download these 3 files from Kaggle:
#   - final-model/Llama-3.1-8B-Instruct.Q4_K_M.gguf  (~4.9GB)
#   - reward_log.jsonl  (training proof)
#   - training_summary.json  (final metrics)

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  SWARM-OS KAGGLE TRAINING PIPELINE")
    log.info(f"  GPU:  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    log.info("=" * 60)

    verify_reward_alignment()
    run_stage_2_grpo()
