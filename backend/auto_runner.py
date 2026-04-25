"""
Swarm-OS Auto Dataset Runner
=============================
Sends orchestration prompts to the backend to generate swarm_dataset.jsonl records.
Each successful orchestration produces 1 rejected (OOM) + 1 chosen (PASS) = 1 DPO pair.

Usage:
    1. Start the backend:  cd d:\op\backend && python main.py
    2. Run this script:    python auto_runner.py
    3. Check results:      type swarm_dataset.jsonl | find /c /v ""

Prerequisites:
    - Backend running on http://localhost:8000
    - Docker Desktop running with GPU passthrough
    - pip install requests
"""

import time
import requests
import json
import sys

# ── Configuration ──
BACKEND_URL = "http://localhost:8000"
RUNS_PER_PROMPT = 1  # 1 is optimal — diversity > repetition
DELAY_BETWEEN_RUNS = 5  # seconds between orchestrations

# ── 22 Diverse Prompts (covering all scenario types) ──
PROMPTS = [
    # --- Tier 1: Mixed Precision / Base Optimization (Autocast) ---
    "Deploy a standard feedforward neural network for our tabular dataset. Lock the VRAM exactly at 500m. The activations are spilling over the limit.",
    "I'm training a simple linear regression model on financial data, but PyTorch keeps crashing at the 500MB Docker limit. Fix the memory footprint.",
    "We need a multilayer perceptron for customer churn prediction. The memory is slightly over our 500m budget. Please optimize it.",
    "Our basic recommendation engine is throwing CUDA OutOfMemory errors. Enforce a 500MB hardware limit and use precision tricks to fit it.",
    "Can you run this dense layer architecture? It's hitting 600MB, but the container only allows 500MB. Lower the precision safely.",

    # --- Tier 2: Checkpointing / CNNs ---
    "Our ResNet computer vision model is crashing. Enforce a 500m VRAM limit and fix the Out-Of-Memory error caused by the massive convolutional feature maps.",
    "We are processing 4K medical X-rays with a CNN. The activations are huge. Lock VRAM to 500m and use gradient checkpointing to save us.",
    "Deploy a U-Net for satellite imagery segmentation. The memory spikes during the forward pass are killing the container. 500m limit.",
    "I've got a deep convolutional network for self-driving car telemetry. It needs to run on edge devices with max 500MB VRAM. Make it fit.",
    "The 3D CNN for video action recognition is way too heavy. We only have 500MB of GPU space. Trade compute speed for memory if you have to.",

    # --- Tier 3: Flash Attention / Transformers ---
    "Deploy an NLP sequence model. The attention mechanism is hitting quadratic memory scaling and blowing past our 500m Docker limit. Fix it.",
    "We are running Llama-3 locally but the context window of 8k tokens is destroying our VRAM. Implement a memory-efficient attention trick for a 500m limit.",
    "Our customer support chatbot uses a massive Transformer. Set the Docker limit to 500m and eliminate the N-squared attention memory bottleneck.",
    "Deploy a Vision Transformer (ViT) for image classification. The self-attention layers are OOMing. 500MB hard limit.",
    "The sequence-to-sequence translation model is failing on long documents. Force a 500m VRAM cap and optimize the attention layers natively.",

    # --- Edge Cases: FinOps & SLA (Testing the Manager Agent) ---
    "URGENT: Deploy the cluster immediately! We only have $1.50 left in our AWS budget for this month, and the client SLA requires this to be running in under 45 seconds.",
    "We are completely out of cloud credits ($0.00 left). I need a massive model deployed with a 500m limit right now.",
    "The client contract says we fail if deployment takes longer than 10 seconds. Spin up the 500MB sandbox and optimize the model instantly.",

    # --- Edge Cases: Cross-Domain Distractions (Testing DB_Admin) ---
    "The primary PostgreSQL database is hitting a deadlock and starving the PyTorch workers. Spawn a DB Admin to clear the queue, then lock the PyTorch container to 500MB and fix the OOM.",
    "Our Redis cache just dropped all connections. Fix the caching layer, then deploy the PyTorch model with a strict 500m VRAM lock and mixed precision.",

    # --- Schema Drift (Patronus AI bonus bounty) ---
    "Our telemetry pipeline just broke. The JSON schema changed from server_id and status to a nested format mid-flight. Fix the ingestion script and enforce 500MB on the PyTorch worker.",
    "The monitoring API silently updated its response format. All our parsers are returning None. Detect the schema drift and rewrite the data pipeline.",
]


def check_backend():
    """Verify the backend is running."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/telemetry", timeout=5)
        r.raise_for_status()
        print("[OK] Backend is running")
        return True
    except Exception as e:
        print(f"[FAIL] Backend unreachable: {e}")
        print("  Start it with: cd d:\\op\\backend && python main.py")
        return False


def run_orchestration(prompt: str, run_index: int, total: int):
    """Send a single orchestration prompt to the backend."""
    print(f"\n{'='*60}")
    print(f"  Run {run_index}/{total}")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"{'='*60}")

    try:
        r = requests.post(
            f"{BACKEND_URL}/api/orchestrate",
            json={"prompt": prompt},
            timeout=180,  # 3 minutes max per orchestration
        )
        result = r.json()
        status = result.get("status", "unknown")
        baseline = result.get("baseline_execution", {})
        optimized = result.get("fsdp_execution", {})

        baseline_status = baseline.get("status", "N/A") if isinstance(baseline, dict) else "N/A"
        optimized_status = optimized.get("status", "N/A") if isinstance(optimized, dict) else "N/A"

        print(f"  Status:    {status}")
        print(f"  Baseline:  {baseline_status}")
        print(f"  Optimized: {optimized_status}")

        if optimized_status == "PASS":
            print(f"  [PASS] SUCCESS -- 1 chosen + 1 rejected pair generated")
        else:
            print(f"  [WARN] Optimized did not PASS -- only rejected record generated")

        return {
            "status": status,
            "baseline": baseline_status,
            "optimized": optimized_status,
            "success": optimized_status == "PASS",
        }

    except requests.exceptions.Timeout:
        print(f"  [FAIL] TIMEOUT -- orchestration took too long")
        return {"status": "timeout", "success": False}
    except Exception as e:
        print(f"  [FAIL] ERROR: {e}")
        return {"status": "error", "success": False}


def main():
    print("=" * 60)
    print("  SWARM-OS AUTO DATASET RUNNER")
    print(f"  Prompts: {len(PROMPTS)}")
    print(f"  Runs per prompt: {RUNS_PER_PROMPT}")
    print(f"  Total orchestrations: {len(PROMPTS) * RUNS_PER_PROMPT}")
    print("=" * 60)

    if not check_backend():
        sys.exit(1)

    total = len(PROMPTS) * RUNS_PER_PROMPT
    results = []
    successes = 0
    failures = 0

    run_index = 0
    for prompt in PROMPTS:
        for _ in range(RUNS_PER_PROMPT):
            run_index += 1
            result = run_orchestration(prompt, run_index, total)
            results.append(result)

            if result["success"]:
                successes += 1
            else:
                failures += 1

            # Progress report
            print(f"\n  Progress: {run_index}/{total} | PASS: {successes} | FAIL/WARN: {failures}")

            if run_index < total:
                print(f"  Waiting {DELAY_BETWEEN_RUNS}s before next run...")
                time.sleep(DELAY_BETWEEN_RUNS)

    # Final report
    print("\n" + "=" * 60)
    print("  DATASET COLLECTION COMPLETE")
    print("=" * 60)
    print(f"  Total runs:      {total}")
    print(f"  Successful:      {successes} (generated DPO pairs)")
    print(f"  Failed:          {failures} (only rejected records)")
    print(f"  Success rate:    {successes/max(1,total)*100:.0f}%")
    print(f"  Expected records: ~{successes * 2 + failures} new records in swarm_dataset.jsonl")
    print()
    print("  Next steps:")
    print("    1. cd d:\\op\\dataset && python build_training_splits.py")
    print("    2. cd d:\\op\\dataset && python validate_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
