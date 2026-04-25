# Swarm-OS AI: Maximum Performance Pipeline

**Objective:** Present the judges with a highly robust, end-to-end RL training pipeline proving that Swarm-OS is not just a UI, but a self-improving agent orchestration system.

---

## 1. The Supervised Fine-Tuning (SFT) Baseline
We began by creating a baseline fine-tune using **Google's Kaggle T4 GPUs** (`train_kaggle.py`). 
We didn't just guess what the model should output; we used Swarm-OS's built-in **Docker Sandbox** as a data telemetry engine. 
- The backend generated heavily constrained Docker workloads (e.g., 500MB VRAM limit).
- Snorkel auto-labeled the outcome of every agent's code.
- We passed the 46 most highly successful "Golden Examples" through a Low-Rank Adaptation (LoRA) procedure using **Unsloth** for 2x faster memory-efficient training.
- We then directly exported the quantized `.gguf` file to plug back into LM Studio, replacing the base Llama 3.1 8B with our custom Docker-aware agent.

## 2. The Scale-Up Plan (GRPO Reinforcement Learning)
To move from a 40% zero-shot success rate on unexpected edge-cases to a 95% pass rate, our pipeline utilizes **HuggingFace TRL (Transformer Reinforcement Learning)** using the **GRPO (Generative Reward Policy Optimization)** algorithm on an A100.

### The 120-Scenario Curriculum
We built an expansive curriculum script (`colab_grpo_training.py`) containing 120 highly-targeted edge cases divided into six distinct categories:
1. **Memory Errors (OOM)**: Single layer overflows, batch size limits, Adam state optimization.
2. **Transformer Specific**: Flash Attention fixes, Vision Transformer patching.
3. **Network & Distributed**: TCP timeouts, NVLink saturation, cluster dropouts.
4. **FinOps & SLA**: Agent disagreement over cost, Spot instance preemptions.
5. **Database & Infrastructure**: SQL deadlocks preventing pipeline ingestion, schema drift.
6. **Adversarial Edge Cases**: Contradictory rules (e.g., zero budget, high limits).

### Physical Oracle Reward Signal
Instead of using a generic LLM to act as the "Judge" (LLM-as-a-Judge), we built a **Physical Oracle**. The Colab/A100 training job literally calls back to our local Swarm-OS backend via an **Ngrok Tunnel**. 
- The model generates Python code.
- It beams the code to the Swarm-OS `/api/code/submit` endpoint.
- Swarm-OS spins up the isolated Docker container, attempts to execute the code, and measures the exact execution physics (VRAM used, crashes).
- That exact numeric physics score is bounced back to Colab as the Reinforcement Learning Reward Signal.

## 3. Technology Stack Used
- **Environment**: Kaggle (T4) for active SFT; Google Colab Pro (A100) for GRPO scaling.
- **Frameworks**: `unsloth` for LoRA gradient patching, `trl` for GRPO/SFT routing.
- **Data Engine**: Snorkel for heuristic weak-supervision and data extraction.
- **Simulation**: Docker + PyTorch for the physical reward boundary logic.
- **Inference**: `.gguf` compiled output for localized LM Studio integration on-device.
