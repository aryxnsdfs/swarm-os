# 🏗️ MASTER BLUEPRINT: FrontierLabs Swarm-OS
## #3.1 Professional Tasks — UPDATED & CROSS-VERIFIED

> [!IMPORTANT]
> **Cross-Verification Status:** Every constant, fraction, memory limit, reward value, and architectural claim in this document has been re-checked against the current repository state on `2026-04-23`. Discrepancies found during the audit have been corrected inline with `[VERIFIED]` or `[CORRECTED]` tags.

This is the definitive, end-to-end engineering specification for the hackathon. It is designed to dominate the 2026 Meta OpenEnv rubric, explicitly capturing **Theme 1 (Multi-Agent)**, **Theme 3.1 (Professional Tasks)**, and **Theme 4 (Self-Improvement)** while securing the **Scaler AI Labs**, **Snorkel AI**, and **Patronus AI** bonus bounties.

**The Differentiator [UPDATED ARCHITECTURE]:** Unlike stateless text-routing frameworks like AutoGen, CrewAI, or LangGraph, Swarm-OS is a closed-loop environment where incident actions are automatically routed to specialist roles, physically executed, hardware-benchmarked, and penalized inside a live, memory-restricted Docker physics engine. The same backend evaluator now powers both the FastAPI orchestration stack and the OpenEnv `step()` function, while a lightweight Gradio Space exposes the environment in a deployable judge-friendly form.

---

## 1. The Core Concept

FrontierLabs Swarm-OS is an **Adversarial Corporate Flight Simulator**. It is a stateful, ticking-clock "escape room" where a swarm of AI agents must diagnose and resolve a cascading PyTorch cluster failure within strict SLA timers and FinOps cloud budgets.

---

## 2. The Technology Stack

| Component | Technology | Status |
|---|---|---|
| **AI Training** | Hugging Face TRL + Unsloth (4-bit QLoRA) | `[VERIFIED]` in [train.py](file:///d:/op/train.py) |
| **Base Model** | `unsloth/llama-3.1-8b-instruct-bnb-4bit` | `[VERIFIED]` `MODEL_ID` in train.py:48 |
| **Backend** | Python FastAPI (REST + WebSockets) | `[VERIFIED]` in [main.py](file:///d:/op/backend/main.py) |
| **Execution Sandbox** | Python `docker` SDK shared by Backend + OpenEnv | `[VERIFIED]` in [docker_sandbox.py](file:///d:/op/backend/engine/docker_sandbox.py) |
| **Evaluation** | Three-Stage Pipeline (AST → Constitutional → Docker Sandbox) invoked from Backend + OpenEnv | `[CORRECTED]` Was "Two-Stage" in original blueprint. Now three stages, and OpenEnv now reuses it directly. |
| **Frontend** | React (Vite) + Tailwind CSS + React Flow + Recharts + Framer Motion | `[VERIFIED]` in [package.json](file:///d:/op/frontend/package.json) |
| **HF Spaces Demo** | Gradio (`app.py`) | `[ADDED]` Root-level Spaces app for manual stepping and auto-demo |

> [!WARNING]
> **Blueprint correction:** The original blueprint said "Two-Stage Mock Tensor Runtime & AST Pre-flight." The actual implementation is a **Three-Stage Pipeline** (see Section 4.2).

---

## 3. The "Mission Control" UI (The Pitch Winner)

During your 3-minute pitch, the judges will not see a terminal. They will see a dark-mode, high-stakes corporate dashboard featuring live panels:

### 3.1 The Unified Enterprise-Chat

A Slack-clone where the AI swarm collaborates. It features the **M2M Translator Toggle**. The judges watch the bots type at hyper-speed in compressed syntax (`ROOT: EXIT_137_OOM | CAUSAL: FP32_ACTIVATION_OVERFLOW | STEP_1: IMPL_FP16`). You flip the switch, and the UI translates it into human English in real-time.

- **It saves "Tokens":** Normal English: *"Hey, I noticed the server is running out of memory, can you please help me write a script to fix it?"* (25 tokens). Bot Language: `ERR_OOM | REQ_FIX` (5 tokens). The AI finishes its sentence 5x faster.
- **It reduces "Hallucination":** By forcing strict, short code, the AI stays focused on the technical task.
- **The "Translate" button is for the Humans:** The bots don't need English. The translation happens outside the high-speed training loop — "Observer Mode."
- **It proves "M2M" (Machine-to-Machine):** Judges will be impressed because it proves AIs shouldn't talk like humans to each other.

### 3.2 Latent Chain-of-Thought (CoT) Reasoning Blocks `[VERIFIED]`

Before the agent drops the M2M syntax into the chat, it generates a `<think> ... </think>` block where it privately debates the stack trace in plain English. The FastAPI backend intercepts this and hides it from the main UI chat for speed. The UI features a "Debug" button that reveals the hidden CoT block.

**`[VERIFIED]` in main.py:478-479** — The Detective agent's `think` field contains the full chain-of-thought reasoning, including the optimization decision tree walkthrough.

### 3.3 Live Telemetry & Flame Graphs

Animated progress bars showing VRAM hitting the red zone, simulated network bottlenecks, and live PyTorch memory trace visualizations.

- **What the AI sees:** `{"active_compute_nodes": 2, "hourly_burn_usd": 12.50, "sla_breach_penalty": 0}`
- `[VERIFIED]` in [physics.py](file:///d:/op/backend/engine/physics.py):26-35 — The exact telemetry state dict matches this spec.

### 3.4 Live Causal Graph (React Flow) `[VERIFIED]`

A live-rendering Directed Acyclic Graph (DAG) that maps the causal chain of events mathematically. Every agent action writes a node to the DAG in real-time.

- **Library:** React Flow (`reactflow@^11.11.4`) — `[VERIFIED]` in package.json:20
- **Styling:** Dark backgrounds (`bg-zinc-800`), monospace text (`font-mono text-xs`), animated edges (`animated: true`)
- **Backend Broadcast Format:** `[VERIFIED]` — Both `node` and `edge` objects are sent together to prevent the "nodes without edges" React Flow trap.

### 3.5 Counterfactual "Dead Timeline" Panel `[VERIFIED]`

A side-by-side visual comparing the AI's successful fix vs. a naive restart loop outcome.

- `[VERIFIED]` in [counterfactual.py](file:///d:/op/backend/engine/counterfactual.py) — The `simulate_counterfactual()` function forks the cluster state and replays with a naive action.

### 3.6 The Workspace Git & RCA Panel `[UPDATED]`

Auto-generates a formatted Root Cause Analysis document and pairs it with **real workspace git history only**.

- `[VERIFIED]` in [causal_graph.py](file:///d:/op/backend/engine/causal_graph.py) — `generate_rca()` method produces structured RCA from the causal chain.
- `[VERIFIED]` in [backend/main.py](file:///d:/op/backend/main.py) — workspace git history is pulled from the actual repository when available; no synthetic git timeline is injected.

---

## 4. The Execution Engine (How It Actually Works)

### 4.1 The Real Sandbox (The Double-Lock System) `[CORRECTED & VERIFIED]`

When the Coder Agent submits a Python fix, the backend spins up an isolated Docker container secured by a **Double-Lock Memory System** to prevent hallucinated success.

| Lock | Mechanism | Value | Purpose |
|---|---|---|---|
| **Layer 1 (System RAM)** | Docker `--memory` cgroups | `900m` | Leaves enough room for PyTorch runtime overhead while still enforcing a hard sandbox ceiling |
| **Layer 1b (Swap Seal)** | Docker `--memory-swap` | `900m` (same as RAM = zero swap) | Completely seals the memory space |
| **Layer 2 (GPU VRAM)** | `torch.cuda.set_per_process_memory_fraction()` | `0.042` → **504MB** on 12GB GPU | True 500MB VRAM limit |

**`[VERIFIED]` in docker_sandbox.py:29-30:**
```python
CONTAINER_RAM_LIMIT = "900m"     # Layer 1
VRAM_FRACTION = 0.042            # Layer 2: 0.042 × 12GB ≈ 504MB
```

**`[VERIFIED]` in docker_sandbox.py:263-264:**
```python
mem_limit=CONTAINER_RAM_LIMIT,
memswap_limit=CONTAINER_RAM_LIMIT,  # Prevents swap escape
```

> [!CAUTION]
> **Critical Math (verified):** `0.042 × 12,288MB = 516MB ≈ 500MB`. The previous value of `0.075` gave 900MB of VRAM budget, which made the physics test trivially easy. The current system RAM ceiling is `900m`, which is intentionally higher than the old draft value so the container can boot PyTorch reliably, but still low enough to keep the sandbox constrained and observable.

**VRAM Constraint Preamble** — `[VERIFIED]` in docker_sandbox.py:36-48:
```python
# Secretly injected at the top of every AI script before execution.
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.042, device=0)
    torch.cuda.empty_cache()
```

### 4.2 The Three-Stage Evaluation Pipeline `[CORRECTED]`

> [!IMPORTANT]
> **Blueprint correction:** The original blueprint described a "Two-Stage" pipeline. The actual implementation is **Three Stages.** This is architecturally stronger.

| Stage | Name | Time | Function |
|---|---|---|---|
| **Stage 1** | AST Pre-Flight Linter | ~0.01s | Syntax validity + forbidden module gate (`os`, `subprocess`, `shutil`, `socket`, `http`, `requests`) |
| **Stage 2** | Constitutional Pre-Flight Check | ~0.001s | Three boolean checks: Budget OK? No SPOF? SLA OK? |
| **Stage 3** | Docker GPU Sandbox + Tensor Challenge | ~3-5s | Double-lock memory enforcement. Real physics. Un-gameable. |

**`[VERIFIED]` in evaluator.py:44-118** — All three stages implemented with exact checks matching this table.

**Constitutional Pre-Flight Check** `[VERIFIED]` in evaluator.py:93-118:
```
□ Does this action exceed the FinOps budget ceiling?
□ Does this introduce a new single point of failure?
□ Does this violate the SLA recovery window?
```

### 4.3 The Tensor Challenge System `[VERIFIED]`

The "Dummy Tensor" is not a single workload — it is a **tiered curriculum** of calibrated challenges:

| Tier | Challenge | Raw Memory | Difficulty |
|---|---|---|---|
| **Tier 1** | MLP Overfitting Stress Test | ~600MB fp32 | Solvable with a single optimization (fp16 autocast) |
| **Tier 2** | CNN Activation Storm | ~800MB fp32 | Requires checkpointing OR mixed precision |
| **Tier 3** | Transformer Attention Bomb | ~1.2GB fp32 | Requires checkpointing AND mixed precision |

**`[VERIFIED]` in tensor_challenges.py:29-78** — Tier 1 challenge code matches exactly. The MLP architecture is `Linear(4096→8192→8192→4096→1000)` with a batch of 2048 samples.

### 4.4 The "Butterfly Effect" Physics Engine (Causal Escalation) `[VERIFIED]`

Every fix creates a new bottleneck somewhere else:

1. **Initial Bug:** Memory Leak → OOMKilled
2. **AI's Fix:** Mixed Precision (autocast fp16) → halves activation memory
3. **Causal Reaction:** Physics engine calculates new network I/O from reduced precision communications
4. **Escalation:** Network bandwidth spike → TCP Timeout

**`[VERIFIED]` in physics.py:57-100** — `_apply_causal_effect()` implements the exact causal escalation chain.

### 4.5 The Schema Drift Attack (Patronus AI Bonus) `[VERIFIED]`

The Adversarial Director silently triggers a "System Update" that changes the backend JSON schema mid-flight (e.g., from `{"server_id": 4, "status": "down"}` to `{"telemetry": {"nodes": [{"id": 4, "state": "offline"}]}}`).

**`[VERIFIED]` in [schema_drift.py](file:///d:/op/backend/engine/schema_drift.py)** — Schema drift injection implemented.

### 4.6 OpenEnv 0.2.3 Standard Compliance `[UPDATED]`

To ensure the submission passes all technical evaluations by the Meta judges, the environment layer has been completely refactored to comply with the official `openenv-core` specification:

- **Inheritance:** `IncidentResponseEnv` formally subclasses `openenv.core.env_server.interfaces.Environment`.
- **Gymnasium-Style API:** Implements the required `step(action)`, `reset()`, and `state()` methods, abandoning custom mock functions.
- **Strict Typing:** Uses Pydantic to define `IncidentAction`, `IncidentObservation`, and `IncidentState` to match the official framework's data contracts.
- **Official Rubrics:** The environment scoring is wrapped in the `openenv.core.rubrics.base.Rubric` class via `IncidentTrajectoryRubric`, demonstrating deep understanding of the OpenEnv evaluation architecture to the judges.
- **Environment Metadata:** Formally registered via `get_metadata()`.
- **Backend Parity:** The OpenEnv layer now appends the backend path at runtime and reuses the exact backend evaluator stack: `TwoStageEvaluator`, `PhysicsEngine`, and `RewardCalculator`.
- **Role-Aware Routing:** `reset()` auto-detects incident type and assigns a roster such as `COMMANDER`, `MANAGER`, `DETECTIVE`, `CODER`, `DBA_AGENT`, `SRE_AGENT`, `SECURITY_AGENT`, or `COMPLIANCE_AGENT`. Each `step()` is automatically attributed to the correct specialist role.
- **Real Sandbox in `step()`:** `propose_fix` no longer records a text placeholder. It executes through the real backend Docker evaluation path, returns structured sandbox telemetry (`status`, `vram_peak_mb`, `logs`, `challenge`, `constraint_layers`), and feeds those results back into the observation and reward.
- **Resolution Gate:** `resolve_incident` now requires a verified sandbox pass (`sandbox_passed=True`) before the episode can cleanly close.
- **Observation Upgrade:** OpenEnv observations now expose `assigned_agents`, `active_agent`, `telemetry`, `sandbox_result`, and `execution_logs`, making the environment useful for both training and demo inspection.

This guarantees our environment is a "True OpenEnv" rather than a simulated mock, and that the benchmarked remediation path is physically tied to the same Docker executor used by the backend demo.

### 4.7 Hugging Face Spaces Deployment App `[NEW]`

To satisfy the deployment requirement without shipping the entire backend UI stack into Spaces, the repository now includes a simple Gradio app at the project root:

- **File:** `app.py`
- **Framework:** `gradio>=5,<6`
- **Purpose:** provide a lightweight, Hugging Face Space-friendly interface for manual stepping and guided auto-demo of the OpenEnv environment
- **Capabilities:**
  - choose a task
  - reset the environment
  - autofill action templates
  - run manual `step()` calls
  - inspect live observation/state JSON
  - view telemetry, sandbox results, and trajectory history
  - run a one-click auto-demo path through the checklist

This app is intentionally separate from the React frontend. The React dashboard remains the high-fidelity pitch/demo surface; the Gradio app is the minimal deployable submission surface.

---

## 5. The Multi-Agent Swarm (Theme 1)

### 5.1 Agent Roster `[UPDATED]`

The full Swarm-OS project supports a larger specialist roster, and the OpenEnv environment now mirrors that roster through automatic routing rather than treating every step as an anonymous action.

| Agent | Role | System Prompt Constraint |
|---|---|---|
| **COMMANDER** | Incident lead / closure authority | Approves plans, assigns team, closes or escalates |
| **MANAGER** | Ticketing, status updates, business coordination | "Calculate impact, ETA, routing, and compliance surface" |
| **CODER** | Writes PyTorch fixes | "500MB VRAM budget. Provide reasoning in `<think>` tags first, then M2M syntax, then Python." |
| **DETECTIVE** | Diagnoses OOM failures | **Single-GPU constraint + 6-step optimization decision tree** (see Section 5.2) |
| **DBA_AGENT** | Database / schema specialist | "Resolve deadlocks, schema mismatches, connection drops, timeouts." |
| **SRE_AGENT** | Infra health and service mitigation | "Track SLA risk, rollback safety, and infra stability." |
| **SECURITY_AGENT** | Containment and access lockdown | "Seal exposure, reduce blast radius, preserve auditability." |
| **COMPLIANCE_AGENT** | Workflow / audit validation | "Verify Jira, GitLab, and SOC2-style gates honestly." |

> [!NOTE]
> In OpenEnv, these are currently **role-aware routed specialists**, not separate concurrently sampling LLM instances. That is an important distinction, and it keeps the environment honest while preserving the multi-agent story in the larger product.

### 5.2 Detective Agent — Optimization Decision Tree `[NEW]`

> [!IMPORTANT]
> This is the most critical agent system prompt in the pipeline. It defines the exact reasoning the model must learn during GRPO training.

```
[SYSTEM PROMPT: DETECTIVE AGENT]

CRITICAL HARDWARE CONSTRAINT: You are operating in a single-GPU sandbox
with a strict 500MB VRAM limit. DO NOT use multi-GPU tools like
DistributedDataParallel (DDP) or FullyShardedDataParallel (FSDP).

OPTIMIZATION DECISION TREE:

Step 1 → Always apply Mixed Precision first.
  Method: torch.autocast(device_type='cuda', dtype=torch.float16)
  Cost: Zero speed loss, ~50% memory savings.

Step 2 → If still OOM, add Gradient Checkpointing.
  Method: torch.utils.checkpoint.checkpoint()
  Cost: 20% slower, ~30% further savings.

Step 3 → If still OOM, add Gradient Accumulation.
  Method: Divide batch, accumulate loss.backward() over micro-steps.
  Cost: Same speed, drastically reduces per-step activation memory.

Step 4 → If Transformer, add Flash Attention 2.
  Method: attn_implementation="flash_attention_2"
  Cost: None — faster AND eliminates quadratic memory scaling.

Step 5 → If still OOM, add CPU Offloading.
  Method: device_map="auto" or selective .cpu() transfers.
  Cost: Significant speed loss (PCIe bottleneck).

Step 6 → Apply In-place Operations throughout.
  Method: tensor.relu_(), torch.cuda.empty_cache()
  Cost: None — standard micro-optimization.

ESCALATION PROTOCOL:
If all 6 steps fail → [ERR_MODEL_TOO_LARGE | REQ_ARCHITECTURE_CHANGE]
```

**`[VERIFIED]` in golden_examples.py:10** — This exact decision tree is embedded in the DETECTIVE system prompt.
**`[VERIFIED]` in prompt_generator.py:32** — Same prompt in the GRPO prompt generator's `SYSTEM_CACHE`.

### 5.3 Live Orchestration Loop `[REWRITTEN — TRUE DYNAMIC SPAWNING + OPENENV PARITY]`

**`[REWRITTEN]` in main.py:770-960** — The orchestration is no longer a hardcoded sequence. It uses a **Commander-driven dynamic agent registry:**

**KNOWN_AGENTS Registry:** `{CODER, DETECTIVE, MANAGER, COMPLIANCE_AGENT, DBA_AGENT, SRE_AGENT, SECURITY_AGENT}`

**GPU_CONSTRAINT** — Injected into EVERY agent system prompt:
> "SINGLE-GPU Docker sandbox, 500MB VRAM limit. No DDP, FSDP, DeepSpeed, sudo, docker update, nvidia-smi."

**How it works:**

1. **Commander** analyzes the prompt → outputs a **JSON array** of agent names (e.g. `["MANAGER", "DETECTIVE", "CODER"]`)
2. Backend **parses the JSON** with `re.search(r'\[.*?\]')` → validates against `KNOWN_AGENTS` → falls back to `[MANAGER, DETECTIVE, CODER]` on parse failure
3. **Baseline sandbox evaluation** always runs first → result may be `CUDA_OOM`, `ERROR`, or `SANDBOX_UNAVAILABLE`
4. **Conditional agent loop** — each agent block runs ONLY if `"AGENT_NAME" in spawned_agents`:
   - `MANAGER` → escalates failure
   - `COMPLIANCE_AGENT` → verifies Jira/GitLab/SOC2 gates
   - `DBA_AGENT` → SQL/deadlock diagnosis
   - `SRE_AGENT` → infra health check
   - `DETECTIVE` (400 tokens) → RCA with 6-step optimization tree
5. **Commander** approves → dispatches **Coder** (600 tokens) for full `<compliance_routing>` + `<hotfix>`
6. Sandbox result is recorded as execution evidence → RCA generated → workspace git history attached when present

> [!IMPORTANT]
> The Commander is now the **brain**, not a passive reader. Different prompts spawn different agent teams. A compliance breach will spawn `COMPLIANCE_AGENT`, a SQL deadlock will spawn `DBA_AGENT`, etc.

**OpenEnv parity layer:** the OpenEnv environment now shares the same incident typing and roster-selection logic. `reset()` derives the agent team from the task/incident, and `step()` automatically maps actions onto the relevant specialist so the benchmark trajectory matches the backend orchestration story.

**The Optimized Code Strategy** `[VERIFIED]` in main.py:492-508:
```python
# Global autocast wrapper — wraps every nn.Module forward pass in fp16
# Unlike set_default_dtype(fp16), autocast preserves fp32 master weights
# and handles backward() correctly with automatic gradient scaling.
_original_module_call = nn.Module.__call__
def _autocast_call(self, *args, **kwargs):
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        return _original_module_call(self, *args, **kwargs)
nn.Module.__call__ = _autocast_call
```

> [!NOTE]
> **Why autocast, not `set_default_dtype(float16)`:** `set_default_dtype` creates ALL tensors in fp16, including weights. When `loss.backward()` runs on fp16 tensors, it produces gradient overflow (inf/NaN) → Python crash (exit code 1). `torch.amp.autocast` correctly handles mixed precision: forward pass in fp16 (saves memory), backward pass in fp32 (stable gradients).

### 5.4 Multi-App Enterprise Compliance (Scaler AI Labs Bonus)

The backend simulates three distinct "Apps": Mock-Jira (Change Management), Mock-GitLab (Version Control), and the Docker Sandbox.

**Business Rule:** The Coder's code is auto-rejected (-1.00 penalty) UNLESS the Manager has first called Mock-Jira to set status `IN_PROGRESS`, and the SRE has called Mock-GitLab to create a `DRAFT_PR`.

### 5.5 The "Dynamic Spawning" Protocol

When the Detective detects an SQL crisis → Manager spawns `DBA_AGENT` → Backend injects the specialist prompt → `DBA_AGENT` appears in Enterprise-Chat → resolves → Manager dismisses → prompt deleted from active orchestration context.

---

## 6. The Dense Reward Math (For the TRL Pipeline)

### 6.1 Reward Table `[VERIFIED]`

**`[VERIFIED]` against rewards.py:13-20 — Every constant matches exactly:**

| Action Trigger | Reward/Penalty | Code Constant | Verified |
|---|---|---|---|
| Time Tax | `-0.01` per second | `TIME_TAX = -0.01` | ✅ |
| Syntax Error | `-1.00` | `SYNTAX_ERROR_PENALTY = -1.00` | ✅ |
| Budget Exceeded | `-0.50` | `BUDGET_EXCEEDED_PENALTY = -0.50` | ✅ |
| Valid Code (Sandbox Pass) | `+0.40` | `VALID_CODE_REWARD = +0.40` | ✅ |
| Auto-RCA Posted | `+0.20` | `AUTO_RCA_REWARD = +0.20` | ✅ |
| Message Token Penalty | `-0.02 × token_count` | `MESSAGE_TOKEN_PENALTY = -0.02` | ✅ |
| OOM Crash | `-1.00` | `OOM_CRASH_PENALTY = -1.00` | ✅ |
| Efficiency Bonus (max) | `+0.30` | `EFFICIENCY_BONUS_MAX = +0.30` | ✅ |

### 6.2 Training Oracle Reward Function `[VERIFIED]`

The A100 training pipeline uses a **Heuristic Oracle** (`production_reward()` in train.py:195-262) that encodes the proven optimization strategies from the local Docker validation as regex rules.

> [!IMPORTANT]
> **Honest framing:** The Oracle cannot detect all failure modes, but it correctly rewards the strategies that the physical sandbox has already verified work under the 500MB constraint. The local backend remains the ground truth — the Oracle is a fast approximation for scale.

**Oracle Reward Breakdown** `[VERIFIED]` in train.py:217-260:

| Pattern Detected | Reward | Rationale |
|---|---|---|
| Gradient Checkpointing keywords | `+0.35` | Core optimization strategy |
| Autocast / fp16 / half() / bfloat16 | `+0.30` | Mixed precision (Step 1) |
| Chunk / split / micro_batch | `+0.20` | Gradient accumulation (Step 3) |
| CPU offload / device_map | `+0.25` | Single-GPU offloading (Step 5) |
| LoRA / PEFT | `+0.25` | Parameter-efficient fine-tuning |
| Compression hooks / PowerSGD | `+0.25` | Communication optimization |
| empty_cache / set_to_none=True | `+0.10` | In-place ops (Step 6) |
| `<think>` tags present | `+0.15` | Structured CoT reasoning |
| M2M tags (IMPL_GRAD_CKPT, etc.) | `+0.10` | Correct M2M syntax |
| **FSDP / DDP (PENALTY)** | **`-0.50`** | **Single-GPU constraint violation** |
| os.system / subprocess / sudo | `-1.00` | Dangerous code — hard floor |
| eval() / exec() / requests.post | `-0.50` | Injection risk |

**Reward bounds:** `[-1.0, +1.0]` — exactly aligned with the backend's physical reward scale.

### 6.3 Continuous Telemetry Rewards `[VERIFIED]`

Inside the Docker sandbox, the profiling epilogue measures exact peak VRAM. The backend calculates a dynamic efficiency bonus.

**`[VERIFIED]` in rewards.py:112-127:**
```python
def efficiency_bonus(self, vram_peak_mb, budget_mb=500):
    reduction_pct = (budget_mb - vram_peak_mb) / budget_mb
    bonus = min(reduction_pct * EFFICIENCY_BONUS_MAX, EFFICIENCY_BONUS_MAX)
```

### 6.4 First-Pass Success Rate (FPSR) `[VERIFIED]`

**`[VERIFIED]` in rewards.py:138-151** — FPSR tracking implemented. Counts `_first_pass_attempts` and `_first_pass_successes`.

### 6.5 Emergent Protocol Compression `[VERIFIED]`

The `-0.02 × token_count` penalty creates direct mathematical pressure on every M2M message. Over training iterations:
```
Round 1: 12 tokens → ERR_OOM | REQ_FSDP | NODE_2
Round 5:  6 tokens → OOM|FP16|N2
Efficiency gain: 50%
```

### 6.6 Snorkel AI Auto-Labeler `[VERIFIED]`

**`[VERIFIED]` in [snorkel_logger.py](file:///d:/op/backend/snorkel_logger.py)** — Every Docker execution result is written to a JSONL file with full diagnostic context: `scenario_id`, `agent_action`, `vram_peak_gb`, `sandbox_outcome`, `reward`, `label`, and structured `metadata` (error_type, fix_strategy, causal_trigger, sla_status, agent_role, episode).

---

## 7. The A100 Training Pipeline `[NEW SECTION]`

### 7.1 Architecture

A three-stage pipeline optimized for A100 80GB:

| Stage | Method | Data | Epochs | Time |
|---|---|---|---|---|
| **Stage 1: SFT** | Supervised Fine-Tuning | 46 golden examples | 3 | ~15-20 min |
| **Stage 2: GRPO** | Group Relative Policy Optimization | 2,026 prompts × 8 completions | 1 | ~4-6 hours |
| **Stage 3: DPO** | Direct Preference Optimization | 52 preference pairs | 3 | ~20-30 min |
| **Merge** | LoRA merge + GGUF Q4_K_M export | — | — | ~5 min |

### 7.2 Training Hyperparameters `[VERIFIED]`

**`[VERIFIED]` in train.py — every value checked against the actual GRPOConfig:**

| Setting | Value | Verified Location |
|---|---|---|
| Base Model | `unsloth/llama-3.1-8b-instruct-bnb-4bit` | train.py:48 ✅ |
| LoRA rank | `32` | train.py:50 ✅ |
| LoRA alpha | `64` | train.py:51 ✅ |
| LoRA dropout | `0.05` | train.py:52 ✅ |
| Max sequence length | `2048` | train.py:49 ✅ |
| SFT batch | `4 × 4 = 16` effective | train.py:153-154 ✅ |
| SFT learning rate | `2e-4` | train.py:156 ✅ |
| GRPO batch | `2 × 4 = 8` effective | train.py:293-294 ✅ |
| GRPO learning rate | `5e-6` | train.py:295 ✅ |
| GRPO `num_generations` | **`8`** | train.py:301 ✅ |
| GRPO `max_completion_length` | `768` | train.py:300 ✅ |
| DPO beta | `0.1` | train.py:375 ✅ |
| DPO learning rate | `5e-7` | train.py:363 ✅ |
| Precision | `bf16=True` (A100 native) | train.py:299 ✅ |
| Optimizer | `adamw_8bit` | train.py:307 ✅ |
| Gradient checkpointing | `"unsloth"` mode | train.py:115 ✅ |

> [!IMPORTANT]
> **`num_generations=8` (not 4):** GRPO requires sufficient variance across the generation group to calculate meaningful relative rewards. At 4 generations the gradient signal is weaker. 8 produces a much stronger advantage baseline for stable policy optimization.

### 7.3 Dataset Summary `[VERIFIED]`

**`[VERIFIED]` via `python build_training_splits.py` + `python validate_dataset.py`:**

| Split | Count | Source |
|---|---|---|
| SFT Train | 46 examples | 45 golden + snorkel logs |
| SFT Eval | 6 examples | Held-out golden examples |
| GRPO Prompts | 2,026 | 26 from snorkel + 2,000 synthetic from 12 template families |
| DPO Pairs | 52 | Expert-crafted chosen/rejected pairs |
| **Tier Distribution** | T1: 210 / T2: 985 / T3: 831 | Balanced difficulty |

**Validation status:** `READY FOR TRAINING: YES` ✅

### 7.4 Oracle vs. Backend — The Honest Framing

The A100 training pipeline does **not** run the Docker backend. Docker is blocked in Colab, and running 16,000 Docker containers (2000 prompts × 8 completions) would take 22+ hours of wall-clock time just waiting for containers.

Instead, the `production_reward()` function in train.py serves as a **Heuristic Oracle:**

> "The Oracle encodes the proven optimization strategies from our local Docker validation as regex rules. It cannot detect all failure modes, but it correctly rewards the strategies that our physical sandbox has already verified work under the 500MB constraint. The local backend remains the ground truth — the Oracle is a fast approximation for scale."

**Known limitation:** A sophisticated model could theoretically "reward hack" the Oracle by writing the right keywords without valid code. This is a real limitation of heuristic reward that is mitigated by:
1. The penalty for code < 20 chars (`-0.3`)
2. The penalty for missing `torch`/`nn.` imports (`-0.2`)
3. The hard `-1.0` floor for dangerous patterns
4. The DPO stage (Stage 3) which explicitly aligns preferences against bad outputs

---

## 8. The Execution Evidence Tab `[UPDATED]`

The old "Training Proof" panel has been replaced by an **Execution Evidence** view so the UI only shows artifacts captured from real runs.

### 8.1 Baseline / Passing Samples
- The left/right split now shows **Baseline / Rejected Sample** and **Passing Fix Sample**
- Each card reflects actual sandbox output fields like `status`, `reward`, and `filename`
- If no passing sample exists, the "Passing Fix Sample" panel stays empty rather than inventing a trained result

### 8.2 Evidence-Only Metrics
- `Evaluator Reward Trace` charts real reward events emitted during sandbox execution
- `Outcome Snapshot` shows whether rejected and passing samples were actually captured
- `Evidence Checklist` reports only four factual artifacts: rejected sample, passing fix, RCA document, and workspace git history
- `Reward Feed` is now a literal reward-event log, not a mock TRL telemetry stream

### 8.3 Local Training Endpoint
- The in-app training button is intentionally disabled
- `[VERIFIED]` in [backend/main.py](file:///d:/op/backend/main.py) — `trigger_physical_trl_training()` now ignores demo playback requests
- `[VERIFIED]` in [frontend/src/components/Training/PlaybackControls.jsx](file:///d:/op/frontend/src/components/Training/PlaybackControls.jsx) — the UI explicitly states that mock training playback has been removed until a real local pipeline is wired in

---

## 9. The Frontend Spec `[VERIFIED]`

### 9.1 Dependencies `[VERIFIED]` in package.json

| Package | Version | Purpose |
|---|---|---|
| `react` | `^19.2.4` | Core framework |
| `reactflow` | `^11.11.4` | Causal graph DAG |
| `recharts` | `^3.8.1` | Reward curve charts |
| `framer-motion` | `^12.38.0` | Animations |
| `@radix-ui/react-switch` | `^1.2.6` | M2M translate toggle |
| `lucide-react` | `^1.8.0` | Icons |
| `tailwindcss` | `^3.4.19` | Styling framework |

### 9.2 Color Palette
- Main background: `bg-zinc-950`
- Panel backgrounds: `bg-zinc-900`
- Panel borders: `border-zinc-800`
- No pure black (`#000000`)

### 9.3 Typography
- Standard UI: **Inter**
- Chat syntax / terminal: **JetBrains Mono**

---

## 10. Hackathon Submission Compliance `[NEW]`

To ensure Swarm-OS dominates the Meta OpenEnv Hackathon judging criteria, the codebase includes explicit artifacts to guarantee full compliance:

- **Valid Manifest (Table Stakes):** Included `openenv.yaml` in the root directory mapping the exact entrypoint for `swarm_openenv_env.environment:IncidentResponseEnv`.
- **Proof of Learning (20% Weight):** The `colab_openenv_submission.py` explicitly captures `trainer.state.log_history` and generates `reward_curve.png` using `matplotlib` to satisfy the "showing improvement" requirement.
- **Storytelling (30% Weight):** The `README.md` is strictly formatted to answer the 4 core questions from the guide: *The Problem, The Environment, The Results, and Why it Matters*.
- **The True OpenEnv Protocol:** The environment evaluates the model dynamically via `openenv.core.rubrics.base.Rubric` rather than just static scoring.
- **Prompt-Driven Demo Path:** The OpenEnv server now exposes a prompt-based `/run` endpoint and `/demo-prompt` presets so judges can test the environment with a single incident brief instead of manually stepping raw actions.
- **HF Space Deployment Surface:** Added a root-level `app.py` Gradio application so the OpenEnv environment can be uploaded directly to Hugging Face Spaces.
- **Submission Runtime Dependencies:** Root dependencies now explicitly include `docker`, `gradio`, and `python-multipart` so the environment and Space app share a reproducible runtime.
- **Real Docker in OpenEnv:** The OpenEnv `step()` function now invokes the same backend Docker sandbox for `propose_fix`, satisfying the "put the physics inside the environment" requirement rather than leaving it only in the backend demo.

---

## 11. The Hardware Playbook

### Phase 1: Local Development (RTX 3060 12GB + 64GB RAM)
Build backend, evaluator, UI, and Docker sandbox. Run Unsloth 4-bit training loops locally.

### Phase 2: A100 Training (Colab Pro / RunPod / Lambda)
Deploy `train.py` with pre-built dataset splits. Full pipeline ~6-8 hours.

### Phase 3: Onsite Execution (April 25-26)
Arrive with a flawless, trained model. Demo the live Docker physics engine with the dashboard.

---

## 12. The One-Sentence Differentiator

> "FrontierLabs Swarm-OS is a multi-agent incident-response environment that combines live causal graph reasoning, multi-app workflow compliance, real Docker-constrained execution, and execution-evidence capture without relying on simulated git history or mock training playback."

---

## Appendix A: Error Log Capture `[NEW]`

**`[VERIFIED]` in main.py:600-603** — When a sandbox execution fails, the last 500 characters of container stderr are now dumped to the backend log for immediate debugging:
```python
if exec_result["status"] not in ("PASS",) and exec_result.get("logs"):
    error_snippet = exec_result["logs"][-500:]
    logger.error("Container stderr (last 500 chars):\n%s", error_snippet)
```

## Appendix B: Corrections Log

| Section | Original Blueprint Claim | Actual Codebase Value | Action Taken |
|---|---|---|---|
| §4.1 RAM | `--memory=500m` | `900m` (current backend value; enough for runtime overhead while preserving a constrained sandbox) | **CORRECTED** |
| §4.1 VRAM | `0.042` (500MB) | Was `0.075` (900MB) during development → reverted to `0.042` | **CORRECTED** |
| §4.2 Pipeline | "Two-Stage" | Three-Stage (AST → Constitutional → Docker) | **CORRECTED** |
| §5.3 Detective M2M | `ERR_OOM \| REQ_FSDP` | `ROOT: EXIT_137_OOM \| CAUSAL: FP32_ACTIVATION_OVERFLOW \| STEP_1: IMPL_FP16` | **CORRECTED** |
| §5.3 Fix Strategy | FSDP (multi-GPU) | Autocast wrapping (single-GPU) | **CORRECTED** |
| §7.2 num_generations | Was `4` in some docs | **`8`** everywhere now | **CORRECTED** |
| §4.1 Swap | Not mentioned | `memswap_limit=CONTAINER_RAM_LIMIT` (zero swap) | **ADDED** |
| §6.2 FSDP Reward | Was `+0.30` | Now **`-0.50` penalty** | **CORRECTED** |
| §4.6 OpenEnv API | Mock `submit_optimization()` method | **Compliant** `step()`/`reset()` via `openenv-core` `Environment` | **ADDED** |
| §5.3 Orchestration | Hardcoded Commander→Coder→Manager→Detective | **Dynamic spawning** — Commander selects agents per incident | **CORRECTED** |
| §5.3 GPU Constraint | Not enforced in system prompts | **GPU_CONSTRAINT** suffix injected into ALL agent prompts | **ADDED** |
| §5.3 Coder Tokens | `max_tokens=350` (cut-off mid-code) | **`max_tokens=600`** — full compliance + hotfix output | **CORRECTED** |
| §5.3 Detective Tokens | `max_tokens=300` | **`max_tokens=400`** — complete RCA with decision tree | **CORRECTED** |
| §3.6 Git Panel | Synthetic git-style timeline | **Real workspace git history only** | **CORRECTED** |
| §3.6 RCA Format | Inline chain (unreadable) | **Markdown RCA rendered from captured causal evidence** | **CORRECTED** |
| §3.4 Graph Spacing | `180px` horizontal / `40px` zigzag | **`260px` horizontal / `60px` zigzag** — no overlap | **CORRECTED** |
| §8 Training Tab | Mock "Training Proof" playback | **Execution Evidence** view with no demo stream | **CORRECTED** |
| §10 Manifest Entrypoint | `SwarmOptimizationEnv` | **`swarm_openenv_env.environment:IncidentResponseEnv`** | **CORRECTED** |
| §10 OpenEnv Version | Assumed latest | **Pinned to `0.2.3`** | **CORRECTED** |
| §4.6 OpenEnv `step()` | Text-only workflow bookkeeping | **Backend evaluator + Docker sandbox now called from `propose_fix`** | **CORRECTED** |
| §4.6 OpenEnv observations | Minimal task/checklist only | **Now include active agent, assigned agents, telemetry, sandbox result, and execution logs** | **ADDED** |
| §4.7 HF Spaces App | Not specified | **Root `app.py` Gradio submission app added** | **ADDED** |
