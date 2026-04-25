# FrontierLabs Swarm-OS — Setup Guide

## Prerequisites

### Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| Node.js | v18+ | Frontend build toolchain |
| npm | v9+ | Package manager (ships with Node) |
| Python | 3.10+ | Backend API server |
| pip | latest | Python package manager |
| Git | any | Version control |

### Optional (for production GPU inference)

| Tool | Version | Purpose |
|------|---------|---------|
| CUDA Toolkit | 12.1+ | GPU acceleration |
| Docker Desktop | latest | Sandbox execution |
| NVIDIA GPU | RTX 3060 12GB+ | Model inference |

---

## Quick Start (Demo Mode)

The frontend runs fully standalone — no backend, no GPU, no Docker needed.
This is the recommended mode for live pitching.

```bash
# 1. Clone and install frontend
cd d:\op\frontend
npm install

# 2. Start the development server
npm run dev

# 3. Open http://localhost:5173
# 4. Click "Start Scenario" to run the full simulation
```

---

## Full Stack Setup (Backend + Frontend)

### Step 1: Frontend

```bash
cd d:\op\frontend
npm install
npm run dev
```

### Step 2: Backend (separate terminal)

```bash
cd d:\op\backend
pip install -r requirements.txt
python main.py
```

The backend starts on `http://localhost:8000` with API docs at `/docs`.

---

## About the Model System

### Current State: Mock Inference

All LLM inference is currently **mocked** for development. The backend returns
pre-scripted responses per agent role. No GPU, no model downloads, no VRAM needed.

This is by design — the demo frontend drives the entire simulation from
pre-built scenario scripts in `frontend/src/data/scenarios.js`.

### Enabling Real Model Inference

To connect real LLM inference, you need:

1. **Install Unsloth (QLoRA inference)**:
   ```bash
   pip install unsloth[colab-new] bitsandbytes transformers accelerate
   ```

2. **Download models** (one-time, automatic on first run):
   - Llama-3.1-8B-Instruct 4-bit: ~4GB download, ~6GB VRAM
   - DeepSeek-R1-Distill-Llama-8B 4-bit: ~4GB download, ~5.5GB VRAM

3. **Enable inference** in `backend/model/inference.py`:
   ```python
   # Change mock_mode from True to False
   engine = InferenceEngine(config_manager, mock_mode=False)
   ```

4. **Verify GPU** (Python):
   ```python
   import torch
   print(torch.cuda.is_available())        # Should be True
   print(torch.cuda.get_device_name(0))    # e.g., "NVIDIA GeForce RTX 3060"
   print(f"{torch.cuda.mem_get_info()[1]/1e9:.1f}GB VRAM")
   ```

### GPU Requirements by Model

| Model | VRAM Required | Card |
|-------|--------------|------|
| Llama-3.1-8B 4-bit | 6 GB | RTX 3060 12GB |
| DeepSeek-R1-Distill 4-bit | 5.5 GB | RTX 3060 12GB |
| Llama-3.1-8B Full | 16 GB | RTX 4090 24GB |
| Llama-3.1-70B | 40 GB | A100/H100 |

---

## Project Structure

```
d:\op\
├── frontend/                  # React + Vite + TailwindCSS
│   ├── src/
│   │   ├── components/        # All UI panels
│   │   ├── data/scenarios.js  # Simulation event timeline
│   │   ├── store/             # State management (Context + useReducer)
│   │   └── hooks/             # useSimulation hook
│   └── package.json
│
├── backend/                   # Python FastAPI
│   ├── main.py                # API server + WebSocket
│   ├── config.yaml            # Model registry + agent overrides
│   ├── engine/                # Physics, rewards, evaluator, causal graph
│   ├── agents/                # Swarm orchestrator
│   ├── model/                 # Config + inference abstraction
│   └── snorkel_logger.py      # JSONL auto-labeler
│
└── SETUP.md                   # This file
```

---

## Environment Variables (Optional)

None required for demo mode. For production:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SWARM_MOCK_MODE` | `true` | Set to `false` for real inference |
| `SWARM_GPU_VRAM` | `12` | Available VRAM in GB |
| `SWARM_LOG_LEVEL` | `INFO` | Python logging level |

---

## Troubleshooting

- **Frontend won't start**: Run `npm install` first. Requires Node.js 18+.
- **Backend import errors**: Run `pip install -r requirements.txt`. Requires Python 3.10+.
- **Models not loading**: Check CUDA with `python -c "import torch; print(torch.cuda.is_available())"`.
- **Docker sandbox fails**: Install Docker Desktop and ensure it's running.
