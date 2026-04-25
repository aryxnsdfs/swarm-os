#!/usr/bin/env bash
# ============================================================================
# Swarm-OS — HF Space entrypoint
#
#   1. Pull the GGUF from your HF model repo into /data/models (persistent).
#   2. Start llama-cpp-python OpenAI-compatible server on 127.0.0.1:1234.
#      This replaces LM Studio so backend/main.py and inference.py keep
#      talking to the same LOCAL_OPENAI_BASE_URL they used locally.
#   3. Wait until the model is ready, then exec uvicorn on 0.0.0.0:$PORT
#      (HF Space app_port = 7860).
#
# Required Space secrets:
#   HF_TOKEN          — read access to the model repo
#   MODEL_REPO_ID     — e.g. aryan-gupta-2010/meta_hackthon_2010_2026
#   MODEL_FILENAME    — e.g. Llama-3.1-8B-Instruct.Q4_K_M.gguf
# Optional:
#   LLM_PROVIDER=local           (already exported in Dockerfile)
#   LOCAL_OPENAI_BASE_URL=...    (already exported in Dockerfile)
#   ALLOW_SCRIPTED_BASELINE=1    (fall back to scripted run if model fails)
#   N_GPU_LAYERS                 (default -1 = offload all layers to GPU)
#   N_CTX                        (default 4096)
# ============================================================================
set -euo pipefail

PORT="${PORT:-7860}"
# Strip whitespace/newlines that HF Secrets UI sometimes adds
MODEL_REPO_ID="$(echo -n "${MODEL_REPO_ID:-aryxn323/meta_hackthon_2010_2026}" | tr -d '[:space:]')"
MODEL_FILENAME="$(echo -n "${MODEL_FILENAME:-Llama-3.1-8B-Instruct.Q4_K_M.gguf}" | tr -d '\n\r' | xargs)"
MODEL_DIR="/data/models"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILENAME}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
N_CTX="${N_CTX:-4096}"

echo "[start.sh] PORT=${PORT}  MODEL=${MODEL_REPO_ID}/${MODEL_FILENAME}"

# ----- 1. Download GGUF (cached in /data/models across restarts) -----
mkdir -p "${MODEL_DIR}"
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[start.sh] GGUF not cached, downloading from HF Hub..."
  python - <<PY
import os
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=os.environ["MODEL_REPO_ID"],
    filename=os.environ["MODEL_FILENAME"],
    local_dir=os.environ.get("MODEL_DIR", "/data/models"),
    token=os.environ.get("HF_TOKEN") or None,
)
print(f"[start.sh] downloaded -> {path}")
PY
else
  echo "[start.sh] GGUF cache hit: ${MODEL_PATH}"
fi

# Make MODEL_PATH visible to inference.py (matches its env var contract)
export MODEL_PATH="${MODEL_PATH}"

# ----- 2. Launch llama.cpp OpenAI server on 127.0.0.1:1234 -----
echo "[start.sh] starting llama-cpp-python server on 127.0.0.1:1234 (n_gpu_layers=${N_GPU_LAYERS}, n_ctx=${N_CTX})"
python -m llama_cpp.server \
    --model "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port 1234 \
    --n_gpu_layers "${N_GPU_LAYERS}" \
    --n_ctx "${N_CTX}" \
    > /tmp/llama_cpp_server.log 2>&1 &
LLAMA_PID=$!
echo "[start.sh] llama-cpp-python pid=${LLAMA_PID}"

# Forward signals so SIGTERM from the Space tears down the model server too.
trap 'echo "[start.sh] shutting down..."; kill -TERM "${LLAMA_PID}" 2>/dev/null || true; wait || true' INT TERM

# ----- 3. Wait for /v1/models readiness (max ~120s) -----
echo "[start.sh] waiting for llama-cpp /v1/models ..."
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:1234/v1/models >/dev/null 2>&1; then
    echo "[start.sh] llama-cpp ready after ${i}s"
    break
  fi
  if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
    echo "[start.sh] llama-cpp died during boot. Last 80 log lines:"
    tail -n 80 /tmp/llama_cpp_server.log || true
    if [[ "${ALLOW_SCRIPTED_BASELINE:-0}" != "1" ]]; then
      echo "[start.sh] set ALLOW_SCRIPTED_BASELINE=1 to keep the UI alive without a live model."
      exit 1
    fi
    echo "[start.sh] continuing without live model (scripted baseline)."
    break
  fi
  sleep 1
done

# ----- 4. Boot the FastAPI dashboard (serves React + /ws + /api/*) -----
echo "[start.sh] starting backend uvicorn on 0.0.0.0:${PORT}"
cd /app/backend
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}" --log-level info
