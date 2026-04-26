#!/usr/bin/env bash
# ============================================================================
# Swarm-OS — HF Space entrypoint
# ============================================================================
set -euo pipefail

PORT="${PORT:-7860}"
MODEL_REPO_ID="$(echo -n "${MODEL_REPO_ID:-aryxn323/meta_hackthon_2010_2026}" | tr -d '[:space:]')"
MODEL_FILENAME="$(echo -n "${MODEL_FILENAME:-Llama-3.1-8B-Instruct.Q4_K_M.gguf}" | tr -d '\n\r' | xargs)"
MODEL_DIR="/data/models"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILENAME}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
N_CTX="${N_CTX:-4096}"
SWARM_LOG_FILE="${SWARM_LOG_FILE:-/data/swarm-os-container.log}"

# Force line buffering on every shell command output. This is the actual fix
# for HF Space Container "No logs" — Docker buffers up to 64 KB on a non-TTY
# stdout pipe, so we use stdbuf to force line-buffered output everywhere.
mkdir -p "$(dirname "${SWARM_LOG_FILE}")"
: > "${SWARM_LOG_FILE}"
export SWARM_LOG_FILE="${SWARM_LOG_FILE}"
export PYTHONUNBUFFERED=1

# Helper: print + flush + tee to log file
log() {
  printf "%s\n" "$*"
  printf "%s\n" "$*" >> "${SWARM_LOG_FILE}" 2>/dev/null || true
}

log "[start.sh] $(date -u +%FT%TZ)  PORT=${PORT}  MODEL=${MODEL_REPO_ID}/${MODEL_FILENAME}"
log "[start.sh] PYTHONUNBUFFERED=${PYTHONUNBUFFERED}  PWD=$(pwd)"

# ----- 1. Download GGUF (cached in /data/models across restarts) -----
mkdir -p "${MODEL_DIR}"
if [[ ! -f "${MODEL_PATH}" ]]; then
  log "[start.sh] GGUF not cached — downloading from HF Hub (this takes 2-3 min first boot)..."
  stdbuf -oL -eL python -u - <<PY
import os, sys
sys.stdout.reconfigure(line_buffering=True)
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=os.environ["MODEL_REPO_ID"],
    filename=os.environ["MODEL_FILENAME"],
    local_dir=os.environ.get("MODEL_DIR", "/data/models"),
    token=os.environ.get("HF_TOKEN") or None,
)
print(f"[start.sh] downloaded -> {path}", flush=True)
PY
else
  log "[start.sh] GGUF cache hit: ${MODEL_PATH}"
fi

export MODEL_PATH="${MODEL_PATH}"

# ----- 2. Launch llama.cpp OpenAI server on 127.0.0.1:1234 -----
log "[start.sh] starting llama-cpp-python server on 127.0.0.1:1234 (n_gpu_layers=${N_GPU_LAYERS}, n_ctx=${N_CTX})"
# Write llama-cpp output to a log file. We tail -F that file into our main
# stdout below so HF Container tab shows token-generation activity in real time.
env -u PORT stdbuf -oL -eL python -u -m llama_cpp.server \
    --model "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port 1234 \
    --n_gpu_layers "${N_GPU_LAYERS}" \
    --n_ctx "${N_CTX}" \
    > /tmp/llama_cpp_server.log 2>&1 &
LLAMA_PID=$!
log "[start.sh] llama-cpp-python pid=${LLAMA_PID}"

# Stream llama-cpp progress to main stdout (prefix with [llama-cpp] for clarity).
# stdbuf -oL forces tail and sed to be line-buffered so output flushes per line.
stdbuf -oL -eL tail -F /tmp/llama_cpp_server.log 2>/dev/null \
  | stdbuf -oL -eL sed 's/^/[llama-cpp] /' &
TAIL_PID=$!

trap 'log "[start.sh] shutting down..."; kill -TERM "${LLAMA_PID}" "${TAIL_PID}" 2>/dev/null || true; wait || true' INT TERM

# ----- 3. Wait for /v1/models readiness (max ~120s) -----
log "[start.sh] waiting for llama-cpp /v1/models ..."
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:1234/v1/models >/dev/null 2>&1; then
    log "[start.sh] llama-cpp ready after ${i}s"
    break
  fi
  if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
    log "[start.sh] llama-cpp died during boot. Last 80 log lines:"
    tail -n 80 /tmp/llama_cpp_server.log || true
    if [[ "${ALLOW_SCRIPTED_BASELINE:-0}" != "1" ]]; then
      log "[start.sh] set ALLOW_SCRIPTED_BASELINE=1 to keep the UI alive without a live model."
      exit 1
    fi
    log "[start.sh] continuing without live model (scripted baseline)."
    break
  fi
  sleep 1
done

# ----- 4. Boot the FastAPI dashboard (serves React + /ws + /api/*) -----
log "[start.sh] starting backend uvicorn on 0.0.0.0:${PORT}"
log "[start.sh] ============================================================"
log "[start.sh]  IF Container tab still shows 'No logs', visit:"
log "[start.sh]    https://aryxn323-swarm-os.hf.space/logs"
log "[start.sh]  for the full inference.py + backend + llama-cpp log stream."
log "[start.sh] ============================================================"
cd /app/backend

# Use `script` to allocate a pseudo-TTY for uvicorn. This is THE canonical
# Docker fix: a non-TTY stdout in Docker uses 64 KB block-buffering, which
# is why HF Space's Container tab shows "No logs" until the buffer fills.
# A PTY forces line-buffered output so every log line appears instantly.
# The persistent /data/swarm-os-container.log file is populated separately
# by Python's FileHandler in main.py — no shell-level tee needed (which
# would re-introduce the buffering problem).
exec script -qefc \
  "stdbuf -oL -eL python -u -m uvicorn main:app --host 0.0.0.0 --port ${PORT} --log-level info --access-log --no-server-header" \
  /dev/null
