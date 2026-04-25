# ============================================================================
# Swarm-OS — Hugging Face Space (Docker SDK, GPU)
#
# Single-image stack:
#   1. Stage `web`   — node:20 builds the React frontend into frontend/dist
#   2. Stage `app`   — CUDA 12.1 runtime
#                       - llama-cpp-python[server]  (CUDA build)  -> 127.0.0.1:1234
#                       - backend/main.py uvicorn                  -> 0.0.0.0:7860
#                       - frontend/dist mounted at  /
#                       - inference.py reachable from `python inference.py`
# ============================================================================

# -------- Stage 1: frontend build --------
FROM node:20-slim AS web
WORKDIR /web

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# -------- Stage 2: runtime --------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    HF_HOME=/data/hf \
    HUGGINGFACE_HUB_CACHE=/data/hf \
    LOCAL_OPENAI_BASE_URL=http://127.0.0.1:1234 \
    LOCAL_OPENAI_API_KEY=lm-studio \
    LLM_PROVIDER=local

# System deps:
#   - python3.11 + pip
#   - build-essential / cmake / git for compiling llama-cpp-python from source with CUDA
#   - curl for the start.sh readiness probe
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        build-essential cmake git curl ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (cached layer)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt \
    && CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-binary=llama-cpp-python "llama-cpp-python[server]>=0.2.90"

# App code
COPY pyproject.toml README.md openenv.yaml inference.py start.sh /app/
COPY server /app/server
COPY swarm_openenv_env /app/swarm_openenv_env
COPY backend /app/backend
COPY outputs /app/outputs

# Built frontend from stage 1
COPY --from=web /web/dist /app/frontend/dist

# Persistent cache dir for the GGUF (HF Spaces mount /data as persistent storage)
RUN mkdir -p /data/models /data/hf && chmod -R 777 /data

RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["bash", "/app/start.sh"]
