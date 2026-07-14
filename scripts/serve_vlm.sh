#!/usr/bin/env bash
# DMS reproduction project — launch the local vLLM OpenAI-compatible server
# for Qwen2.5-VL-7B-Instruct (the 7B-grade VLM mandated by the task spec).
#
# The eval harness / agents talk to it via the OpenAI Chat Completions API at
# http://127.0.0.1:8000/v1 (pass --vllm_base_url to run_eval_harness.py).
#
# Usage:
#   bash scripts/serve_vlm.sh [--port 8000] [--gpu-memory-utilization 0.85]
#
# Requires the model snapshot at $MODEL_DIR (default below) and the vLLM
# virtualenv at $VLLM_VENV (default .venv_vllm). Stops an existing server on
# the same port before starting a fresh one.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${VLLM_PORT:-8000}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
MODEL_DIR="${VLLM_MODEL_DIR:-${PROJECT_ROOT}/models/Qwen2.5-VL-7B-Instruct}"
VLLM_VENV="${VLLM_VENV:-${PROJECT_ROOT}/.venv_vllm}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --model) MODEL_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model not found at $MODEL_DIR" >&2
  exit 1
fi

# Stop any existing server on this port (best-effort).
existing_pid="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
if [[ -n "$existing_pid" ]]; then
  echo "Stopping existing server on port $PORT (pid $existing_pid)..."
  kill "$existing_pid" 2>/dev/null || true
  sleep 3
fi

echo "Starting vLLM server: model=$MODEL_DIR port=$PORT gpu_mem_util=$GPU_MEM_UTIL"
exec "${VLLM_VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt image=4
