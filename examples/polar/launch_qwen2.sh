#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 MODEL_PATH PATH_MANIFEST.json {serial|identical_path|dynamic_layer} [GPU_ID] [PORT]" >&2
  exit 2
fi

MODEL_PATH=$1
MANIFEST=$(realpath "$2")
STRATEGY=$3
GPU_ID=${4:-0}
PORT=${5:-8000}

case "$STRATEGY" in
  serial|identical_path|dynamic_layer) ;;
  *) echo "Invalid scheduler policy: $STRATEGY" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VLLM_POLAR_PATH_MANIFEST="$MANIFEST"
export VLLM_POLAR_SCHEDULER="$STRATEGY"
export VLLM_POLAR_KV_CACHE_MODE="${VLLM_POLAR_KV_CACHE_MODE:-execution}"
export VLLM_PLUGINS=""
# The PoLar request-context hook is currently implemented in the V1 runner.
export VLLM_USE_V2_MODEL_RUNNER=0

echo "[polar] runner=V1 strategy=$STRATEGY manifest=$MANIFEST gpu=$GPU_ID"

exec vllm serve "$MODEL_PATH" \
  --served-model-name polar-qwen \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --hf-overrides '{"architectures":["PolarQwen2ForCausalLM"]}' \
  --port "$PORT"
