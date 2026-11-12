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

exec vllm serve "$MODEL_PATH" \
  --served-model-name polar-qwen \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --enforce-eager \
  --hf-overrides '{"architectures":["PolarQwen2ForCausalLM"]}' \
  --port "$PORT"
