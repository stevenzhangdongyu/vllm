#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "Usage: $0 MODEL_PATH PATH_MANIFEST.json PROMPTS.jsonl OUTPUT_DIR [GPU_ID] [PORT]" >&2
  exit 2
fi

MODEL_PATH=$1
MANIFEST=$(realpath "$2")
PROMPTS=$(realpath "$3")
OUTPUT_DIR=$4
GPU_ID=${5:-0}
PORT=${6:-8000}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if ! command -v curl >/dev/null 2>&1 || ! command -v python >/dev/null 2>&1; then
  echo "curl and python are required" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" || ! -f "$PROMPTS" ]]; then
  echo "Manifest or prompts file not found" >&2
  exit 2
fi
if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Port ${PORT} already serves a healthy endpoint; refusing to reuse or stop it" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
SERVER_PID=
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for strategy in serial identical_path dynamic_layer; do
  strategy_dir="$OUTPUT_DIR/$strategy"
  mkdir -p "$strategy_dir"
  server_log="$strategy_dir/server.log"

  echo "[start] strategy=$strategy gpu=$GPU_ID port=$PORT"
  bash "$SCRIPT_DIR/launch_qwen2.sh" \
    "$MODEL_PATH" "$MANIFEST" "$strategy" "$GPU_ID" "$PORT" \
    >"$server_log" 2>&1 &
  SERVER_PID=$!

  ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[error] server exited for $strategy; inspect $server_log" >&2
      tail -n 100 "$server_log" >&2 || true
      exit 1
    fi
    if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "[error] server readiness timeout for $strategy; inspect $server_log" >&2
    tail -n 100 "$server_log" >&2 || true
    exit 1
  fi

  echo "[warmup] strategy=$strategy requests=8 concurrency=8"
  python "$SCRIPT_DIR/validate_polar_vllm.py" \
    "$PROMPTS" "$strategy_dir/warmup" \
    --strategy "$strategy" --concurrency 8 --limit 8 --max-tokens 8 --ignore-eos

  echo "[benchmark] strategy=$strategy requests=all concurrency=8 max_tokens=64"
  python "$SCRIPT_DIR/validate_polar_vllm.py" \
    "$PROMPTS" "$strategy_dir/benchmark" \
    --strategy "$strategy" --concurrency 8 --limit 0 --max-tokens 64 --ignore-eos

  grep -E 'Resolved architecture|Initializing a V1 LLM engine|Using V2 Model Runner|PoLar heterogeneous|enable_prefix_caching|Traceback|ERROR' \
    "$server_log" > "$strategy_dir/server_check.log" || true
  if grep -q 'Using V2 Model Runner' "$strategy_dir/server_check.log"; then
    echo "[error] $strategy unexpectedly used V2 Model Runner" >&2
    exit 1
  fi
  if ! grep -q 'PoLar heterogeneous path execution is experimental' "$strategy_dir/server_check.log"; then
    echo "[error] PoLar forward hook was not observed for $strategy" >&2
    exit 1
  fi
  if grep -q 'Traceback\|ERROR' "$strategy_dir/server_check.log"; then
    echo "[error] server reported an error for $strategy; inspect $server_log" >&2
    exit 1
  fi

  cleanup
  SERVER_PID=
  sleep 2
done

python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
strategies = ("serial", "identical_path", "dynamic_layer")
rows = []
for strategy in strategies:
    summary_path = root / strategy / "benchmark" / f"{strategy}_summary.json"
    with summary_path.open(encoding="utf-8") as stream:
        rows.append(json.load(stream))

comparison = {
    "comparison_note": (
        "Same path workload, prompts, concurrency, and max_tokens. "
        "EOS ignored for fixed generation work. Client-side non-streaming latency; not TTFT/TPOT."
    ),
    "strategies": rows,
}
output_path = root / "comparison.json"
output_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
print(f"[done] comparison={output_path}")
for row in rows:
    print(
        f"{row['strategy']:16s} requests/s={row['request_throughput_per_second']:.3f} "
        f"tokens/s={row['total_tokens_per_second']:.2f} "
        f"p50={row['latency_p50_seconds']:.3f}s "
        f"p95={row['latency_p95_seconds']:.3f}s "
        f"p99={row['latency_p99_seconds']:.3f}s "
        f"failed={row['failed_requests']}"
    )
PY
