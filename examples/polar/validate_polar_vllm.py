#!/usr/bin/env python3
"""Replay a prepared PoLar path workload against a running vLLM server."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[position]


def request_one(
    base_url: str,
    model: str,
    row: dict,
    max_tokens: int,
    temperature: float,
    ignore_eos: bool,
):
    payload = json.dumps(
        {
            "model": model,
            "prompt": row["token_ids"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "ignore_eos": ignore_eos,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read())
        latency = time.perf_counter() - started
        choice = result["choices"][0]
        usage = result.get("usage", {})
        return {
            "request_id": row["request_id"],
            "path": row["path"],
            "text": choice.get("text", ""),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency_seconds": latency,
            "error": None,
        }
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
        return {
            "request_id": row.get("request_id"),
            "path": row.get("path"),
            "latency_seconds": time.perf_counter() - started,
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="polar-qwen")
    parser.add_argument("--strategy", choices=("serial", "identical_path", "dynamic_layer"), required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means all prompts")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1:
        parser.error("--concurrency and --max-tokens must be positive")
    import os

    server_strategy = os.environ.get("VLLM_POLAR_SCHEDULER", "")
    if server_strategy and server_strategy != args.strategy:
        parser.error(
            f"server policy is {server_strategy}, but requested policy is {args.strategy}; "
            "restart the server with examples/polar/launch_qwen2.sh"
        )

    rows = [
        json.loads(line)
        for line in args.prompts_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        parser.error("no prompts selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"{args.strategy}_results.jsonl"
    summary_path = args.output_dir / f"{args.strategy}_summary.json"

    start = time.perf_counter()
    results = []
    with results_path.open("w", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    request_one,
                    args.base_url,
                    args.model,
                    row,
                    args.max_tokens,
                    args.temperature,
                    args.ignore_eos,
                )
                for row in rows
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
    wall_seconds = time.perf_counter() - start
    successful = [result for result in results if not result.get("error")]
    latencies = [result["latency_seconds"] for result in successful]
    prompt_tokens = sum(result.get("prompt_tokens") or 0 for result in successful)
    completion_tokens = sum(result.get("completion_tokens") or 0 for result in successful)
    summary = {
        "strategy": args.strategy,
        "expected_requests": len(rows),
        "completed_requests": len(successful),
        "failed_requests": len(rows) - len(successful),
        "wall_seconds": wall_seconds,
        "request_throughput_per_second": len(successful) / wall_seconds if wall_seconds else 0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens_per_second": (prompt_tokens + completion_tokens) / wall_seconds if wall_seconds else 0,
        "latency_mean_seconds": statistics.mean(latencies) if latencies else None,
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "latency_p99_seconds": percentile(latencies, 0.99),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
        "base_url": args.base_url,
        "model": args.model,
        "results_jsonl": str(results_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
