#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx2[http2,brotli,zstd]",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run bench_tps.py [ARGS]
# 3. Or make executable and run:
#      chmod +x bench_tps.py && ./bench_tps.py
# ─────────────────

"""Fail-closed process-per-GPU TPS benchmark for oxidize-c."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence
from urllib.parse import urlsplit

import bench_tps_core as core


BenchmarkSummary = core.BenchmarkSummary
RequestRecord = core.RequestRecord
WorkerSpec = core.WorkerSpec
run_phase = core.run_phase
summarize_window = core.summarize_window
verify_completion = core.verify_completion
worker_specs = core.worker_specs


def parse_config(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verified oxidize-c Gemma 4 TPS benchmark")
    urls = parser.add_mutually_exclusive_group()
    urls.add_argument("--urls")
    urls.add_argument("--url")
    parser.add_argument("--model", default="google/gemma-4-31B-it-qat-w4a16-ct")
    parser.add_argument("--concurrency-per-gpu", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-seconds", type=float, default=15.0)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--target-tps", type=float, default=1000.0)
    parser.add_argument("--required-identity", default="oxidize-c")
    parser.add_argument("--ledger", type=Path, default=Path("bench_tps.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("bench_tps_summary.json"))
    parser.add_argument("--prompt", default="Write a concise explanation of GPU inference batching.")
    args = parser.parse_args(argv)
    raw_urls = args.urls or args.url or "http://localhost:8080/v1"
    args.endpoints = tuple(part.strip() for part in raw_urls.split(","))
    args.concurrency_value = (
        args.concurrency_per_gpu
        if args.concurrency_per_gpu is not None
        else args.concurrency if args.concurrency is not None else 64
    )
    positive = (
        args.concurrency_value, args.max_tokens, args.warmup_seconds,
        args.duration_seconds, args.timeout_seconds, args.target_tps,
    )
    if args.concurrency_per_gpu is not None and args.concurrency is not None:
        parser.error("use only one of --concurrency-per-gpu or --concurrency")
    if not args.endpoints or any(not endpoint for endpoint in args.endpoints):
        parser.error("--urls must contain nonempty comma-separated endpoints")
    if len(set(args.endpoints)) != len(args.endpoints):
        parser.error("--urls endpoints must be unique")
    if any(
        urlsplit(endpoint).scheme not in {"http", "https"}
        or not urlsplit(endpoint).netloc
        for endpoint in args.endpoints
    ):
        parser.error("--urls endpoints must be absolute HTTP(S) URLs")
    if any(value <= 0 for value in positive):
        parser.error("concurrency, token, duration, timeout, and target arguments must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_config(argv)
    total_loops = len(args.endpoints) * args.concurrency_value
    common = {
        "endpoints": args.endpoints,
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "concurrency_per_gpu": args.concurrency_value,
        "required_identity": args.required_identity,
    }
    with core.create_client(total_loops, args.timeout_seconds) as client:
        warmup, _, _ = run_phase(
            client=client, **common, duration_s=args.warmup_seconds, phase="warmup",
        )
        measured, started_at, ended_at = run_phase(
            client=client, **common, duration_s=args.duration_seconds, phase="measure",
        )
    summary = summarize_window(
        records=measured,
        expected_workers=tuple(str(index) for index in range(len(args.endpoints))),
        started_at=started_at,
        ended_at=ended_at,
        target_tps=args.target_tps,
    )
    core.write_ledger(args.ledger, [*warmup, *measured])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(f"{json.dumps(summary, indent=2, sort_keys=True)}\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
