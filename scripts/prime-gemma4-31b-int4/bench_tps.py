#!/usr/bin/env python3
"""Async throughput benchmark for OpenAI-compatible Gemma 4 endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import aiohttp


@dataclass
class StreamResult:
    ttft_ms: float
    output_tokens: int
    total_ms: float


async def stream_one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> StreamResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft_ms = 0.0
    output_tokens = 0

    async with session.post(f"{url}/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if ttft_ms == 0.0 and chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                output_tokens = usage["completion_tokens"]

    total_ms = (time.perf_counter() - t0) * 1000.0
    if output_tokens == 0:
        output_tokens = max_tokens
    return StreamResult(ttft_ms=ttft_ms, output_tokens=output_tokens, total_ms=total_ms)


async def run_sweep(
    url: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    prompt: str,
    warmup: int,
) -> dict[str, float]:
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(warmup):
            await stream_one(session, url, model, prompt, min(32, max_tokens))

        t0 = time.perf_counter()
        tasks = [
            stream_one(session, url, model, prompt, max_tokens)
            for _ in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    total_out = sum(r.output_tokens for r in results)
    tps = total_out / elapsed if elapsed > 0 else 0.0
    ttfts = [r.ttft_ms for r in results if r.ttft_ms > 0]
    decode_ms = [
        (r.total_ms - r.ttft_ms) / max(r.output_tokens, 1) for r in results
    ]

    return {
        "concurrency": float(concurrency),
        "output_tps": tps,
        "total_output_tokens": float(total_out),
        "elapsed_s": elapsed,
        "ttft_p50_ms": statistics.median(ttfts) if ttfts else 0.0,
        "tpot_p50_ms": statistics.median(decode_ms) if decode_ms else 0.0,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 TPS benchmark")
    parser.add_argument("--url", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="google/gemma-4-31B-it-qat-w4a16-ct")
    parser.add_argument(
        "--concurrency",
        default="1,4,8,16,32,48,64,80,96",
        help="Comma-separated concurrency levels",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--target-tps", type=float, default=1000.0)
    parser.add_argument(
        "--prompt",
        default=(
            "Summarize the key tradeoffs between tensor parallelism and "
            "data parallelism for LLM inference on dual H100 GPUs."
        ),
    )
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    best = 0.0
    print(f"url={args.url} model={args.model} max_tokens={args.max_tokens}")
    print(f"{'conc':>6} {'out_tps':>10} {'ttft_p50':>10} {'tpot_p50':>10}")
    print("-" * 42)

    for c in levels:
        row = await run_sweep(
            args.url, args.model, c, args.max_tokens, args.prompt, warmup=1
        )
        best = max(best, row["output_tps"])
        print(
            f"{int(row['concurrency']):>6} "
            f"{row['output_tps']:>10.1f} "
            f"{row['ttft_p50_ms']:>10.1f} "
            f"{row['tpot_p50_ms']:>10.1f}"
        )

    print("-" * 42)
    print(f"peak_output_tps={best:.1f} target={args.target_tps:.0f}")
    if best >= args.target_tps:
        print("PASS: target TPS reached")
    else:
        gap = args.target_tps - best
        print(f"GAP: {gap:.1f} tok/s below target — try TIER=alpha or TIER=charlie")


if __name__ == "__main__":
    asyncio.run(main())
