"""OXK full decode-token benchmark (Python port), Qwen3-30B-A3B.

Reports tok/s for one *real* token's worth of GEMVs (every attention + MoE
projection across all layers) instead of a single isolated GEMV. tok/s is
derived from measured GB/s applied to the known per-token weight-byte volume,
so there is no "1 GEMV = 1 token" fiction.

The pure-Python kernel is slow, so OXK_BENCH_LAYERS defaults to 1: we time a
small subset and project the full 48-layer token from the measured GB/s.

Keep the plan in sync with oxidize-golang/cmd/bench_oxk/main.go and
oxidize-kernels/benches/oxk_token_bench.rs.
"""

import os
import random
import sys
import time

# Prefer the oxidize-python sibling of this script; fall back to ~/oxidize.
for _cand in (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "oxidize-python"),
    os.path.expanduser("~/oxidize/oxidize-python"),
):
    if os.path.isdir(os.path.join(_cand, "oxidize_python")):
        sys.path.insert(0, _cand)
        break

from oxidize_python.core.oxk import (  # noqa: E402
    QK_K,
    BLOCK_Q4_K_SIZE,
    BLOCK_Q8_K_BYTES,
    quantize_q8_k_into,
    gemv_q4k_range,
    oxk_cpu_summary,
)

# Canonical Qwen3-30B-A3B config (see Go bench for full notes).
HIDDEN = 2048
NUM_LAYERS = 48
Q_OUT = 4096
KV_OUT = 512
N_EXPERTS = 8  # active per token
MOE_INTER = 768
ROUTER_OUT = 128
VOCAB = 151936


def one_layer_plan():
    # (label, rows, cols, count)
    return [
        ("attn.q", Q_OUT, HIDDEN, 1),
        ("attn.k", KV_OUT, HIDDEN, 1),
        ("attn.v", KV_OUT, HIDDEN, 1),
        ("attn.o", HIDDEN, Q_OUT, 1),
        ("moe.router", ROUTER_OUT, HIDDEN, 1),
        ("moe.gate", MOE_INTER, HIDDEN, N_EXPERTS),
        ("moe.up", MOE_INTER, HIDDEN, N_EXPERTS),
        ("moe.down", HIDDEN, MOE_INTER, N_EXPERTS),
    ]


def token_plan(n_layers):
    ops = []
    for _ in range(n_layers):
        ops.extend(one_layer_plan())
    ops.append(("lm_head", VOCAB, HIDDEN, 1))
    return ops


def plan_bytes(ops):
    total = 0
    for _label, rows, cols, count in ops:
        row_bytes = (cols // QK_K) * BLOCK_Q4_K_SIZE
        total += rows * row_bytes * count
    return total


def plan_flops(ops):
    return sum(rows * cols * 2.0 * count for _l, rows, cols, count in ops)


def env_int(key, default):
    v = os.getenv(key)
    if v:
        try:
            return int(v)
        except ValueError:
            pass
    return default


def main():
    print("=== Python OXK full decode-token Benchmark (Qwen3-30B-A3B) ===")
    print(oxk_cpu_summary())

    n_layers = max(1, min(NUM_LAYERS, env_int("OXK_BENCH_LAYERS", 1)))
    tokens = max(1, env_int("OXK_BENCH_TOKENS", 1))

    timed_ops = token_plan(n_layers)
    full_ops = token_plan(NUM_LAYERS)
    timed_bytes = plan_bytes(timed_ops)
    full_bytes = plan_bytes(full_ops)
    full_flops = plan_flops(full_ops)

    # One contiguous weight buffer; each op streams a distinct region.
    weights = bytearray(random.getrandbits(8) for _ in range(timed_bytes))
    # Tame the per-block f16 headers (d, dmin) so accumulators stay finite.
    for off in range(0, len(weights) - 3, BLOCK_Q4_K_SIZE):
        for h in range(2):
            raw = weights[off + h * 2] | (weights[off + h * 2 + 1] << 8)
            tamed = (raw & 0x83FF) | (0x3000 + ((raw >> 10) & 0x7) * 0x400)
            weights[off + h * 2] = tamed & 0xFF
            weights[off + h * 2 + 1] = (tamed >> 8) & 0xFF

    q8k_by_cols = {}
    out_by_rows = {}
    for _label, rows, cols, _count in timed_ops:
        if cols not in q8k_by_cols:
            blocks = cols // QK_K
            vec = [(i % 255) / 64.0 - 2.0 for i in range(cols)]
            q8k = bytearray(blocks * BLOCK_Q8_K_BYTES)
            quantize_q8_k_into(vec, blocks, q8k)
            q8k_by_cols[cols] = q8k
        if rows not in out_by_rows:
            out_by_rows[rows] = [0.0] * rows

    def run_token():
        cursor = 0
        sink = 0.0
        for _label, rows, cols, count in timed_ops:
            blocks = cols // QK_K
            row_bytes = blocks * BLOCK_Q4_K_SIZE
            q8k = q8k_by_cols[cols]
            out = out_by_rows[rows]
            for _ in range(count):
                region = weights[cursor : cursor + rows * row_bytes]
                gemv_q4k_range(region, blocks, q8k, out)
                cursor += rows * row_bytes
                sink += out[0]
        return sink

    run_token()  # warmup

    start = time.perf_counter()
    sink = 0.0
    for _ in range(tokens):
        sink += run_token()
    elapsed = time.perf_counter() - start

    gbps = timed_bytes * tokens / 1e9 / elapsed
    gflops = plan_flops(timed_ops) * tokens / 1e9 / elapsed
    full_token_sec = full_bytes / 1e9 / gbps
    proj_tok_s = 1.0 / full_token_sec

    print(f"\nTimed: {n_layers} layer(s) + lm_head, {tokens} token-pass(es)  (sink={sink:.3e})")
    print(f"Streamed per timed pass: {timed_bytes / 1e6:.2f} MB")
    print(f"Throughput:              {gflops:.4f} GFLOP/s")
    print(f"Memory bandwidth:        {gbps:.4f} GB/s")
    print(f"Full token (48L) weight bytes: {full_bytes / 1e9:.2f} GB, {full_flops / 1e9:.1f} GFLOP")
    print(f"Projected full-token decode:   {full_token_sec:.3f} s/token  =>  {proj_tok_s:.4f} tok/s")


if __name__ == "__main__":
    main()
