"""Flash attention kernels mirroring oxidize-golang/core/flash_attention."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from oxidize_python.core.simd import simd


class Error(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"flash_attention: {message}")


def dot_product_f32(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    width = preferred().lane_width_f32()
    if width <= 1:
        return sum(x * y for x, y in zip(a, b, strict=True))
    total = 0.0
    for i in range(0, len(a), width):
        chunk = zip(a[i : i + width], b[i : i + width], strict=False)
        total += sum(x * y for x, y in chunk)
    return total


def preferred() -> simd.Backend:
    return simd.preferred()


def flash_attention_decode_f32(
    query: list[float],
    key_cache: list[float],
    value_cache: list[float],
    output: list[float],
    seq_len: int,
    head_dim: int,
    scale: float,
) -> None:
    if len(query) < head_dim:
        raise Error("query too small")
    if len(key_cache) < seq_len * head_dim:
        raise Error("key cache too small")
    if len(value_cache) < seq_len * head_dim:
        raise Error("value cache too small")
    if len(output) < head_dim:
        raise Error("output too small")
    q = np.asarray(query[:head_dim], dtype=np.float32)
    K = np.asarray(key_cache[: seq_len * head_dim], dtype=np.float32).reshape(seq_len, head_dim)
    V = np.asarray(value_cache[: seq_len * head_dim], dtype=np.float32).reshape(seq_len, head_dim)
    scores = (K @ q) * scale
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    result = (weights[:, None] * V).sum(axis=0)
    output[:head_dim] = result.tolist()


def flash_attention_decode_gqa(
    query: list[float],
    key_layer: list[float],
    value_layer: list[float],
    output: list[float],
    seq_len: int,
    head_dim: int,
    kv_len: int,
    kv_head: int,
) -> None:
    if len(query) < head_dim:
        raise Error("query too small")
    expected = seq_len * kv_len
    if len(key_layer) < expected or len(value_layer) < expected:
        raise Error("kv cache too small")
    if len(output) < head_dim:
        raise Error("output too small")
    if seq_len == 0:
        for i in range(head_dim):
            output[i] = 0.0
        return
    if head_dim == 0 or kv_len % head_dim != 0:
        raise Error("invalid kv_len for head_dim")
    kv_heads = kv_len // head_dim
    if kv_head < 0 or kv_head >= kv_heads:
        raise Error("invalid kv_head")
    scale = 1.0 / math.sqrt(head_dim)
    kv_off = kv_head * head_dim
    q = np.asarray(query[:head_dim], dtype=np.float32)
    # Gather this KV head's rows: shape (seq_len, head_dim)
    kv_arr = np.asarray(key_layer[: seq_len * kv_len], dtype=np.float32).reshape(seq_len, kv_len)
    K = kv_arr[:, kv_off : kv_off + head_dim]
    V = np.asarray(value_layer[: seq_len * kv_len], dtype=np.float32).reshape(
        seq_len, kv_len
    )[:, kv_off : kv_off + head_dim]
    scores = (K @ q) * scale
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    result = (weights[:, None] * V).sum(axis=0)
    output[:head_dim] = result.tolist()


def flash_attention_decode_heads_gqa(
    query_heads: list[float],
    key_layer: list[float],
    value_layer: list[float],
    output: list[float],
    seq_len: int,
    head_dim: int,
    kv_len: int,
    num_heads: int,
    kv_heads: int,
) -> None:
    q_len = num_heads * head_dim
    if len(query_heads) < q_len or len(output) < q_len:
        raise Error("query/output too small")
    if kv_heads <= 0 or num_heads % kv_heads != 0:
        raise Error("invalid head grouping")
    group = num_heads // kv_heads
    for h in range(num_heads):
        kv_h = h // group
        q = query_heads[h * head_dim : (h + 1) * head_dim]
        out = output[h * head_dim : (h + 1) * head_dim]
        flash_attention_decode_gqa(q, key_layer, value_layer, out, seq_len, head_dim, kv_len, kv_h)


def flash_attention_decode_heads_f32(
    queries: list[float],
    key_cache: list[float],
    value_cache: list[float],
    output: list[float],
    head_count: int,
    seq_len: int,
    head_dim: int,
    scale: float,
) -> None:
    if len(queries) < head_count * head_dim:
        raise Error("queries too small")
    if len(key_cache) < head_count * seq_len * head_dim:
        raise Error("key cache too small")
    if len(value_cache) < head_count * seq_len * head_dim:
        raise Error("value cache too small")
    if len(output) < head_count * head_dim:
        raise Error("output too small")

    def _one(h: int) -> None:
        q = queries[h * head_dim : (h + 1) * head_dim]
        kc = key_cache[h * seq_len * head_dim : (h + 1) * seq_len * head_dim]
        vc = value_cache[h * seq_len * head_dim : (h + 1) * seq_len * head_dim]
        o = output[h * head_dim : (h + 1) * head_dim]
        flash_attention_decode_f32(q, kc, vc, o, seq_len, head_dim, scale)

    with ThreadPoolExecutor(max_workers=head_count) as pool:
        list(pool.map(_one, range(head_count)))


def flash_attention_prefill_f32(
    queries: list[float],
    keys: list[float],
    values: list[float],
    output: list[float],
    seq_len: int,
    head_dim: int,
    scale: float,
) -> None:
    if len(queries) < seq_len * head_dim:
        raise Error("queries too small")
    if len(keys) < seq_len * head_dim:
        raise Error("keys too small")
    if len(values) < seq_len * head_dim:
        raise Error("values too small")
    if len(output) < seq_len * head_dim:
        raise Error("output too small")
    block = 64
    for block_start in range(0, seq_len, block):
        end = min(block_start + block, seq_len)
        for i in range(block_start, end):
            q = queries[i * head_dim : (i + 1) * head_dim]
            for d in range(head_dim):
                output[i * head_dim + d] = 0.0
            scores = [0.0] * (i + 1)
            for j in range(i + 1):
                k = keys[j * head_dim : (j + 1) * head_dim]
                scores[j] = dot_product_f32(q, k) * scale
            max_score = max(scores)
            total = sum(math.exp(s - max_score) for s in scores)
            inv = 1.0 / total
            probs = [math.exp(s - max_score) * inv for s in scores]
            for d in range(head_dim):
                acc = 0.0
                for j in range(i + 1):
                    acc += probs[j] * values[j * head_dim + d]
                output[i * head_dim + d] = acc
