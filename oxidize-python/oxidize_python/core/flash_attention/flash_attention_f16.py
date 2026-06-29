"""Flash attention decode against an f16-stored KV cache.

Mirrors oxidize-golang/core/flash_attention/dot_product_f16c.go and
flash_attention_f16.go (which mirror ``flash_attention_decode_f16`` /
``flash_attention_decode_heads_f16`` in oxidize-core). Keys and values are
IEEE-754 half bits (``int`` uint16 values) and are converted to f32 on the fly
via the f16c-style helpers, avoiding a separate dequant pass.
"""

from __future__ import annotations

import math

from oxidize_python.core.flash_attention.flash_attention import Error
from oxidize_python.core.quantization._f16 import f16_bits_to_f32


def dot_product_f32_f16(a: list[float], b: list[int]) -> float:
    """Dot product of an f32 query and an f16 key (uint16 bits).

    Mirrors ``DotProductF32F16`` (Go) / ``dot_product_f32_f16_avx2`` (Rust).
    """
    n = min(len(a), len(b))
    total = 0.0
    for i in range(n):
        total += a[i] * f16_bits_to_f32(b[i])
    return total


def axpy_f32(out: list[float], scale: float, row: list[float]) -> None:
    """out[i] += scale * row[i] in a single fused pass.

    Mirrors ``AxpyF32`` (Go) / ``axpy_f32_avx2`` (Rust).
    """
    n = min(len(out), len(row))
    for i in range(n):
        out[i] += scale * row[i]


def axpy_f32_f16(out: list[float], scale: float, row: list[int]) -> None:
    """out[i] += scale * f16_to_f32(row[i]) over an f16 value row.

    Mirrors ``AxpyF32F16`` (Go) / ``axpy_f32_f16_avx2`` (Rust).
    """
    n = min(len(out), len(row))
    for i in range(n):
        out[i] += scale * f16_bits_to_f32(row[i])


def flash_attention_decode_f16(
    query: list[float],
    key_layer: list[int],
    value_layer: list[int],
    output: list[float],
    seq_len: int,
    head_dim: int,
    kv_len: int,
    kv_head: int,
) -> None:
    """Single-head decode attention against an f16 KV cache.

    KV is laid out as ``[seq_len][kv_len]`` row-major (mirroring oxidize-core).
    Keys and values are uint16 IEEE-754 half bits, converted to f32 on the fly
    via the f16c-style helpers (no separate dequant pass). Uses online softmax.
    Mirrors ``FlashAttentionDecodeF16``.
    """
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
    q = query[:head_dim]
    m = float("-inf")
    l = 0.0
    for i in range(head_dim):
        output[i] = 0.0
    for t in range(seq_len):
        row = t * kv_len + kv_off
        key = key_layer[row : row + head_dim]
        score = dot_product_f32_f16(q, key) * scale
        new_max = score if score > m else m
        alpha = math.exp(m - new_max) if m != float("-inf") else 0.0
        beta = math.exp(score - new_max)
        m = new_max
        l = l * alpha + beta
        # out = out*alpha + beta*value (online softmax), value is f16.
        if alpha != 1.0:
            for d in range(head_dim):
                output[d] *= alpha
        axpy_f32_f16(output, beta, value_layer[row : row + head_dim])
    if l > 0.0:
        inv = 1.0 / l
        for d in range(head_dim):
            output[d] *= inv


def flash_attention_decode_heads_f16(
    query_heads: list[float],
    key_layer: list[int],
    value_layer: list[int],
    output: list[float],
    seq_len: int,
    head_dim: int,
    kv_len: int,
    num_heads: int,
    kv_heads: int,
) -> None:
    """Grouped-query decode attention over an f16 KV cache.

    KV is shared across query heads via
    ``kv_head = query_head // (num_heads // kv_heads)``. Mirrors
    ``FlashAttentionDecodeHeadsF16``.
    """
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
        flash_attention_decode_f16(
            q, key_layer, value_layer, out, seq_len, head_dim, kv_len, kv_h
        )
        output[h * head_dim : (h + 1) * head_dim] = out
