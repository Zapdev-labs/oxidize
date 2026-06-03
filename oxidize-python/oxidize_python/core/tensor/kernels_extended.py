"""Extended tensor kernels ported from oxidize-core/src/compute/tensor.rs."""

from __future__ import annotations

import math

from oxidize_python.core.tensor.dtype import ActivationFn
from oxidize_python.core.tensor.errors import (
    AttentionError,
    GemmError,
    LinearActivationError,
    SoftmaxError,
)
from oxidize_python.core.tensor.gemv import gemm_f32, gemv_f32_transposed


def extract_bits(bitstream: bytes, index: int, bits: int) -> int:
    if bits == 0 or bits > 32:
        return 0
    byte_index = (index * bits) // 8
    bit_offset = (index * bits) % 8
    value = 0
    for i in range((bits + 7) // 8):
        if byte_index + i >= len(bitstream):
            break
        value |= bitstream[byte_index + i] << (8 * i)
    return (value >> bit_offset) & ((1 << bits) - 1)


def linear_activation_f32(
    matrix: list[float],
    rows: int,
    cols: int,
    vector: list[float],
    activation: ActivationFn,
    output: list[float],
) -> None:
    expected_matrix = rows * cols
    if len(matrix) != expected_matrix:
        raise LinearActivationError(
            f"invalid matrix length: expected {expected_matrix}, got {len(matrix)}"
        )
    if len(vector) != cols:
        raise LinearActivationError("vector too small")
    if len(output) < rows:
        raise LinearActivationError("output too small")
    gemv_f32_transposed(matrix, rows, cols, vector, output)
    for i in range(rows):
        v = output[i]
        if activation == ActivationFn.RELU:
            output[i] = max(0.0, v)
        elif activation == ActivationFn.GELU:
            output[i] = 0.5 * v * (1.0 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v**3)))
        elif activation == ActivationFn.SILU:
            output[i] = v / (1.0 + math.exp(-v))


def softmax_f32(input_: list[float], output: list[float]) -> None:
    if not input_:
        raise SoftmaxError("empty input")
    if len(output) < len(input_):
        raise SoftmaxError("output too small")
    max_val = max(input_)
    exps = [math.exp(v - max_val) for v in input_]
    total = sum(exps)
    inv = 1.0 / total if total else 0.0
    for i, e in enumerate(exps):
        output[i] = e * inv


def scaled_dot_product_attention_f32(
    query: list[float],
    key: list[float],
    value: list[float],
    num_heads: int,
    head_dim: int,
    seq_len: int,
    scale: float,
    output: list[float],
) -> None:
    if num_heads <= 0 or head_dim <= 0 or seq_len <= 0:
        raise AttentionError("invalid attention shape")
    q_size = num_heads * head_dim * seq_len
    if len(query) < q_size or len(key) < q_size or len(value) < q_size:
        raise AttentionError("buffer too small")
    if len(output) < q_size:
        raise AttentionError("output too small")
    for h in range(num_heads):
        for i in range(seq_len):
            scores = [0.0] * seq_len
            q_base = h * head_dim * seq_len + i * head_dim
            for j in range(seq_len):
                k_base = h * head_dim * seq_len + j * head_dim
                dot = sum(query[q_base + d] * key[k_base + d] for d in range(head_dim))
                scores[j] = dot * scale
            attn = [0.0] * seq_len
            softmax_f32(scores, attn)
            out_base = h * head_dim * seq_len + i * head_dim
            for d in range(head_dim):
                output[out_base + d] = sum(
                    attn[j] * value[h * head_dim * seq_len + j * head_dim + d]
                    for j in range(seq_len)
                )


def flash_attention_prefill_f32(
    query: list[float],
    key: list[float],
    value: list[float],
    num_heads: int,
    head_dim: int,
    seq_len: int,
    scale: float,
    output: list[float],
) -> None:
    scaled_dot_product_attention_f32(
        query, key, value, num_heads, head_dim, seq_len, scale, output
    )


def flash_attention_decode_f32(
    query: list[float],
    key_cache: list[float],
    value_cache: list[float],
    num_heads: int,
    head_dim: int,
    cache_len: int,
    scale: float,
    output: list[float],
) -> None:
    scaled_dot_product_attention_f32(
        query, key_cache, value_cache, num_heads, head_dim, cache_len, scale, output
    )


def batched_gemm_f32(
    a: list[float],
    b: list[float],
    batch: int,
    m: int,
    k: int,
    n: int,
    output: list[float],
) -> None:
    if batch <= 0:
        raise GemmError("invalid batch")
    expected = batch * m * n
    if len(output) < expected:
        raise GemmError("output too small")
    a_stride = m * k
    b_stride = k * n
    out_stride = m * n
    for bi in range(batch):
        gemm_f32(
            a[bi * a_stride : (bi + 1) * a_stride],
            m,
            k,
            b[bi * b_stride : (bi + 1) * b_stride],
            n,
            output[bi * out_stride : (bi + 1) * out_stride],
        )


def apply_alibi_bias_f32(
    scores: list[float],
    seq_len: int,
    num_heads: int,
    head_idx: int,
    slope: float,
) -> None:
    for i in range(seq_len):
        for j in range(seq_len):
            idx = head_idx * seq_len * seq_len + i * seq_len + j
            if idx < len(scores):
                scores[idx] -= slope * abs(i - j)


def repeat_kv_heads_f32(
    key: list[float],
    value: list[float],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    out_key: list[float],
    out_value: list[float],
) -> None:
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise AttentionError("invalid kv head repeat")
    repeat = num_heads // num_kv_heads
    for h in range(num_heads):
        src = (h // repeat) * head_dim * seq_len
        dst = h * head_dim * seq_len
        for i in range(head_dim * seq_len):
            if src + i < len(key) and dst + i < len(out_key):
                out_key[dst + i] = key[src + i]
            if src + i < len(value) and dst + i < len(out_value):
                out_value[dst + i] = value[src + i]


def moe_route_topk_f32(
    router_logits: list[float],
    experts: int,
    top_k: int,
    out_weights: list[float],
    out_indices: list[int],
) -> None:
    if experts <= 0 or top_k <= 0:
        raise AttentionError("invalid moe routing")
    ranked = sorted(range(len(router_logits)), key=lambda i: router_logits[i], reverse=True)
    chosen = ranked[:top_k]
    raw = [math.exp(router_logits[i]) for i in chosen]
    total = sum(raw) or 1.0
    for i, idx in enumerate(chosen):
        if i < len(out_weights):
            out_weights[i] = raw[i] / total
        if i < len(out_indices):
            out_indices[i] = idx % experts
