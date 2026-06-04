"""Tensor ops mirroring oxidize-golang/core/tensor/ops.go."""

from __future__ import annotations

import math

from oxidize_python.core.tensor.dtype import ActivationFn
from oxidize_python.core.tensor.errors import (
    AttentionError,
    LayerNormError,
    RmsNormError,
    RopeError,
    SoftmaxError,
    SwiGluError,
)
from oxidize_python.core.tensor.gemv import gemm_f32, gemv_f32_transposed


def apply_rope_f32(
    input_: list[float],
    output: list[float],
    position: int,
    head_dim: int,
    theta: float,
) -> None:
    if head_dim <= 0 or head_dim % 2 != 0:
        raise RopeError("headDim must be positive and even")
    if len(input_) < head_dim * 2:
        raise RopeError("input buffer too small")
    if len(output) < head_dim * 2:
        raise RopeError("output buffer too small")
    for i in range(0, head_dim, 2):
        freq = 1.0 / (theta ** (2 * i / head_dim))
        angle = position * freq
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x0, x1 = input_[i], input_[i + 1]
        output[i] = x0 * cos_a - x1 * sin_a
        output[i + 1] = x0 * sin_a + x1 * cos_a
    output[head_dim : head_dim * 2] = input_[head_dim : head_dim * 2]


def apply_rope_head_f32(
    input_: list[float],
    output: list[float],
    position: int,
    head_dim: int,
    theta: float,
) -> None:
    if head_dim <= 0 or head_dim % 2 != 0:
        raise RopeError("headDim must be positive and even")
    if len(input_) < head_dim or len(output) < head_dim:
        raise RopeError("buffer too small for headDim")
    if position == 0:
        output[:head_dim] = input_[:head_dim]
        return
    half = head_dim // 2
    inv_head = 1.0 / head_dim
    freq_mul = theta ** (-2.0 * inv_head)
    freq = 1.0
    for i in range(half):
        angle = position * freq
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x0, x1 = input_[i], input_[half + i]
        output[i] = x0 * cos_a - x1 * sin_a
        output[half + i] = x0 * sin_a + x1 * cos_a
        freq *= freq_mul


def rms_norm_f32(input_: list[float], weight: list[float], output: list[float], eps: float) -> None:
    if not input_:
        raise RmsNormError("empty input")
    if len(weight) < len(input_):
        raise RmsNormError("weight too small")
    if len(output) < len(input_):
        raise RmsNormError("output too small")
    sum_sq = sum(v * v for v in input_)
    inv = 1.0 / math.sqrt(sum_sq / len(input_) + eps)
    for i, v in enumerate(input_):
        output[i] = v * inv * weight[i]


def rms_norm_gemv_f32_transposed(
    input_: list[float],
    norm_weight: list[float],
    matrix: list[float],
    output: list[float],
    rows: int,
    cols: int,
    eps: float,
) -> None:
    normalized = [0.0] * cols
    rms_norm_f32(input_[:cols], norm_weight[:cols], normalized, eps)
    gemv_f32_transposed(matrix, rows, cols, normalized, output)


def layer_norm_f32(
    input_: list[float],
    weight: list[float],
    bias: list[float],
    output: list[float],
    eps: float,
) -> None:
    if not input_:
        raise LayerNormError("empty input")
    if len(weight) < len(input_):
        raise LayerNormError("weight too small")
    if len(bias) < len(input_):
        raise LayerNormError("bias too small")
    if len(output) < len(input_):
        raise LayerNormError("output too small")
    mean = sum(input_) / len(input_)
    variance = sum((v - mean) ** 2 for v in input_) / len(input_)
    inv = 1.0 / math.sqrt(variance + eps)
    for i, v in enumerate(input_):
        output[i] = (v - mean) * inv * weight[i] + bias[i]


def softmax_f32(input_: list[float], output: list[float], dim: int) -> None:
    if dim <= 0 or len(input_) % dim != 0:
        raise SoftmaxError("dim must divide input length")
    batch = len(input_) // dim
    for b in range(batch):
        base = b * dim
        max_val = max(input_[base : base + dim])
        exps = [math.exp(input_[base + i] - max_val) for i in range(dim)]
        total = sum(exps)
        inv = 1.0 / total
        for i in range(dim):
            output[base + i] = exps[i] * inv


def apply_swiglu_f32(gate: list[float], up: list[float], output: list[float]) -> None:
    if len(gate) != len(up):
        raise SwiGluError("gate/up length mismatch")
    if len(output) < len(gate):
        raise SwiGluError("output too small")
    for i, g in enumerate(gate):
        silu = g / (1.0 + math.exp(-g))
        output[i] = silu * up[i]


def linear_activation_f32(
    input_: list[float],
    weight: list[float],
    rows: int,
    shared: int,
    cols: int,
    bias: list[float] | None,
    output: list[float],
    act: ActivationFn,
) -> None:
    gemm = [0.0] * (rows * cols)
    gemm_f32(weight, input_, rows, shared, cols, gemm)
    for r in range(rows):
        for c in range(cols):
            v = gemm[r * cols + c]
            if bias is not None:
                v += bias[r * cols + c]
            if act == ActivationFn.SILU:
                v = v / (1.0 + math.exp(-v))
            elif act == ActivationFn.GELU:
                v = 0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))
            elif act == ActivationFn.RELU:
                v = max(0.0, v)
            output[r * cols + c] = v


def scaled_dot_product_attention_f32(
    query: list[float],
    keys: list[float],
    values: list[float],
    output: list[float],
    seq_len: int,
    head_dim: int,
    scale: float,
) -> None:
    if len(query) < head_dim:
        raise AttentionError("query too small")
    if len(keys) < seq_len * head_dim:
        raise AttentionError("keys too small")
    if len(values) < seq_len * head_dim:
        raise AttentionError("values too small")
    if len(output) < head_dim:
        raise AttentionError("output too small")
    scores = [
        scale * sum(query[d] * keys[s * head_dim + d] for d in range(head_dim))
        for s in range(seq_len)
    ]
    max_score = max(scores)
    exps = [math.exp(s - max_score) for s in scores]
    total = sum(exps)
    weights = [e / total for e in exps]
    for d in range(head_dim):
        output[d] = sum(weights[s] * values[s * head_dim + d] for s in range(seq_len))
