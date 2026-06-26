"""Temporal encoder mirroring oxidize-golang/core/video/temporal.go."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from oxidize_python.core.tensor.gemv import gemm_f32
from oxidize_python.core.tensor.ops import apply_rope_head_f32, rms_norm_f32
from oxidize_python.core.video.config import TemporalConfig
from oxidize_python.core.video.errors import VideoError


@dataclass
class TemporalLayerWeights:
    """Weights of one temporal self-attention layer.

    Mirrors temporal.rs:TemporalLayerWeights.
    """

    attn_norm: list[float]
    q_proj: list[float]
    k_proj: list[float]
    v_proj: list[float]
    o_proj: list[float]
    ffn_norm: list[float]
    ffn_gate: list[float]
    ffn_up: list[float]
    ffn_down: list[float]


def zero_temporal_layer_weights(cfg: TemporalConfig) -> TemporalLayerWeights:
    """Build zero-initialized layer weights (norms are ones)."""
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    return TemporalLayerWeights(
        attn_norm=[1.0] * h,
        q_proj=[0.0] * (h * h),
        k_proj=[0.0] * (h * h),
        v_proj=[0.0] * (h * h),
        o_proj=[0.0] * (h * h),
        ffn_norm=[1.0] * h,
        ffn_gate=[0.0] * (h * inter),
        ffn_up=[0.0] * (h * inter),
        ffn_down=[0.0] * (inter * h),
    )


@dataclass
class TemporalWeights:
    """Full temporal encoder weights. Mirrors temporal.rs:TemporalWeights."""

    layers: list[TemporalLayerWeights] = field(default_factory=list)
    final_norm: list[float] = field(default_factory=list)
    # Optional learnable token (len hidden_size). Empty when use_cls_token is False.
    cls_token: list[float] = field(default_factory=list)


def zero_temporal_weights(cfg: TemporalConfig) -> TemporalWeights:
    """Build zero-initialized weights for cfg."""
    layers = [zero_temporal_layer_weights(cfg) for _ in range(cfg.num_layers)]
    final_norm = [1.0] * cfg.hidden_size
    cls = [0.0] * cfg.hidden_size if cfg.use_cls_token else []
    return TemporalWeights(layers=layers, final_norm=final_norm, cls_token=cls)


@dataclass
class TemporalWorkspace:
    """Persistent scratch buffers for the temporal encoder.

    Allocate once via new_temporal_workspace and reuse across calls. Mirrors
    temporal.rs:TemporalWorkspace.
    """

    hidden: list[float]
    residual: list[float]
    qkv: list[float]
    attn: list[float]
    attn_softmax: list[float]
    attn_out: list[float]
    ffn_gate: list[float]
    ffn_up: list[float]
    ffn_silu: list[float]
    ffn_down: list[float]
    normed: list[float]
    proj_out: list[float]
    rope: list[float]


def new_temporal_workspace(cfg: TemporalConfig) -> TemporalWorkspace:
    """Allocate all scratch buffers for cfg's worst case."""
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    seq = cfg.max_frames + 1
    attn_size = cfg.num_heads * seq * seq
    return TemporalWorkspace(
        hidden=[0.0] * (seq * h),
        residual=[0.0] * (seq * h),
        qkv=[0.0] * (seq * 3 * h),
        attn=[0.0] * attn_size,
        attn_softmax=[0.0] * attn_size,
        attn_out=[0.0] * (seq * h),
        ffn_gate=[0.0] * (seq * inter),
        ffn_up=[0.0] * (seq * inter),
        ffn_silu=[0.0] * (seq * inter),
        ffn_down=[0.0] * (seq * h),
        normed=[0.0] * (seq * h),
        proj_out=[0.0] * (seq * h),
        rope=[0.0] * h,
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def forward_temporal(
    cfg: TemporalConfig,
    weights: TemporalWeights,
    input_: list[float],
    input_seq_len: int,
    ws: TemporalWorkspace | None = None,
) -> list[float]:
    """Run the temporal encoder over a [input_seq_len, hidden] input matrix.

    Returns the normalized [seq_len, hidden] output (seq_len includes the cls
    token when enabled). Mirrors temporal.rs:forward_temporal.
    """
    cfg.validate()
    if input_seq_len == 0 or input_seq_len > cfg.max_frames:
        raise VideoError(
            f"frame count {input_seq_len} out of range [1, {cfg.max_frames}]"
        )
    h = cfg.hidden_size
    if len(input_) != input_seq_len * h:
        raise VideoError(
            f"input buffer length {len(input_)} does not match "
            f"seq_len*hidden ({input_seq_len}*{h})"
        )
    if len(weights.layers) != cfg.num_layers:
        raise VideoError(
            f"temporal_layers shape mismatch: expected {cfg.num_layers} "
            f"got {len(weights.layers)}"
        )

    use_cls = len(weights.cls_token) != 0
    seq_len = input_seq_len + 1 if use_cls else input_seq_len
    if ws is None:
        ws = new_temporal_workspace(cfg)

    hidden = ws.hidden
    if use_cls:
        hidden[:h] = weights.cls_token[:h]
        hidden[h : h + input_seq_len * h] = input_[: input_seq_len * h]
    else:
        hidden[: seq_len * h] = input_[: seq_len * h]

    for layer_idx, layer in enumerate(weights.layers):
        try:
            _forward_temporal_layer(cfg, layer, hidden, seq_len, ws)
        except VideoError as exc:
            raise VideoError(f"layer {layer_idx}: {exc}") from exc

    out = [0.0] * (seq_len * h)
    for row in range(seq_len):
        rms_norm_f32(
            hidden[row * h : (row + 1) * h],
            weights.final_norm,
            _view := [0.0] * h,
            cfg.rms_norm_eps,
        )
        out[row * h : (row + 1) * h] = _view
    return out


def _forward_temporal_layer(
    cfg: TemporalConfig,
    layer: TemporalLayerWeights,
    hidden: list[float],
    seq_len: int,
    ws: TemporalWorkspace,
) -> None:
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    head_dim = cfg.head_dim()
    if head_dim == 0:
        raise VideoError("head_dim must be non-zero")
    if (
        len(layer.q_proj) != h * h
        or len(layer.k_proj) != h * h
        or len(layer.v_proj) != h * h
        or len(layer.o_proj) != h * h
    ):
        raise VideoError("QKV/O projection shape mismatch")
    if (
        len(layer.ffn_gate) != h * inter
        or len(layer.ffn_up) != h * inter
        or len(layer.ffn_down) != inter * h
    ):
        raise VideoError("FFN projection shape mismatch")

    # ---- Pre-norm + QKV + attention ----
    residual = hidden[: seq_len * h]
    normed = [0.0] * (seq_len * h)
    for row in range(seq_len):
        view = [0.0] * h
        rms_norm_f32(hidden[row * h : (row + 1) * h], layer.attn_norm, view, cfg.rms_norm_eps)
        normed[row * h : (row + 1) * h] = view

    q_part = [0.0] * (seq_len * h)
    k_part = [0.0] * (seq_len * h)
    v_part = [0.0] * (seq_len * h)
    gemm_f32(normed, layer.q_proj, seq_len, h, h, q_part)
    gemm_f32(normed, layer.k_proj, seq_len, h, h, k_part)
    gemm_f32(normed, layer.v_proj, seq_len, h, h, v_part)

    # Apply RoPE to Q and K along the time axis (position = row index).
    rope = [0.0] * head_dim
    for pos in range(seq_len):
        for head in range(cfg.num_heads):
            start = pos * h + head * head_dim
            end = start + head_dim
            apply_rope_head_f32(q_part[start:end], rope, pos, head_dim, cfg.rope_theta)
            q_part[start:end] = rope[:head_dim]
            apply_rope_head_f32(k_part[start:end], rope, pos, head_dim, cfg.rope_theta)
            k_part[start:end] = rope[:head_dim]

    attn_out = [0.0] * (seq_len * h)
    _compute_causal_attention(
        q_part, k_part, v_part, attn_out, seq_len, h, cfg.num_heads, head_dim
    )

    # Output projection + residual.
    proj_out = [0.0] * (seq_len * h)
    gemm_f32(attn_out, layer.o_proj, seq_len, h, h, proj_out)
    for i in range(seq_len * h):
        hidden[i] = residual[i] + proj_out[i]

    # ---- FFN block ----
    residual = hidden[: seq_len * h]
    for row in range(seq_len):
        view = [0.0] * h
        rms_norm_f32(hidden[row * h : (row + 1) * h], layer.ffn_norm, view, cfg.rms_norm_eps)
        normed[row * h : (row + 1) * h] = view

    gate = [0.0] * (seq_len * inter)
    up = [0.0] * (seq_len * inter)
    silu = [0.0] * (seq_len * inter)
    down = [0.0] * (seq_len * h)
    gemm_f32(normed, layer.ffn_gate, seq_len, h, inter, gate)
    gemm_f32(normed, layer.ffn_up, seq_len, h, inter, up)
    for i in range(seq_len * inter):
        g = gate[i]
        silu[i] = g * _sigmoid(g) * up[i]
    gemm_f32(silu, layer.ffn_down, seq_len, inter, h, down)
    for i in range(seq_len * h):
        hidden[i] = residual[i] + down[i]


def _compute_causal_attention(
    q: list[float],
    k: list[float],
    v: list[float],
    output: list[float],
    seq_len: int,
    hidden: int,
    num_heads: int,
    head_dim: int,
) -> None:
    """Per-head causal self-attention over a [seq, hidden] matrix.

    Mirrors temporal.rs:compute_causal_attention.
    """
    scale = 1.0 / math.sqrt(head_dim)
    for h_idx in range(num_heads):
        for q_pos in range(seq_len):
            q_off = q_pos * hidden + h_idx * head_dim
            q_row = q[q_off : q_off + head_dim]

            scores = [0.0] * (q_pos + 1)
            max_score = -math.inf
            for k_pos in range(q_pos + 1):
                k_off = k_pos * hidden + h_idx * head_dim
                dot = 0.0
                for d in range(head_dim):
                    dot += q_row[d] * k[k_off + d]
                s = dot * scale
                scores[k_pos] = s
                if s > max_score:
                    max_score = s
            if math.isinf(max_score) or math.isnan(max_score):
                max_score = 0.0

            total = 0.0
            for k_pos in range(q_pos + 1):
                p = math.exp(scores[k_pos] - max_score)
                scores[k_pos] = p
                total += p
            inv_sum = 1.0 / total if total > 0.0 else 1.0

            out_off = q_pos * hidden + h_idx * head_dim
            for d in range(head_dim):
                output[out_off + d] = 0.0
            for k_pos in range(q_pos + 1):
                v_off = k_pos * hidden + h_idx * head_dim
                a = scores[k_pos] * inv_sum
                for d in range(head_dim):
                    output[out_off + d] += a * v[v_off + d]
