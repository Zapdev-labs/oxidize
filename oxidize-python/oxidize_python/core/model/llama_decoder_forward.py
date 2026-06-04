"""Decoder forward pass mirroring oxidize-golang/core/model/llama_decoder_forward.go."""

from __future__ import annotations

import math
from collections.abc import Callable

from oxidize_python.core.flash_attention.flash_attention import flash_attention_decode_heads_gqa
from oxidize_python.core.model.llama_decoder import LlamaDecoderStack
from oxidize_python.core.model.model import Logits
from oxidize_python.core.tensor.ops import apply_rope_head_f32, rms_norm_f32


def forward_token(stack: LlamaDecoderStack, token: int) -> list[float]:
    return forward_token_with_context(stack, token, None, None)


def forward_token_with_context(
    stack: LlamaDecoderStack,
    token: int,
    target_context: list[float] | None,
    kv_context: Callable[[int, list[float]], tuple[list[float], list[float]]] | None,
) -> list[float]:
    h = stack.config.hidden_size
    hidden = [0.0] * h
    stack.fill_token_embedding(token, hidden)

    for layer_idx, layer in enumerate(stack.layers):
        head_dim = stack.config.kv_head_dim()
        if layer.attention.q_norm_weight:
            head_dim = len(layer.attention.q_norm_weight)
        num_heads = stack.config.num_attention_heads
        num_kv = stack.config.num_key_value_heads
        q_size = num_heads * head_dim
        kv_len = num_kv * head_dim

        attn_out = [0.0] * q_size
        mlp_out = [0.0] * h

        normed = [0.0] * h
        rms_norm_f32(hidden, layer.input_layernorm, normed, stack.config.rms_norm_eps)

        q = [0.0] * q_size
        k = [0.0] * kv_len
        v = [0.0] * kv_len
        if layer.attention.q_proj.is_loaded():
            layer.attention.q_proj.gemv(normed, q)
        if layer.attention.k_proj.is_loaded():
            layer.attention.k_proj.gemv(normed, k)
        if layer.attention.v_proj.is_loaded():
            layer.attention.v_proj.gemv(normed, v)

        k_ctx: list[float] | None = None
        v_ctx: list[float] | None = None
        if kv_context is not None and target_context is not None:
            k_ctx, v_ctx = kv_context(layer_idx, target_context)

        head_scratch = [0.0] * head_dim
        pos = stack.position_offset

        def apply_qk_norm(vec: list[float], weight: list[float], heads: int) -> None:
            if len(weight) != head_dim:
                return
            for hi in range(heads):
                start = hi * head_dim
                rms_norm_f32(
                    vec[start : start + head_dim],
                    weight,
                    head_scratch,
                    stack.config.rms_norm_eps,
                )
                vec[start : start + head_dim] = head_scratch[:]

        apply_qk_norm(q, layer.attention.q_norm_weight, num_heads)
        apply_qk_norm(k, layer.attention.k_norm_weight, num_kv)
        if k_ctx is not None:
            apply_qk_norm(k_ctx, layer.attention.k_norm_weight, num_kv)

        def apply_rope(vec: list[float], heads: int) -> None:
            for hi in range(heads):
                start = hi * head_dim
                apply_rope_head_f32(
                    vec[start : start + head_dim],
                    head_scratch,
                    pos,
                    head_dim,
                    stack.config.rope_theta,
                )
                vec[start : start + head_dim] = head_scratch[:]

        apply_rope(q, num_heads)
        apply_rope(k, num_kv)
        if k_ctx is not None:
            apply_rope(k_ctx, num_kv)

        cache = stack.kv_cache[layer_idx]
        if k_ctx is not None and v_ctx is not None:
            cache.keys.extend(k_ctx)
            cache.values.extend(v_ctx)
            cache.seq_len += 1
        cache.keys.extend(k)
        cache.values.extend(v)
        cache.seq_len += 1

        flash_attention_decode_heads_gqa(
            q,
            cache.keys,
            cache.values,
            attn_out,
            cache.seq_len,
            head_dim,
            kv_len,
            num_heads,
            num_kv,
        )

        o_result = [0.0] * h
        if layer.attention.o_proj.is_loaded():
            layer.attention.o_proj.gemv(attn_out, o_result)
        else:
            o_result[: min(h, q_size)] = attn_out[: min(h, q_size)]
            for i in range(q_size, h):
                o_result[i] = 0.0
        attn_out = o_result

        for i in range(h):
            hidden[i] += attn_out[i]

        normed = [0.0] * h
        rms_norm_f32(hidden, layer.post_attention_layernorm, normed, stack.config.rms_norm_eps)
        inter = stack.config.intermediate_size
        gate = [0.0] * inter
        up = [0.0] * inter
        if layer.mlp_gate.is_loaded():
            layer.mlp_gate.gemv(normed, gate)
        if layer.mlp_up.is_loaded():
            layer.mlp_up.gemv(normed, up)
        for i in range(inter):
            g = gate[i]
            gate[i] = (g / (1.0 + math.exp(-g))) * up[i]
        if layer.mlp_down.is_loaded():
            layer.mlp_down.gemv(gate, mlp_out)

        for i in range(h):
            hidden[i] += mlp_out[i]

    if stack.norm:
        out = [0.0] * h
        rms_norm_f32(hidden, stack.norm, out, stack.config.rms_norm_eps)
        hidden = out

    stack.position_offset += 1
    return hidden


def forward_batch(stack: LlamaDecoderStack, tokens: list[int]) -> list[float]:
    if not tokens:
        raise ValueError("empty token batch")
    if len(tokens) == 1:
        return forward_token(stack, tokens[0])

    b = len(tokens)
    h = stack.config.hidden_size
    hidden = [0.0] * (b * h)

    if stack.tok_embeddings.is_loaded():
        for t, tok in enumerate(tokens):
            stack.fill_token_embedding(tok, hidden[t * h : (t + 1) * h])

    for layer_idx, layer in enumerate(stack.layers):
        head_dim = stack.config.kv_head_dim()
        if layer.attention.q_norm_weight:
            head_dim = len(layer.attention.q_norm_weight)
        num_heads = stack.config.num_attention_heads
        num_kv = stack.config.num_key_value_heads
        q_size = num_heads * head_dim
        kv_len = num_kv * head_dim

        normed = [0.0] * (b * h)
        for t in range(b):
            rms_norm_f32(
                hidden[t * h : (t + 1) * h],
                layer.input_layernorm,
                normed[t * h : (t + 1) * h],
                stack.config.rms_norm_eps,
            )

        q_all = [0.0] * (b * q_size)
        k_all = [0.0] * (b * kv_len)
        v_all = [0.0] * (b * kv_len)
        if layer.attention.q_proj.is_loaded():
            layer.attention.q_proj.gemm(normed, q_all, b)
        if layer.attention.k_proj.is_loaded():
            layer.attention.k_proj.gemm(normed, k_all, b)
        if layer.attention.v_proj.is_loaded():
            layer.attention.v_proj.gemm(normed, v_all, b)

        head_scratch = [0.0] * head_dim
        for t in range(b):
            pos = stack.position_offset + t
            q = q_all[t * q_size : (t + 1) * q_size]
            k = k_all[t * kv_len : (t + 1) * kv_len]
            if len(layer.attention.q_norm_weight) == head_dim:
                for hi in range(num_heads):
                    start = hi * head_dim
                    rms_norm_f32(
                        q[start : start + head_dim],
                        layer.attention.q_norm_weight,
                        head_scratch,
                        stack.config.rms_norm_eps,
                    )
                    q[start : start + head_dim] = head_scratch[:]
            if len(layer.attention.k_norm_weight) == head_dim:
                for hi in range(num_kv):
                    start = hi * head_dim
                    rms_norm_f32(
                        k[start : start + head_dim],
                        layer.attention.k_norm_weight,
                        head_scratch,
                        stack.config.rms_norm_eps,
                    )
                    k[start : start + head_dim] = head_scratch[:]
            for hi in range(num_heads):
                start = hi * head_dim
                apply_rope_head_f32(
                    q[start : start + head_dim],
                    head_scratch,
                    pos,
                    head_dim,
                    stack.config.rope_theta,
                )
                q[start : start + head_dim] = head_scratch[:]
            for hi in range(num_kv):
                start = hi * head_dim
                apply_rope_head_f32(
                    k[start : start + head_dim],
                    head_scratch,
                    pos,
                    head_dim,
                    stack.config.rope_theta,
                )
                k[start : start + head_dim] = head_scratch[:]

        attn_pre_o = [0.0] * (b * q_size)
        cache = stack.kv_cache[layer_idx]
        for t in range(b):
            k = k_all[t * kv_len : (t + 1) * kv_len]
            v = v_all[t * kv_len : (t + 1) * kv_len]
            cache.keys.extend(k)
            cache.values.extend(v)
            cache.seq_len += 1
            q = q_all[t * q_size : (t + 1) * q_size]
            out = attn_pre_o[t * q_size : (t + 1) * q_size]
            flash_attention_decode_heads_gqa(
                q,
                cache.keys,
                cache.values,
                out,
                cache.seq_len,
                head_dim,
                kv_len,
                num_heads,
                num_kv,
            )

        attn_out_all = [0.0] * (b * h)
        if layer.attention.o_proj.is_loaded():
            layer.attention.o_proj.gemm(attn_pre_o, attn_out_all, b)
        elif q_size == h:
            attn_out_all[:] = attn_pre_o
        for i in range(len(hidden)):
            hidden[i] += attn_out_all[i]

        normed_mlp = [0.0] * (b * h)
        for t in range(b):
            rms_norm_f32(
                hidden[t * h : (t + 1) * h],
                layer.post_attention_layernorm,
                normed_mlp[t * h : (t + 1) * h],
                stack.config.rms_norm_eps,
            )
        inter = stack.config.intermediate_size
        gate = [0.0] * (b * inter)
        up = [0.0] * (b * inter)
        if layer.mlp_gate.is_loaded():
            layer.mlp_gate.gemm(normed_mlp, gate, b)
        if layer.mlp_up.is_loaded():
            layer.mlp_up.gemm(normed_mlp, up, b)
        for i in range(len(gate)):
            g = gate[i]
            gate[i] = (g / (1.0 + math.exp(-g))) * up[i]
        mlp_out_all = [0.0] * (b * h)
        if layer.mlp_down.is_loaded():
            layer.mlp_down.gemm(gate, mlp_out_all, b)
        for i in range(len(hidden)):
            hidden[i] += mlp_out_all[i]

    out_hidden = [0.0] * (b * h)
    for t in range(b):
        rms_norm_f32(
            hidden[t * h : (t + 1) * h],
            stack.norm,
            out_hidden[t * h : (t + 1) * h],
            stack.config.rms_norm_eps,
        )
    stack.position_offset += b
    return list(out_hidden[(b - 1) * h : b * h])


def logits(stack: LlamaDecoderStack, hidden: list[float]) -> Logits:
    if not stack.output.is_loaded():
        raise ValueError("decoder stack is missing output projection")
    vocab = stack.output._output_dim()
    in_dim = stack.output._input_dim()
    if len(hidden) < in_dim:
        raise ValueError(f"hidden width {len(hidden)} smaller than output input width {in_dim}")
    out: Logits = [0.0] * vocab
    stack.output.gemv(hidden[:in_dim], out)
    return out
