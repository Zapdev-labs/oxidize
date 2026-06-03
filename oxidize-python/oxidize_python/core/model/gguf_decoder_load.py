"""GGUF weight loading for Llama decoder mirroring gguf_decoder_load.go."""

from __future__ import annotations

from dataclasses import dataclass

from oxidize_python.core.ggufcore.gguf import MappedFile, mapped_tensor_infos
from oxidize_python.core.model.dflash_weights import (
    F32Weight,
    f32_weight_from_dims,
    ones,
    quantized_gemv_supported,
    transpose_f32,
)
from oxidize_python.core.model.llama_decoder import (
    DecoderAttentionLayer,
    DecoderLayer,
    LlamaDecoderConfig,
    LlamaDecoderStack,
)
from oxidize_python.core.quantization import types as quant
from oxidize_python.core.quantization.dequant_k import dequantize
from oxidize_python.internal.gguf.types import TensorInfo


@dataclass
class _GgufWeightLoader:
    infos: list[TensorInfo]
    bytes_all: bytes
    config: LlamaDecoderConfig

    def _find(self, name: str) -> TensorInfo | None:
        for info in self.infos:
            if info.name == name:
                return info
        return None

    def load_f32_with_dims(self, name: str) -> tuple[list[float], list[int], bool]:
        info = self._find(name)
        if info is None:
            return [], [], False
        qtype = quant.from_ggml_type(info.ggml_type)
        count = 1
        for d in info.dimensions:
            count *= int(d)
        qsize = quant.quantized_size(qtype, count)
        off = info.absolute_offset
        end = off + qsize
        if end > len(self.bytes_all):
            raise ValueError(f"tensor {name} out of bounds")
        out = [0.0] * count
        dequantize(qtype, self.bytes_all[off:end], out)
        return out, list(info.dimensions), True

    def load_proj(self, name: str) -> F32Weight:
        info = self._find(name)
        if info is None:
            return F32Weight([], 0, 0)
        if len(info.dimensions) != 2:
            data, dims, ok = self.load_f32_with_dims(name)
            if not ok:
                return F32Weight([], 0, 0)
            return f32_weight_from_dims(data, dims)
        qtype = quant.from_ggml_type(info.ggml_type)
        in_dim = int(info.dimensions[0])
        out_dim = int(info.dimensions[1])
        if quantized_gemv_supported(qtype, in_dim):
            count = out_dim * in_dim
            qsize = quant.quantized_size(qtype, count)
            off = info.absolute_offset
            end = off + qsize
            if end > len(self.bytes_all):
                raise ValueError(f"tensor {name} out of bounds")
            raw = bytes(self.bytes_all[off:end])
            return F32Weight.from_quantized(raw, qtype, out_dim, in_dim)
        data, _, ok = self.load_f32_with_dims(name)
        if not ok:
            return F32Weight([], 0, 0)
        return F32Weight.from_slice(transpose_f32(data, in_dim, out_dim), out_dim, in_dim)

    def load_proj_any(self, *names: str) -> F32Weight:
        for name in names:
            w = self.load_proj(name)
            if w.is_loaded():
                return w
        return F32Weight([], 0, 0)

    def load_f32_any(self, *names: str) -> list[float] | None:
        for name in names:
            data, _, ok = self.load_f32_with_dims(name)[:3]
            if ok:
                return data
        return None


def load_llama_decoder_stack_from_gguf(
    mapped: MappedFile,
    config: LlamaDecoderConfig,
) -> tuple[LlamaDecoderStack, Exception | None]:
    try:
        stack = LlamaDecoderStack.new(config)
        loader = _GgufWeightLoader(
            mapped_tensor_infos(mapped.parsed),
            mapped.bytes,
            config,
        )
        norm = loader.load_f32_any("output_norm.weight", "norm.weight", "model.norm.weight")
        if norm:
            stack.norm = norm
        stack.output = loader.load_proj_any("lm_head.weight", "output.weight")
        stack.tok_embeddings = loader.load_proj_any(
            "model.embed_tokens.weight",
            "tok_embeddings.weight",
            "token_embd.weight",
        )
        if stack.tok_embeddings.is_loaded() and config.vocab_size == 0:
            stack.config.vocab_size = stack.tok_embeddings._output_dim()
        if not stack.output.is_loaded() and stack.tok_embeddings.is_loaded():
            stack.output = stack.tok_embeddings

        for layer_idx in range(config.layer_count):
            prefix = f"blk.{layer_idx}"
            in_ln = ones(config.hidden_size)
            data, _, ok = loader.load_f32_with_dims(f"{prefix}.attn_norm.weight")
            if ok:
                in_ln = data
            post_ln = ones(config.hidden_size)
            data, _, ok = loader.load_f32_with_dims(f"{prefix}.post_attention_norm.weight")
            if ok:
                post_ln = data
            else:
                data, _, ok = loader.load_f32_with_dims(f"{prefix}.ffn_norm.weight")
                if ok:
                    post_ln = data
            q_norm = ones(config.kv_head_dim() or config.head_dim())
            data, _, ok = loader.load_f32_with_dims(f"{prefix}.attn_q_norm.weight")
            if ok:
                q_norm = data
            k_norm = ones(config.kv_head_dim() or config.head_dim())
            data, _, ok = loader.load_f32_with_dims(f"{prefix}.attn_k_norm.weight")
            if ok:
                k_norm = data
            stack.layers.append(
                DecoderLayer(
                    input_layernorm=in_ln,
                    post_attention_layernorm=post_ln,
                    mlp_gate=loader.load_proj(f"{prefix}.ffn_gate.weight"),
                    mlp_up=loader.load_proj(f"{prefix}.ffn_up.weight"),
                    mlp_down=loader.load_proj(f"{prefix}.ffn_down.weight"),
                    attention=DecoderAttentionLayer(
                        q_proj=loader.load_proj(f"{prefix}.attn_q.weight"),
                        k_proj=loader.load_proj(f"{prefix}.attn_k.weight"),
                        v_proj=loader.load_proj(f"{prefix}.attn_v.weight"),
                        o_proj=loader.load_proj(f"{prefix}.attn_output.weight"),
                        q_norm_weight=q_norm,
                        k_norm_weight=k_norm,
                    ),
                )
            )
        return stack, None
    except Exception as err:
        return LlamaDecoderStack.new(config), err
