"""GGUF-derived inference config mirroring oxidize-golang inference_config.go."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from oxidize_python.core.ggufcore import gguf as ggufcore
from oxidize_python.core.ggufcore.gguf import MappedFile, mapped_tensor_infos
from oxidize_python.core.kv_cache import Quantization as KvQuantization
from oxidize_python.core.model.model import Architecture
from oxidize_python.internal.gguf.types import MetadataValue


class KvCacheDType(StrEnum):
    F32 = "f32"
    F16 = "f16"
    Q8 = "q8"
    Q4 = "q4"


@dataclass
class InferenceConfig:
    vocab_size: int = 32000
    context_size: int = 2048
    layer_count: int = 32
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    key_value_head_dim: int = 128
    kv_cache_dtype: str = "f32"
    kv_quantization: KvQuantization = KvQuantization.ASYMMETRIC
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    architecture: Architecture = Architecture.LLAMA
    sliding_window: int = 0
    num_experts: int = 0
    num_experts_per_token: int = 1
    alibi_num_heads: int = 0
    # Gemma-family fields (see oxidize-core inference.rs).
    rope_theta_swa: float = 0.0
    sliding_window_pattern: int = 0
    embedding_scale: float = 1.0
    gelu_ffn: bool = False
    sandwich_norm: bool = False

    def head_dim(self) -> int:
        if self.num_attention_heads == 0:
            return 0
        return self.hidden_size // self.num_attention_heads

    def kv_head_dim(self) -> int:
        if self.key_value_head_dim != 0:
            return self.key_value_head_dim
        return self.head_dim()


def default_inference_config() -> InferenceConfig:
    return InferenceConfig()


def _meta_uint64(v: MetadataValue | None) -> int | None:
    if v is None:
        return None
    n, ok = v.as_uint64()
    return int(n) if ok else None


def _meta_float32(v: MetadataValue | None) -> float | None:
    if v is None:
        return None
    f, ok = v.as_float32()
    return f if ok else None


def architecture_from_gguf_string(arch: str) -> Architecture:
    match arch:
        case "llama":
            return Architecture.LLAMA
        case "mistral":
            return Architecture.MISTRAL
        case "mixtral":
            return Architecture.MIXTRAL
        case "deepseek" | "deepseek_v2" | "deepseek_v3" | "deepseek_moe":
            return Architecture.DEEPSEEK
        case "qwen" | "qwen2" | "qwen2moe" | "qwen3" | "qwen35":
            return Architecture.QWEN
        case "gemma" | "gemma2" | "gemma3" | "gemma4":
            return Architecture.GEMMA
        case "phi" | "phi2" | "phi3":
            return Architecture.PHI
        case "falcon":
            return Architecture.FALCON
        case "gpt2":
            return Architecture.GPT2
        case "gptj":
            return Architecture.GPTJ
        case "gptneox":
            return Architecture.GPT_NEOX
        case "minimax" | "minimax-m2" | "minimax-text-01":
            return Architecture.MINIMAX
        case _:
            return Architecture.LLAMA


def _apply_token_embedding_dims(out: InferenceConfig, dims: list[int]) -> None:
    """GGUF token_embd dims are [embedding_length, vocab_size] (oxidize-core)."""
    if len(dims) < 2:
        return
    if out.hidden_size == 0:
        out.hidden_size = int(dims[0])
    if out.vocab_size == 0:
        out.vocab_size = int(dims[1])


def inference_config_from_gguf(mapped: MappedFile) -> InferenceConfig:
    file = mapped.parsed
    out = default_inference_config()
    arch = ggufcore.architecture(file) or "llama"
    out.architecture = architecture_from_gguf_string(arch)

    def arch_key(suffix: str) -> str:
        return f"{arch}.{suffix}"

    def arch_u32(suffix: str) -> int | None:
        for key in (arch_key(suffix), f"llama.{suffix}"):
            if n := _meta_uint64(file.metadata.get(key)):
                return n
        return None

    def arch_f32(suffix: str) -> float | None:
        for key in (arch_key(suffix), f"llama.{suffix}"):
            if f := _meta_float32(file.metadata.get(key)):
                return f
        return None

    if n := arch_u32("context_length"):
        out.context_size = n
    if n := arch_u32("embedding_length"):
        out.hidden_size = n
    if n := arch_u32("attention.head_count"):
        out.num_attention_heads = n
    if n := arch_u32("attention.head_count_kv"):
        out.num_key_value_heads = n
    else:
        out.num_key_value_heads = out.num_attention_heads
    if n := arch_u32("attention.key_length"):
        out.key_value_head_dim = n
    if n := arch_u32("feed_forward_length"):
        out.intermediate_size = n
    if n := arch_u32("block_count"):
        out.layer_count = n
    if f := arch_f32("attention.layer_norm_rms_epsilon"):
        out.rms_norm_eps = f
    if f := arch_f32("rope.freq_base"):
        out.rope_theta = f
    if n := arch_u32("vocab_size"):
        out.vocab_size = n
    if n := arch_u32("attention.sliding_window"):
        out.sliding_window = n
    if n := arch_u32("expert_count"):
        out.num_experts = n
    if n := arch_u32("expert_used_count"):
        out.num_experts_per_token = n

    for info in mapped_tensor_infos(file):
        if info.name in (
            "token_embd.weight",
            "tok_embeddings.weight",
            "model.embed_tokens.weight",
        ):
            _apply_token_embedding_dims(out, info.dimensions)
    if out.vocab_size == 0:
        out.vocab_size = 32000

    # Gemma 2/3/4: interleaved local/global attention with dual RoPE theta,
    # embedding scaling, GeGLU, and sandwich normalization. Gated behind the
    # Gemma architecture so other models are unaffected.
    if out.architecture == Architecture.GEMMA:
        pattern = 2 if arch == "gemma2" else 6
        if n := arch_u32("attention.sliding_window_pattern"):
            pattern = n
        out.sliding_window_pattern = pattern
        if f := arch_f32("rope.freq_base_swa"):
            out.rope_theta_swa = f
        else:
            out.rope_theta_swa = 10000.0
        if out.hidden_size > 0:
            out.embedding_scale = math.sqrt(out.hidden_size)
        out.gelu_ffn = True
        out.sandwich_norm = True
    return out


def load_inference_from_gguf(mapped: MappedFile):
    from oxidize_python.core.model.gguf_decoder_load import load_llama_decoder_stack_from_gguf
    from oxidize_python.core.model.inference import InferenceModel, WeightStorage

    config = inference_config_from_gguf(mapped)
    decoder_cfg = llama_decoder_config_from_inference(config)
    stack, err = load_llama_decoder_stack_from_gguf(mapped, decoder_cfg)
    if err:
        raise err
    if stack.output.is_loaded() and config.vocab_size == 0:
        config.vocab_size = stack.output._output_dim()
    if stack.tok_embeddings.is_loaded() and config.vocab_size == 0:
        config.vocab_size = stack.tok_embeddings._output_dim()
    return InferenceModel(config, WeightStorage(file=mapped), stack)


def llama_decoder_config_from_inference(cfg: InferenceConfig):
    from oxidize_python.core.model.llama_decoder import LlamaDecoderConfig

    return LlamaDecoderConfig(
        hidden_size=cfg.hidden_size,
        layer_count=cfg.layer_count,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        key_value_head_dim=cfg.kv_head_dim(),
        vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        sliding_window=cfg.sliding_window,
        rope_theta_swa=cfg.rope_theta_swa,
        sliding_window_pattern=cfg.sliding_window_pattern,
        embedding_scale=cfg.embedding_scale,
        gelu_ffn=cfg.gelu_ffn,
        sandwich_norm=cfg.sandwich_norm,
    )
