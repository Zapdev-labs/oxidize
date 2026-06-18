"""Model fingerprinting for autotune."""

from __future__ import annotations

from dataclasses import dataclass

from oxidize_python.core.ggufcore import gguf as ggufcore
from oxidize_python.core.model.inference_config import inference_config_from_gguf
from oxidize_python.core.quantization.types import Type, from_ggml_type


@dataclass
class ModelFingerprint:
    architecture: str
    layer_count: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    file_size_bytes: int
    quant: Type
    is_moe: bool = False
    expert_count: int = 0
    has_mtp: bool = False


def fingerprint(mapped: ggufcore.MappedFile) -> ModelFingerprint:
    cfg = inference_config_from_gguf(mapped)
    file_size = len(mapped.bytes)
    quant, is_moe, expert_count, has_mtp = _scan_tensors(mapped.parsed)
    arch = str(cfg.architecture).lower() if cfg.architecture else ggufcore.architecture(mapped.parsed).lower()
    return ModelFingerprint(
        architecture=arch or "llama",
        layer_count=cfg.layer_count,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.kv_head_dim(),
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        file_size_bytes=file_size,
        quant=quant,
        is_moe=is_moe,
        expert_count=expert_count,
        has_mtp=has_mtp,
    )


def fingerprint_from_parts(
    architecture: str,
    layer_count: int,
    hidden_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
    vocab_size: int,
    file_size_bytes: int,
    quant: Type,
) -> ModelFingerprint:
    return ModelFingerprint(
        architecture=architecture,
        layer_count=layer_count,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        file_size_bytes=file_size_bytes,
        quant=quant,
    )


def _scan_tensors(file: ggufcore.GGUFFile) -> tuple[Type, bool, int, bool]:
    hist: dict[int, int] = {}
    is_moe = False
    has_mtp = False
    max_experts = 0
    for t in file.tensor_infos:
        elems = 1
        for d in t.dimensions:
            elems *= int(d)
        hist[t.ggml_type] = hist.get(t.ggml_type, 0) + elems
        name = t.name
        if "_exps" in name or "experts" in name:
            is_moe = True
        if "nextn" in name or "mtp" in name:
            has_mtp = True
        if name.endswith(".ffn_gate_inp.weight") and len(t.dimensions) >= 2:
            max_experts = max(max_experts, int(t.dimensions[-1]))
    best_type = max(hist, key=hist.get) if hist else 0
    return from_ggml_type(best_type), is_moe, max_experts, has_mtp


def kv_bytes_per_token(model: ModelFingerprint, kv_dtype_bytes: int) -> int:
    if model.layer_count == 0 or model.head_dim == 0:
        return 0
    per_layer = model.num_kv_heads * model.head_dim * 2 * kv_dtype_bytes
    return per_layer * model.layer_count


def per_layer_weight_bytes(model: ModelFingerprint) -> int:
    if model.layer_count == 0:
        return 0
    transformer_share = int(model.file_size_bytes * 0.85)
    return transformer_share // model.layer_count


def model_summary(model: ModelFingerprint) -> str:
    moe = f" moe={model.expert_count}" if model.is_moe else ""
    mtp = " mtp=yes" if model.has_mtp else ""
    return (
        f"{model.architecture}-like layers={model.layer_count} hidden={model.hidden_size} "
        f"heads={model.num_attention_heads} kv_heads={model.num_kv_heads} head_dim={model.head_dim} "
        f"vocab={model.vocab_size} size={model.file_size_bytes // (1024 * 1024)} MiB "
        f"quant={model.quant}{moe}{mtp}"
    )
