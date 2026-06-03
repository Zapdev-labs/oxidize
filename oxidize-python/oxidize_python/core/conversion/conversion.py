"""GGUF conversion helpers mirroring oxidize-golang/core/conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from oxidize_python.core.quantization.types import Type
from oxidize_python.internal.gguf.types import File, MetadataValue


class Architecture(StrEnum):
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    GEMMA = "gemma"
    PHI = "phi"
    UNKNOWN = "unknown"


def detect_architecture(metadata: dict[str, MetadataValue]) -> Architecture:
    v = metadata.get("general.architecture")
    if v is None:
        return Architecture.UNKNOWN
    arch = v.string.lower()
    match arch:
        case "llama":
            return Architecture.LLAMA
        case "mistral":
            return Architecture.MISTRAL
        case "qwen" | "qwen2":
            return Architecture.QWEN
        case "gemma":
            return Architecture.GEMMA
        case "phi" | "phi2":
            return Architecture.PHI
        case _:
            return Architecture.UNKNOWN


def map_hf_tensor_name(name: str) -> str:
    if not name.startswith("model.") and not name.startswith("lm_head"):
        return name
    parts = name.split(".")
    if parts[0] == "lm_head":
        if len(parts) >= 2:
            return "output." + ".".join(parts[1:])
        return "output.weight"
    if len(parts) < 2:
        return name
    match parts[1]:
        case "embed_tokens":
            if len(parts) >= 3:
                return "token_embd." + ".".join(parts[2:])
            return "token_embd.weight"
        case "norm":
            return "output_norm.weight"
        case "layers":
            if len(parts) < 4:
                return name
            layer, rest = parts[2], parts[3:]
            if not rest:
                return name
            match rest[0]:
                case "self_attn":
                    if len(rest) < 2:
                        return f"blk.{layer}.attn"
                    child = rest[1]
                    mapping = {
                        "q_proj": f"blk.{layer}.attn_q.weight",
                        "k_proj": f"blk.{layer}.attn_k.weight",
                        "v_proj": f"blk.{layer}.attn_v.weight",
                        "o_proj": f"blk.{layer}.attn_output.weight",
                        "q_norm": f"blk.{layer}.attn_q_norm.weight",
                        "k_norm": f"blk.{layer}.attn_k_norm.weight",
                    }
                    return mapping.get(child, f"blk.{layer}.attn_" + "_".join(rest[1:]))
                case "mlp":
                    if len(rest) < 2:
                        return f"blk.{layer}.ffn"
                    child = rest[1]
                    mapping = {
                        "gate_proj": f"blk.{layer}.ffn_gate.weight",
                        "up_proj": f"blk.{layer}.ffn_up.weight",
                        "down_proj": f"blk.{layer}.ffn_down.weight",
                        "experts": f"blk.{layer}.ffn_gate_exps.weight",
                    }
                    return mapping.get(child, f"blk.{layer}.ffn_" + "_".join(rest[1:]))
                case "input_layernorm":
                    return f"blk.{layer}.attn_norm.weight"
                case "post_attention_layernorm":
                    return f"blk.{layer}.ffn_norm.weight"
    return name


@dataclass
class Plan:
    architecture: Architecture
    tensor_name_map: dict[str, str] = field(default_factory=dict)
    target_quantization: Type | None = None
    special_tokens: dict[str, int] = field(default_factory=dict)


def build_plan(file: File, target: Type | None) -> Plan:
    arch = detect_architecture(file.metadata)
    rename = {info.name: map_hf_tensor_name(info.name) for info in file.tensor_infos}
    special: dict[str, int] = {}
    for key in (
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.unknown_token_id",
    ):
        if tid := parse_special_token_id(file.metadata, key):
            special[key] = tid
    return Plan(
        architecture=arch,
        tensor_name_map=rename,
        target_quantization=target,
        special_tokens=special,
    )


def parse_special_token_id(metadata: dict[str, MetadataValue], key: str) -> int | None:
    v = metadata.get(key)
    if v is None:
        return None
    n, ok = v.as_uint64()
    if ok:
        return int(n)
    if v.string.isdigit():
        return int(v.string)
    return None
