"""Core package mirroring oxidize-golang/core/core.go."""

from oxidize_python.core import kv_cache, quantization, tensor, workspace
from oxidize_python.core.backend import Backend, ComputeBackend
from oxidize_python.core.cpu_kernels.cpu_kernels import implemented_kernels
from oxidize_python.core.ggufcore import MappedFile as MappedGgufFile
from oxidize_python.core.ggufcore import ParseError as GgufParseError
from oxidize_python.core.ggufcore import architecture
from oxidize_python.core.ggufcore import load_mapped as load_gguf_mapped
from oxidize_python.core.kv_cache import Cache as KvCache
from oxidize_python.core.kv_cache import Config as KvCacheConfig
from oxidize_python.core.model import (
    InferenceConfig,
    InferenceModel,
    Model,
    SamplingConfig,
    Token,
    default_sampling_config,
)
from oxidize_python.core.paged.paged import Scheduler
from oxidize_python.core.quantization.types import Type as QuantizationType
from oxidize_python.core.safetensors.safetensors import MappedFile as MappedSafeTensors
from oxidize_python.core.tensor.dtype import DType
from oxidize_python.core.tokenizer import Tokenizer
from oxidize_python.core.validation.validation import implemented_suites
from oxidize_python.core.workspace import (
    WorkspaceHealth,
    benchmark_input,
    health,
    wasm_status,
)


def workspace_health() -> WorkspaceHealth:
    return health()


def implemented_cpu_kernels():
    return implemented_kernels()


def implemented_lora_features() -> list[str]:
    return ["alpha-scaling", "merge-strategies", "rank-budget"]


def implemented_dflash_features() -> list[str]:
    try:
        from oxidize_python.core.model.dflash import implemented_dflash_features

        return [str(f) for f in implemented_dflash_features()]
    except ImportError:
        return []


__all__ = [
    "Backend",
    "ComputeBackend",
    "DType",
    "GgufParseError",
    "InferenceConfig",
    "InferenceModel",
    "KvCache",
    "KvCacheConfig",
    "MappedGgufFile",
    "MappedSafeTensors",
    "Model",
    "QuantizationType",
    "SamplingConfig",
    "Scheduler",
    "Token",
    "Tokenizer",
    "WorkspaceHealth",
    "architecture",
    "benchmark_input",
    "default_sampling_config",
    "health",
    "implemented_cpu_kernels",
    "implemented_dflash_features",
    "implemented_lora_features",
    "implemented_suites",
    "kv_cache",
    "load_gguf_mapped",
    "quantization",
    "tensor",
    "wasm_status",
    "workspace",
    "workspace_health",
]
