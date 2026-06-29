"""CUDA backend types mirroring oxidize-golang/core/backends/cuda/types.go.

Defines the GGML type ids used to dispatch quantized GEMV kernels, the
layer-cache configuration, and the GpuState bundle of resident weight caches,
buffer pools, and activation buffers.

In the pure-Python build the "device" buffers are host-side ``list[float]`` /
``bytearray`` slices; the bookkeeping (LRU eviction, byte budgets, pooling)
matches the Rust/Go ports exactly so the same integers flow through unchanged.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from threading import RLock


class GgmlType(IntEnum):
    """Numeric GGML tensor type ids used to dispatch quantized GEMV kernels.

    Values match the upstream GGML enum so the same integer flows through the
    Rust, Go, and Python ports unchanged.
    """

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q8_0 = 8
    Q2_K = 10
    Q4_K = 12
    Q6_K = 14


@dataclass(frozen=True)
class DequantKernel:
    """On-the-fly dequant kernel descriptor for a quantized type.

    ``name`` is informational (mirrors Rust), ``block_bytes`` is the raw block
    size in bytes, and ``vals_per_blk`` is how many decoded f32 values a block
    produces.
    """

    name: str
    block_bytes: int
    vals_per_blk: int


def dequant_kernel_for(t: GgmlType) -> DequantKernel | None:
    """Mirror gemv_quantized.rs:dequant_kernel_for.

    Returns the kernel descriptor for a type, or ``None`` when no GPU dequant
    path exists and the caller must fall back to the CPU quantized path.
    """
    table = {
        GgmlType.Q8_0: DequantKernel("dequant_q8_0_kernel", 34, 32),
        GgmlType.Q4_K: DequantKernel("dequant_q4_k_kernel", 144, 256),
        GgmlType.Q6_K: DequantKernel("dequant_q6_k_kernel", 210, 256),
        GgmlType.Q2_K: DequantKernel("dequant_q2_k_kernel", 84, 256),
        GgmlType.Q4_0: DequantKernel("dequant_q4_0_kernel", 18, 32),
    }
    return table.get(GgmlType(t))


def supports_quantized_gpu(t: GgmlType) -> bool:
    """Mirror gemv_quantized.rs:supports_quantized_gpu.

    Callers should fall back to the CPU quantized path when this returns False.
    """
    return dequant_kernel_for(t) is not None


# LayerID tags a group of weight matrices as belonging to the same model layer.
LayerID = int


@dataclass
class CudaLayerConfig:
    """Layer-by-layer VRAM management config (AirLLM-style).

    Mirrors types.rs:CudaLayerConfig.

    ``max_resident_layers``: maximum number of layers to keep resident in VRAM
    at once. 0 = unlimited (default, loads all layers).
    ``max_vram_bytes``: maximum VRAM bytes to use for weight caching.
    0 = unlimited (default).
    """

    max_resident_layers: int = 0
    max_vram_bytes: int = 0


@dataclass
class LayerEntry:
    """Resident weight keys owned by a layer and the bytes they consume.

    Mirrors types.rs:LayerEntry.
    """

    f32_keys: list[WeightCacheKey] = field(default_factory=list)
    f16_keys: list[WeightCacheKey] = field(default_factory=list)
    bytes: int = 0


@dataclass(frozen=True)
class WeightCacheKey:
    """Identifies a resident weight matrix.

    The Rust port keys by (pointer, len, content_hash); in pure-Python we key by
    a stable content hash plus length so identical host buffers map to the same
    resident entry.
    """

    hash: int
    len: int


@dataclass
class GpuActivationBuffer:
    """GPU-resident activation buffers for a single decode step.

    Allocated once and reused across tokens. Mirrors
    types.rs:GpuActivationBuffer. In the pure-Python build these are host-side
    lists.
    """

    hidden: list[float]
    normed: list[float]
    ffn_gate: list[float]
    ffn_up: list[float]
    ffn_down_in: list[float]
    hidden_size: int
    intermediate_size: int


SIZE_OF_F32 = 4
SIZE_OF_U16 = 2


def f32_cache_key(data: list[float]) -> WeightCacheKey:
    """Derive a stable cache key from an f32 weight matrix.

    Uses the length and an FNV-1a hash of its IEEE-754 bit pattern, matching
    gpu_state.go:f32CacheKey.
    """
    h = _Fnv1a()
    for v in data:
        h.write(struct.pack("<f", v))
    return WeightCacheKey(hash=h.sum(), len=len(data))


def byte_cache_key(data: bytes | bytearray) -> WeightCacheKey:
    """Derive a stable cache key from raw quantized bytes."""
    h = _Fnv1a()
    h.write(bytes(data))
    return WeightCacheKey(hash=h.sum(), len=len(data))


class _Fnv1a:
    """64-bit FNV-1a hasher matching Go's hash/fnv New64a output."""

    _OFFSET = 0xCBF29CE484222325
    _PRIME = 0x100000001B3
    _MASK = 0xFFFFFFFFFFFFFFFF

    def __init__(self) -> None:
        self._h = self._OFFSET

    def write(self, data: bytes) -> None:
        h = self._h
        for b in data:
            h ^= b
            h = (h * self._PRIME) & self._MASK
        self._h = h

    def sum(self) -> int:
        return self._h


class GpuState:
    """Per-context bundle of resident weight caches, buffer pools, layer state.

    Mirrors types.rs:GpuState. In the pure-Python build the "device" buffers are
    host lists; the bookkeeping for LRU / budget logic matches the Rust and Go
    ports.
    """

    def __init__(self) -> None:
        self.lock = RLock()

        self.resident_f32: dict[WeightCacheKey, list[float]] = {}
        self.resident_f16: dict[WeightCacheKey, list[int]] = {}
        self.resident_quant: dict[WeightCacheKey, bytearray] = {}

        self.f32_pool: dict[int, list[list[float]]] = {}
        self.q8k_pool: dict[int, list[bytearray]] = {}

        self.layer_config = CudaLayerConfig()
        self.layer_lru: list[LayerID] = []
        self.layer_map: dict[LayerID, LayerEntry] = {}

        self.resident_bytes: int = 0

        self.orphan_f16_keys: list[WeightCacheKey] = []
        self.orphan_quant_keys: list[WeightCacheKey] = []

        self.activation: GpuActivationBuffer | None = None
