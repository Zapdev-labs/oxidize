"""Public layer-cache API for layer-by-layer VRAM management.

Mirrors oxidize-golang/core/backends/cuda/layer_cache.go (port of
oxidize-core/src/backends/cuda/gpu_state.rs's public surface).
"""

from __future__ import annotations

from dataclasses import dataclass

from oxidize_python.core.backends.cuda.gpu_state import (
    enforce_budget,
    evict_layer_internal,
    remove_from_lru,
    touch_layer,
    with_gpu,
)
from oxidize_python.core.backends.cuda.types import (
    SIZE_OF_F32,
    CudaLayerConfig,
    GpuState,
    LayerEntry,
    f32_cache_key,
)


@dataclass
class F32Weight:
    """A single f32 weight matrix passed to ``preload_layer``."""

    data: list[float]
    rows: int = 0
    cols: int = 0


def set_layer_config(config: CudaLayerConfig) -> None:
    """Configure the layer cache budget. Mirrors gpu_state.rs:set_layer_config."""

    def _f(s: GpuState) -> None:
        s.layer_config = config
        enforce_budget(s)

    with_gpu(_f)


def preload_layer(layer: int, weights: list[F32Weight]) -> None:
    """Mark a layer as needed and upload its f32 weights if not resident.

    Evicts LRU layers when over budget. Mirrors gpu_state.rs:preload_layer.
    """

    def _f(s: GpuState) -> None:
        if layer in s.layer_map:
            touch_layer(s, layer)
            return
        entry = LayerEntry()
        for w in weights:
            key = f32_cache_key(w.data)
            if key not in s.resident_f32:
                buf = list(w.data)
                entry.bytes += len(buf) * SIZE_OF_F32
                s.resident_f32[key] = buf
            entry.f32_keys.append(key)
        s.resident_bytes += entry.bytes
        s.layer_map[layer] = entry
        touch_layer(s, layer)

    with_gpu(_f)


def evict_layer(layer: int) -> None:
    """Explicitly evict a layer from VRAM. Mirrors gpu_state.rs:evict_layer."""

    def _f(s: GpuState) -> None:
        remove_from_lru(s, layer)
        evict_layer_internal(s, layer)

    with_gpu(_f)


def resident_vram_bytes() -> int:
    """Report bytes of weight data currently resident on the GPU.

    Mirrors gpu_state.rs:resident_vram_bytes.
    """
    result = {"n": 0}

    def _f(s: GpuState) -> None:
        result["n"] = s.resident_bytes

    with_gpu(_f)
    return result["n"]


def clear_resident_cache() -> None:
    """Clear all resident weight caches (f16, f32, quant, and layer entries).

    Mirrors gpu_state.rs:clear_resident_cache.
    """

    def _f(s: GpuState) -> None:
        s.resident_f16 = {}
        s.resident_f32 = {}
        s.resident_quant = {}
        s.layer_map = {}
        s.layer_lru = []
        s.orphan_f16_keys = []
        s.orphan_quant_keys = []
        s.resident_bytes = 0

    with_gpu(_f)
