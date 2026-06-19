"""Process-wide GPU state, buffer pools, and layer LRU/budget management.

Mirrors oxidize-golang/core/backends/cuda/gpu_state.go (itself a port of
oxidize-core/src/backends/cuda/gpu_state.rs).

The Rust port keys this off ``thread_local!`` because each thread owns a CUDA
context. Python threads share interpreter state and the GIL, so this port keeps
a single process-wide state guarded by a re-entrant lock. This avoids
duplicating weight caches per thread while preserving the same LRU/budget
semantics.
"""

from __future__ import annotations

import math
from threading import Lock
from typing import Callable, TypeVar

from oxidize_python.core.backends.cuda.types import (
    SIZE_OF_F32,
    SIZE_OF_U16,
    GpuState,
    WeightCacheKey,
)

_state_lock = Lock()
_global_state: GpuState | None = None

T = TypeVar("T")


def global_gpu_state() -> GpuState:
    """Lazily initialise and return the process-wide GpuState."""
    global _global_state
    with _state_lock:
        if _global_state is None:
            _global_state = GpuState()
        return _global_state


def with_gpu(fn: Callable[[GpuState], T]) -> T:
    """Run ``fn`` against the process-wide GpuState with its lock held.

    Mirrors gpu_state.rs:with_gpu.
    """
    st = global_gpu_state()
    with st.lock:
        return fn(st)


def reset_gpu_state() -> None:
    """Drop the process-wide GpuState (primarily for tests)."""
    global _global_state
    with _state_lock:
        _global_state = None


# --- buffer pools ---------------------------------------------------------


def get_f32_buffer(s: GpuState, n: int) -> list[float]:
    """Return a reusable zeroed f32 buffer from the pool, or allocate one.

    Mirrors GpuState::get_f32_buffer.
    """
    pool = s.f32_pool.get(n)
    if pool:
        buf = pool.pop()
        for i in range(len(buf)):
            buf[i] = 0.0
        return buf
    return [0.0] * n


def return_f32_buffer(s: GpuState, buf: list[float]) -> None:
    """Return a buffer to the pool. Mirrors return_f32_buffer."""
    s.f32_pool.setdefault(len(buf), []).append(buf)


def get_q8k_buffer(s: GpuState, n: int) -> bytearray:
    """Return a reusable byte buffer from the pool, or allocate one."""
    pool = s.q8k_pool.get(n)
    if pool:
        return pool.pop()
    return bytearray(n)


def return_q8k_buffer(s: GpuState, buf: bytearray) -> None:
    """Return a byte buffer to the pool."""
    s.q8k_pool.setdefault(len(buf), []).append(buf)


# --- layer LRU / budget ---------------------------------------------------


def touch_layer(s: GpuState, layer: int) -> None:
    """Mark a layer most-recently-used and enforce the budget.

    Mirrors GpuState::touch_layer.
    """
    if s.layer_config.max_resident_layers == 0 and s.layer_config.max_vram_bytes == 0:
        return  # unlimited
    remove_from_lru(s, layer)
    s.layer_lru.append(layer)
    enforce_budget(s)


def remove_from_lru(s: GpuState, layer: int) -> None:
    try:
        s.layer_lru.remove(layer)
    except ValueError:
        pass


def enforce_budget(s: GpuState) -> None:
    """Evict LRU layers and orphan entries until within budget.

    Mirrors GpuState::enforce_budget / enforce_budget_protecting(None).
    """
    enforce_budget_protecting(s, None)


def enforce_budget_protecting(s: GpuState, protect: WeightCacheKey | None) -> None:
    max_layers = s.layer_config.max_resident_layers
    max_bytes = s.layer_config.max_vram_bytes

    while True:
        over_layer = max_layers > 0 and len(s.layer_lru) > max_layers
        over_bytes = max_bytes > 0 and s.resident_bytes > max_bytes
        if not over_layer and not over_bytes:
            break
        if not s.layer_lru:
            break
        evict = s.layer_lru.pop(0)
        evict_layer_internal(s, evict)

    while max_bytes > 0 and s.resident_bytes > max_bytes:
        if s.orphan_f16_keys:
            key = s.orphan_f16_keys.pop(0)
            buf = s.resident_f16.pop(key, None)
            if buf is not None:
                s.resident_bytes -= len(buf) * SIZE_OF_U16
            continue
        if s.orphan_quant_keys:
            key = s.orphan_quant_keys[0]
            if protect is not None and key == protect:
                # Don't evict the entry the caller still needs.
                break
            s.orphan_quant_keys.pop(0)
            buf = s.resident_quant.pop(key, None)
            if buf is not None:
                s.resident_bytes -= len(buf)
            continue
        break


def ensure_vram_headroom(s: GpuState, additional: int) -> None:
    """Evict until resident_bytes + additional fits the budget.

    Called before inserting new weights. Mirrors
    GpuState::ensure_vram_headroom.
    """
    max_bytes = s.layer_config.max_vram_bytes
    if max_bytes == 0:
        return
    while s.resident_bytes + additional > max_bytes:
        if s.layer_lru:
            evict = s.layer_lru.pop(0)
            evict_layer_internal(s, evict)
            continue
        if s.orphan_f16_keys:
            key = s.orphan_f16_keys.pop(0)
            buf = s.resident_f16.pop(key, None)
            if buf is not None:
                s.resident_bytes -= len(buf) * SIZE_OF_U16
            continue
        if s.orphan_quant_keys:
            key = s.orphan_quant_keys.pop(0)
            buf = s.resident_quant.pop(key, None)
            if buf is not None:
                s.resident_bytes -= len(buf)
            continue
        break


def touch_orphan_quant(s: GpuState, key: WeightCacheKey) -> None:
    try:
        s.orphan_quant_keys.remove(key)
    except ValueError:
        pass
    s.orphan_quant_keys.append(key)


def ensure_resident_quant(
    s: GpuState, key: WeightCacheKey, host: bytes | bytearray
) -> None:
    """Upload quantized weights once and reuse the buffer on later tokens.

    Mirrors GpuState::ensure_resident_quant.
    """
    if key not in s.resident_quant:
        ensure_vram_headroom(s, len(host))
        buf = bytearray(host)
        s.resident_bytes += len(buf)
        s.resident_quant[key] = buf
        s.orphan_quant_keys.append(key)
        enforce_budget_protecting(s, key)
    else:
        touch_orphan_quant(s, key)


def evict_layer_internal(s: GpuState, layer: int) -> None:
    """Drop a layer's weights when no other layer references them.

    Mirrors GpuState::evict_layer_internal.
    """
    entry = s.layer_map.pop(layer, None)
    if entry is None:
        return
    for key in entry.f32_keys:
        if key_referenced_f32(s, key):
            continue
        buf = s.resident_f32.pop(key, None)
        if buf is not None:
            s.resident_bytes -= len(buf) * SIZE_OF_F32
    for key in entry.f16_keys:
        if key_referenced_f16(s, key):
            continue
        buf = s.resident_f16.pop(key, None)
        if buf is not None:
            s.resident_bytes -= len(buf) * SIZE_OF_U16


def key_referenced_f32(s: GpuState, key: WeightCacheKey) -> bool:
    return any(key in e.f32_keys for e in s.layer_map.values())


def key_referenced_f16(s: GpuState, key: WeightCacheKey) -> bool:
    return any(key in e.f16_keys for e in s.layer_map.values())


# --- RMS-norm helper ------------------------------------------------------


def rms_norm_into(
    x: list[float], weight: list[float], eps: float, out: list[float]
) -> None:
    """Compute y = (x / rms(x)) * weight, matching the GGML/Llama RMS-norm.

    Mirrors gpu_kernels.go:rmsNormInto.
    """
    n = len(x)
    ss = 0.0
    for v in x:
        ss += v * v
    scale = 1.0 / math.sqrt(ss / n + eps)
    for i in range(n):
        out[i] = x[i] * scale * weight[i]
