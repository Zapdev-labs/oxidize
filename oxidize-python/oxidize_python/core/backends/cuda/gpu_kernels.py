"""GPU activation-buffer kernels and the GPU-native forward pass.

Mirrors oxidize-golang/core/backends/cuda/gpu_kernels.go (port of
oxidize-core/src/backends/cuda/gpu_kernels.rs and gpu_native_forward.rs).

In the pure-Python build the "device" buffers are host lists and the kernels run
on the host; the activation-buffer lifecycle, resident weight caching, and fused
RMS-norm + Q/K/V projection match the Rust/Go semantics.
"""

from __future__ import annotations

from oxidize_python.core.backends.cuda.cuda import (
    GemvCudaError,
    MemoryError,
    _gemv_quantized_into,
)
from oxidize_python.core.backends.cuda.gpu_state import rms_norm_into, with_gpu
from oxidize_python.core.backends.cuda.types import (
    GgmlType,
    GpuActivationBuffer,
    GpuState,
    dequant_kernel_for,
    f32_cache_key,
)


def gpu_init_activation_buffers(hidden_size: int, intermediate_size: int) -> None:
    """Allocate (or reallocate) activation buffers for a given model size.

    Safe to call multiple times; existing buffers are kept when the dimensions
    are unchanged. Mirrors gpu_kernels.rs:gpu_init_activation_buffers.
    """
    if hidden_size <= 0 or intermediate_size <= 0:
        raise MemoryError("activation buffer dims must be positive")

    def _f(s: GpuState) -> None:
        ab = s.activation
        if (
            ab is not None
            and ab.hidden_size == hidden_size
            and ab.intermediate_size == intermediate_size
        ):
            return
        s.activation = GpuActivationBuffer(
            hidden=[0.0] * hidden_size,
            normed=[0.0] * hidden_size,
            ffn_gate=[0.0] * intermediate_size,
            ffn_up=[0.0] * intermediate_size,
            ffn_down_in=[0.0] * intermediate_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    with_gpu(_f)


def gpu_upload_hidden(hidden: list[float]) -> None:
    """Copy a host hidden-state list into activation.hidden.

    Mirrors gpu_kernels.rs:gpu_upload_hidden.
    """

    def _f(s: GpuState) -> None:
        ab = s.activation
        if ab is None:
            raise MemoryError("activation buffers not initialised")
        if len(hidden) != ab.hidden_size:
            raise MemoryError(
                f"gpu_upload_hidden: slice len {len(hidden)} != hidden_size {ab.hidden_size}"
            )
        ab.hidden[:] = hidden

    with_gpu(_f)


def gpu_download_hidden(out: list[float]) -> None:
    """Copy activation.hidden back into ``out``.

    Mirrors gpu_native_forward.rs:gpu_download_hidden.
    """

    def _f(s: GpuState) -> None:
        ab = s.activation
        if ab is None:
            raise MemoryError("activation buffers not initialised")
        if len(out) != ab.hidden_size:
            raise MemoryError(
                f"gpu_download_hidden: out len {len(out)} != hidden_size {ab.hidden_size}"
            )
        out[:] = ab.hidden

    with_gpu(_f)


def gpu_rms_norm(weight: list[float], eps: float) -> None:
    """Run RMS-norm reading activation.hidden and writing activation.normed.

    ``weight`` is a per-element scale of length hidden_size and is cached in
    resident_f32. Mirrors gpu_kernels.rs:gpu_rms_norm.
    """

    def _f(s: GpuState) -> None:
        ab = s.activation
        if ab is None:
            raise MemoryError("activation buffers not initialised")
        if len(weight) != ab.hidden_size:
            raise MemoryError(
                f"gpu_rms_norm: weight len {len(weight)} != hidden_size {ab.hidden_size}"
            )
        # Cache the norm weight resident, mirroring the Rust pointer-identity pattern.
        key = f32_cache_key(weight)
        if key not in s.resident_f32:
            s.resident_f32[key] = list(weight)
        rms_norm_into(ab.hidden, weight, eps, ab.normed)

    with_gpu(_f)


def gpu_attn_rms_and_qkv_q4k(
    attn_norm: list[float],
    eps: float,
    wq: bytes,
    q_len: int,
    wk: bytes,
    kv_len: int,
    wv: bytes,
    q_type: GgmlType,
    hidden_size: int,
    q_out: list[float],
    k_out: list[float],
    v_out: list[float],
) -> None:
    """Run the fused attention pre-projection on-device.

    RMS-norms the resident hidden state, then projects Q/K/V with the given
    quantized weights, leaving results in q_out/k_out/v_out. Mirrors
    gpu_native_forward.rs:gpu_attn_rms_and_qkv_q4k.

    ``wq``/``wk``/``wv`` are raw quantized weight bytes of type ``q_type``;
    ``q_len``/``kv_len`` are the projection output dimensions and ``hidden_size``
    the input dimension.
    """
    if dequant_kernel_for(q_type) is None:
        raise GemvCudaError(f"unsupported quant type {int(q_type)} for fused qkv")
    if len(q_out) < q_len or len(k_out) < kv_len or len(v_out) < kv_len:
        raise GemvCudaError("qkv output buffers too small")

    def _f(s: GpuState) -> None:
        ab = s.activation
        if ab is None:
            raise MemoryError("activation buffers not initialised")
        if len(attn_norm) != hidden_size or ab.hidden_size != hidden_size:
            raise MemoryError(
                "gpu_attn_rms_and_qkv: hidden_size mismatch "
                f"(norm={len(attn_norm)} hidden={ab.hidden_size} arg={hidden_size})"
            )
        # RMS-norm hidden -> normed.
        rms_norm_into(ab.hidden, attn_norm, eps, ab.normed)
        # Q/K/V projections against the normed hidden state.
        _gemv_quantized_into(s, wq, q_type, ab.normed, q_len, hidden_size, q_out)
        _gemv_quantized_into(s, wk, q_type, ab.normed, kv_len, hidden_size, k_out)
        _gemv_quantized_into(s, wv, q_type, ab.normed, kv_len, hidden_size, v_out)

    with_gpu(_f)
