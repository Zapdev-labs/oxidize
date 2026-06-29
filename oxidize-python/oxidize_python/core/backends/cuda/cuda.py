"""CUDA backend mirroring oxidize-golang/core/backends/cuda.

In the pure-Python build the "device" kernels run on the host (the same way the
CGO_ENABLED=0 Go build does). The bookkeeping for resident weight caches, buffer
pools, and layer-by-layer VRAM management matches the Rust/Go ports so the same
APIs and semantics flow through unchanged. A native CUDA build can later layer
real device allocations on top while reusing this logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from oxidize_python.core import quantization as quant
from oxidize_python.core.backends.cuda.gpu_state import ensure_resident_quant
from oxidize_python.core.backends.cuda.types import (
    GgmlType,
    GpuState,
    byte_cache_key,
    supports_quantized_gpu,
)


@dataclass
class BuildInfo:
    detected_at_build: bool = False
    cuda_path: str = ""


def build_info() -> BuildInfo:
    return BuildInfo(detected_at_build=False, cuda_path="")


class MemoryDevice(IntEnum):
    CPU = 0
    CUDA = 1

    def __str__(self) -> str:
        if self == MemoryDevice.CPU:
            return "cpu"
        if self == MemoryDevice.CUDA:
            return "cuda"
        return f"memory({int(self)})"


MemoryCpu = MemoryDevice.CPU
MemoryCuda = MemoryDevice.CUDA


class MemoryError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"cuda memory: {message}")


class GemvCudaError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"cuda gemv: {message}")


class GemmCudaError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"cuda gemm: {message}")


def initialize() -> None:
    raise MemoryError("cuda backend not linked in this build")


def gemv_f32_cuda(
    matrix: list[float],
    vector: list[float],
    rows: int,
    cols: int,
    output: list[float],
) -> None:
    """Compute output[r] = dot(matrix_row_r, vector) for a dense f32 matrix.

    Mirrors gemv_f32.rs:gemv_f32_cuda. The native build dispatches to
    ``cublasSgemv_v2``; the pure-Python build runs the equivalent host loop.
    """
    validate_gemv_dims(rows, cols)
    if len(matrix) < rows * cols or len(vector) < cols or len(output) < rows:
        raise GemvCudaError("buffer too small")
    for r in range(rows):
        base = r * cols
        s = 0.0
        for c in range(cols):
            s += matrix[base + c] * vector[c]
        output[r] = s


def gemm_f32_cuda(
    left: list[float],
    right: list[float],
    rows: int,
    shared: int,
    cols: int,
    output: list[float],
) -> None:
    """Compute output = left[rows x shared] * right[shared x cols], row-major.

    Mirrors gemm.rs:gemm_f32_cuda. The native build dispatches to
    ``cublasSgemm_v2``; the pure-Python build runs the equivalent host GEMM.
    """
    validate_gemm_dims(rows, shared, cols)
    if len(left) < rows * shared or len(right) < shared * cols or len(output) < rows * cols:
        raise GemmCudaError("buffer too small")
    for r in range(rows):
        lbase = r * shared
        obase = r * cols
        for c in range(cols):
            output[obase + c] = 0.0
        for k in range(shared):
            a = left[lbase + k]
            if a == 0.0:
                continue
            rbase = k * cols
            for c in range(cols):
                output[obase + c] += a * right[rbase + c]


def gemv_quantized_cuda(
    qbytes: bytes,
    ggml_type: int,
    vector: list[float],
    rows: int,
    cols: int,
    output: list[float],
    scratch: list[float] | None = None,
) -> None:
    """Dequantize a quantized weight matrix and compute its GEMV.

    Dequantizes ``qbytes`` (a ``rows x cols`` matrix of GGML type ``ggml_type``)
    and computes ``output[r] = dot(matrix_row_r, vector)``. Mirrors
    gemv_quantized.rs's on-the-fly quantized GEMV dispatch. The optional
    ``scratch`` list (sized rows*cols) is reused as the dequant target to avoid
    allocation.
    """
    validate_gemv_dims(rows, cols)
    if len(vector) < cols or len(output) < rows:
        raise GemvCudaError("buffer too small")
    t = GgmlType(ggml_type)
    if not supports_quantized_gpu(t):
        raise GemvCudaError(f"unsupported quant type {ggml_type} for GPU gemv")
    n = rows * cols
    if scratch is not None and len(scratch) >= n:
        dequant = scratch
    else:
        dequant = [0.0] * n
    _dequantize_matrix(qbytes, t, dequant, n)
    for r in range(rows):
        base = r * cols
        s = 0.0
        for c in range(cols):
            s += dequant[base + c] * vector[c]
        output[r] = s


def _ggml_to_quant_type(t: GgmlType) -> quant.Type:
    """Map a GGML numeric type id to the quantization package Type."""
    table = {
        GgmlType.F32: quant.Type.F32,
        GgmlType.F16: quant.Type.F16,
        GgmlType.Q4_0: quant.Type.Q4_0,
        GgmlType.Q8_0: quant.Type.Q8_0,
        GgmlType.Q2_K: quant.Type.Q2_K,
        GgmlType.Q4_K: quant.Type.Q4_K_M,
        GgmlType.Q6_K: quant.Type.Q6_K,
    }
    qt = table.get(GgmlType(t))
    if qt is None:
        raise GemvCudaError(f"no dequant for ggml type {int(t)}")
    return qt


class _WriteThroughList(list):
    """A list whose slice reads return a write-through window into the parent.

    Several of the shared block-dequant kernels do ``out = output[a:b]`` and then
    assign ``out[i] = v``. On a plain ``list`` that slice is a copy, so the writes
    are silently dropped. This subclass returns a window proxy for slice reads so
    the kernels' per-block writes land back in the underlying buffer. Scalar
    indexing and ``len()`` keep their normal list behaviour, so the kernels'
    bounds checks (``len(output) < blocks * QK``) still work.
    """

    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, slice):
            start = key.start or 0
            return _ListWindow(self, start)
        return list.__getitem__(self, key)


class _ListWindow:
    """Write-through window over ``parent[offset:]`` used by _WriteThroughList."""

    __slots__ = ("_parent", "_offset")

    def __init__(self, parent: list[float], offset: int) -> None:
        self._parent = parent
        self._offset = offset

    def __setitem__(self, i: int, value: float) -> None:
        list.__setitem__(self._parent, self._offset + i, value)

    def __getitem__(self, i: int) -> float:
        return list.__getitem__(self._parent, self._offset + i)


def _dequantize_matrix(
    qbytes: bytes, t: GgmlType, out: list[float], n: int
) -> None:
    """Decode raw quantized bytes into f32 using the shared quant kernels.

    ``out`` is filled in place with exactly ``n`` decoded values. Mirrors
    cuda.go:dequantizeMatrix (which calls quant.DequantizeScalar).
    """
    qt = _ggml_to_quant_type(t)
    # The shared dequant kernels write into a pre-sized output list (they index
    # rather than append), and require room for every block present in the input
    # bytes. Size the buffer to whichever is larger: the values the GEMV needs
    # (rounded up to a whole block) or the values the raw bytes decode to.
    block = qt.block_size() or 1
    bpb = qt.bytes_per_block() or 1
    blocks_from_bytes = len(qbytes) // bpb if bpb else 0
    sized = max(((n + block - 1) // block) * block, blocks_from_bytes * block)
    decoded = _WriteThroughList([0.0] * sized)
    try:
        quant.dequantize(qt, bytes(qbytes), decoded)
    except Exception as exc:  # noqa: BLE001 - wrap into the backend error type
        raise GemvCudaError(f"dequant failed: {exc}") from exc
    for i in range(n):
        out[i] = list.__getitem__(decoded, i)


def _gemv_quantized_into(
    s: GpuState,
    qbytes: bytes,
    t: GgmlType,
    vector: list[float],
    rows: int,
    cols: int,
    out: list[float],
) -> None:
    """Internal GEMV used by the GPU-native forward pass.

    Caches the raw quantized weight resident (uploaded once) and performs the
    dequant-and-dot against ``vector``, writing ``rows`` results into ``out``.
    Mirrors cuda.go:gemvQuantizedInto.
    """
    if len(vector) < cols or len(out) < rows:
        raise GemvCudaError("buffer too small")
    key = byte_cache_key(qbytes)
    ensure_resident_quant(s, key, qbytes)
    resident = s.resident_quant.get(key)
    if resident is None:
        resident = qbytes
    n = rows * cols
    dequant = [0.0] * n
    _dequantize_matrix(resident, t, dequant, n)
    for r in range(rows):
        base = r * cols
        acc = 0.0
        for c in range(cols):
            acc += dequant[base + c] * vector[c]
        out[r] = acc


def validate_gemv_dims(rows: int, cols: int) -> None:
    if rows <= 0 or cols <= 0:
        raise GemvCudaError(f"invalid dims rows={rows} cols={cols}")


def validate_q8_0_gemv_dims(rows: int, cols: int) -> None:
    if cols % 32 != 0:
        raise GemvCudaError("Q8_0 GEMV requires cols to be a multiple of 32")
    validate_gemv_dims(rows, cols)


def validate_gemm_dims(rows: int, shared: int, cols: int) -> None:
    if rows <= 0 or shared <= 0 or cols <= 0:
        raise GemmCudaError("invalid dims")
