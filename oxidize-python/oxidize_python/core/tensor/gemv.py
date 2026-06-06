"""GEMV/GEMM kernels mirroring oxidize-golang/core/tensor/gemv.go.

numpy fast paths are used when numpy is available (virtually always).
The pure-Python fallbacks are kept for correctness reference only.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from oxidize_python.core.tensor.errors import GemmError, GemvError


def gemv_f32(
    matrix: list[float],
    rows: int,
    cols: int,
    vector: list[float],
    output: list[float],
) -> None:
    if rows <= 0 or cols <= 0:
        raise GemvError(f"invalid dims rows={rows} cols={cols}")
    if len(matrix) < rows * cols:
        raise GemvError("matrix buffer too small")
    if len(vector) < cols:
        raise GemvError("vector buffer too small")
    if len(output) < rows:
        raise GemvError("output buffer too small")

    m = np.asarray(matrix[: rows * cols], dtype=np.float32).reshape(rows, cols)
    v = np.asarray(vector[:cols], dtype=np.float32)
    result = m @ v
    output[:rows] = result.tolist()


def gemv_f32_transposed(
    matrix: list[float],
    rows: int,
    cols: int,
    vector: list[float],
    output: list[float],
) -> None:
    if len(matrix) < rows * cols:
        raise GemvError("matrix buffer too small")
    if len(vector) < rows:
        raise GemvError("vector buffer too small")
    if len(output) < cols:
        raise GemvError("output buffer too small")

    # matrix is row-major (rows×cols) but used transposed: output = M^T @ v
    m = np.asarray(matrix[: rows * cols], dtype=np.float32).reshape(rows, cols)
    v = np.asarray(vector[:rows], dtype=np.float32)
    result = m.T @ v
    output[:cols] = result.tolist()


def gemv_quantized_f32(
    qbytes: bytes | bytearray,
    dequant: Callable[[bytes | bytearray, list[float]], None],
    rows: int,
    cols: int,
    vector: list[float],
    output: list[float],
    scratch: list[float] | None = None,
) -> None:
    if rows == 0:
        return
    bytes_per_row = len(qbytes) // rows
    if bytes_per_row * rows != len(qbytes):
        raise GemvError("qbytes not aligned to row boundary")

    buf = [0.0] * cols
    v = np.asarray(vector[:cols], dtype=np.float32)
    for r in range(rows):
        block = qbytes[r * bytes_per_row : (r + 1) * bytes_per_row]
        dequant(block, buf)
        output[r] = float(np.dot(np.asarray(buf, dtype=np.float32), v))


def gemv_quantized_f32_transposed(
    qbytes: bytes | bytearray,
    dequant: Callable[[bytes | bytearray, list[float]], None],
    rows: int,
    cols: int,
    vector: list[float],
    output: list[float],
    scratch: list[float] | None = None,
) -> None:
    bytes_per_row = len(qbytes) // rows
    if bytes_per_row * rows != len(qbytes):
        raise GemvError("qbytes not aligned to row boundary")
    row = [0.0] * cols
    out = np.asarray(output[:cols], dtype=np.float32)
    for r in range(rows):
        dequant(qbytes[r * bytes_per_row : (r + 1) * bytes_per_row], row)
        out += np.asarray(row, dtype=np.float32) * vector[r]
    output[:cols] = out.tolist()


def gemm_f32(
    left: list[float],
    right: list[float],
    rows: int,
    shared: int,
    cols: int,
    output: list[float],
) -> None:
    if len(left) < rows * shared:
        raise GemmError("left buffer too small")
    if len(right) < shared * cols:
        raise GemmError("right buffer too small")
    if len(output) < rows * cols:
        raise GemmError("output buffer too small")

    lhs = np.asarray(left[: rows * shared], dtype=np.float32).reshape(rows, shared)
    rhs = np.asarray(right[: shared * cols], dtype=np.float32).reshape(shared, cols)
    result = (lhs @ rhs).ravel()
    output[: rows * cols] = result.tolist()


def gemm_quantized_f32(
    qbytes: bytes | bytearray,
    dequant: Callable[[bytes | bytearray, list[float]], None],
    rows: int,
    shared: int,
    cols: int,
    batch: int,
    right: list[float],
    output: list[float],
    scratch: list[float] | None = None,
) -> None:
    if batch <= 0:
        return
    expected_in = batch * shared * cols
    if len(right) < expected_in:
        raise GemmError(f"right buffer too small: need {expected_in}, have {len(right)}")
    expected_out = batch * rows * cols
    if len(output) < expected_out:
        raise GemmError(f"output buffer too small: need {expected_out}, have {len(output)}")
    for t in range(batch):
        off_in = t * shared * cols
        off_out = t * rows * cols
        _gemm_quantized_f32_one(
            qbytes,
            dequant,
            rows,
            shared,
            cols,
            right[off_in : off_in + shared * cols],
            output,
            scratch,
            out_offset=off_out,
        )


def _gemm_quantized_f32_one(
    qbytes: bytes | bytearray,
    dequant: Callable[[bytes | bytearray, list[float]], None],
    rows: int,
    shared: int,
    cols: int,
    right: list[float],
    output: list[float],
    scratch: list[float] | None,
    *,
    out_offset: int = 0,
) -> None:
    bytes_per_row = len(qbytes) // rows
    if bytes_per_row * rows != len(qbytes):
        raise GemmError("qbytes not aligned to row boundary")
    row = [0.0] * shared
    r_np = np.asarray(right, dtype=np.float32).reshape(shared, cols)
    for r in range(rows):
        dequant(qbytes[r * bytes_per_row : (r + 1) * bytes_per_row], row)
        row_np = np.asarray(row, dtype=np.float32)
        result = row_np @ r_np
        output[out_offset + r * cols : out_offset + (r + 1) * cols] = result.tolist()


def gemm_f32_tensor_parallel(
    left: list[float],
    right: list[float],
    rows: int,
    shared: int,
    cols: int,
    output: list[float],
    reduce_buf: list[float] | None,
) -> None:
    gemm_f32(left, right, rows, shared, cols, output)
    if reduce_buf is not None and len(reduce_buf) >= len(output):
        out = np.asarray(output, dtype=np.float32)
        rb = np.asarray(reduce_buf[: len(output)], dtype=np.float32)
        result = (out + rb).tolist()
        output[: len(output)] = result


def gemm_i8(
    left: list[int],
    right: list[int],
    rows: int,
    shared: int,
    cols: int,
    output: list[int],
) -> None:
    if len(left) < rows * shared:
        raise GemmError("left buffer too small")
    if len(right) < shared * cols:
        raise GemmError("right buffer too small")
    if len(output) < rows * cols:
        raise GemmError("output buffer too small")
    lhs = np.asarray(left[: rows * shared], dtype=np.int32).reshape(rows, shared)
    rhs = np.asarray(right[: shared * cols], dtype=np.int32).reshape(shared, cols)
    result = (lhs @ rhs).ravel().tolist()
    output[: rows * cols] = result


def _unpack_nibble(n: int) -> int:
    return n - 16 if n >= 8 else n


def gemm_i4(
    left: list[int],
    right: bytes | bytearray,
    rows: int,
    shared: int,
    cols: int,
    output: list[int],
) -> None:
    if len(left) < rows * shared:
        raise GemmError("left buffer too small")
    if len(right) < (shared * cols) // 2:
        raise GemmError("right buffer too small")
    if len(output) < rows * cols:
        raise GemmError("output buffer too small")
    for r in range(rows):
        for c in range(cols):
            s = 0
            for k in range(shared):
                idx = k * cols + c
                packed = right[idx // 2]
                nibble = _unpack_nibble(packed & 0x0F if idx % 2 == 0 else (packed >> 4) & 0x0F)
                s += int(left[r * shared + k]) * nibble
            output[r * cols + c] = s


def sigmoid(x: list[float], out: list[float]) -> None:
    if len(out) < len(x):
        raise GemvError("out too small")
    result = (1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))).tolist()
    out[: len(x)] = result
