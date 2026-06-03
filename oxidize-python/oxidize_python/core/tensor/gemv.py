"""GEMV/GEMM kernels mirroring oxidize-golang/core/tensor/gemv.go."""

from __future__ import annotations

import math
from collections.abc import Callable

from oxidize_python.core.tensor.errors import GemmError, GemvError
from oxidize_python.core.tensor.parallel import parallelize_rows


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

    def _rows(start: int, end: int) -> None:
        for r in range(start, end):
            row = matrix[r * cols : (r + 1) * cols]
            output[r] = sum(row[c] * vector[c] for c in range(cols))

    parallelize_rows(rows, _rows)


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

    from oxidize_python.core.tensor.constants import TRANSPOSED_GEMV_COL_CHUNK

    c = 0
    while c < cols:
        end = min(c + TRANSPOSED_GEMV_COL_CHUNK, cols)

        def _cols(start: int, stop: int) -> None:
            for k in range(start, stop):
                col = c + k
                output[col] = sum(matrix[r * cols + col] * vector[r] for r in range(rows))

        parallelize_rows(end - c, _cols)
        c = end


def gemv_quantized_f32(
    qbytes: bytes | bytearray,
    dequant: Callable[[bytes | bytearray, list[float]], None],
    rows: int,
    cols: int,
    vector: list[float],
    output: list[float],
    scratch: list[float] | None = None,
) -> None:
    if scratch is None:
        scratch = [0.0] * cols
    elif len(scratch) < cols:
        raise GemvError("scratch too small")
    if rows == 0:
        return
    bytes_per_row = len(qbytes) // rows
    if bytes_per_row * rows != len(qbytes):
        raise GemvError("qbytes not aligned to row boundary")

    buf = [0.0] * cols

    def _rows(start: int, end: int) -> None:
        for r in range(start, end):
            block = qbytes[r * bytes_per_row : (r + 1) * bytes_per_row]
            dequant(block, buf)
            output[r] = sum(buf[c] * vector[c] for c in range(cols))

    parallelize_rows(rows, _rows)


def gemv_quantized_f32_transposed(
    qbytes: bytes | bytearray,
    dequant: Callable[[bytes | bytearray, list[float]], None],
    rows: int,
    cols: int,
    vector: list[float],
    output: list[float],
    scratch: list[float] | None = None,
) -> None:
    if scratch is None or len(scratch) < cols:
        scratch = [0.0] * cols
    bytes_per_row = len(qbytes) // rows
    if bytes_per_row * rows != len(qbytes):
        raise GemvError("qbytes not aligned to row boundary")
    row = [0.0] * cols
    for r in range(rows):
        dequant(qbytes[r * bytes_per_row : (r + 1) * bytes_per_row], row)
        v = vector[r]
        for c in range(cols):
            output[c] += row[c] * v


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

    def _rows(start: int, end: int) -> None:
        for r in range(start, end):
            for c in range(cols):
                output[r * cols + c] = sum(
                    left[r * shared + k] * right[k * cols + c] for k in range(shared)
                )

    parallelize_rows(rows, _rows)


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
    if scratch is None or len(scratch) < shared:
        scratch = [0.0] * shared
    bytes_per_row = len(qbytes) // rows
    if bytes_per_row * rows != len(qbytes):
        raise GemmError("qbytes not aligned to row boundary")
    row = [0.0] * shared

    def _rows(start: int, end: int) -> None:
        for r in range(start, end):
            dequant(qbytes[r * bytes_per_row : (r + 1) * bytes_per_row], row)
            for c in range(cols):
                output[out_offset + r * cols + c] = sum(
                    row[k] * right[k * cols + c] for k in range(shared)
                )

    parallelize_rows(rows, _rows)


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
        for i, v in enumerate(output):
            output[i] = v + reduce_buf[i]


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
    for r in range(rows):
        for c in range(cols):
            output[r * cols + c] = sum(
                int(left[r * shared + k]) * int(right[k * cols + c]) for k in range(shared)
            )


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
                if idx % 2 == 0:
                    nibble = _unpack_nibble(packed & 0x0F)
                else:
                    nibble = _unpack_nibble((packed >> 4) & 0x0F)
                s += int(left[r * shared + k]) * nibble
            output[r * cols + c] = s


def sigmoid(x: list[float], out: list[float]) -> None:
    if len(out) < len(x):
        raise GemvError("out too small")
    for i, v in enumerate(x):
        out[i] = 1.0 / (1.0 + math.exp(-v))
