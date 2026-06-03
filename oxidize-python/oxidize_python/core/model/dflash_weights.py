"""Weight containers mirroring oxidize-golang/core/model/dflash_weights.go."""

from __future__ import annotations

from dataclasses import dataclass

from oxidize_python.core.quantization import types as quant
from oxidize_python.core.quantization.dequant_k import dequantize
from oxidize_python.core.tensor import gemv


@dataclass
class QuantWeight:
    bytes: bytes
    q_type: quant.Type
    out_dim: int
    in_dim: int


@dataclass
class F32Weight:
    data: list[float]
    rows: int
    cols: int
    quant: QuantWeight | None = None

    @staticmethod
    def from_slice(data: list[float], rows: int, cols: int) -> F32Weight:
        return F32Weight(data=data, rows=rows, cols=cols)

    @staticmethod
    def from_quantized(raw: bytes, qtype: quant.Type, out_dim: int, in_dim: int) -> F32Weight:
        return F32Weight(
            data=[],
            rows=in_dim,
            cols=out_dim,
            quant=QuantWeight(bytes=raw, q_type=qtype, out_dim=out_dim, in_dim=in_dim),
        )

    def is_loaded(self) -> bool:
        return len(self.data) > 0 or self.quant is not None

    def _input_dim(self) -> int:
        if self.quant is not None:
            return self.quant.in_dim
        return self.cols

    def _output_dim(self) -> int:
        if self.quant is not None:
            return self.quant.out_dim
        return self.rows

    def _dequant_fn(self):
        if self.quant is None:
            return None
        qt = self.quant.q_type

        def _fn(row: bytes, out: list[float]) -> None:
            dequantize(qt, row, out)

        return _fn

    def gemv(self, input_: list[float], output: list[float]) -> None:
        if self.quant is not None:
            q = self.quant
            gemv.gemv_quantized_f32(
                q.bytes,
                self._dequant_fn(),
                q.out_dim,
                q.in_dim,
                input_,
                output,
                None,
            )
            return
        gemv.gemv_f32_transposed(self.data, self.cols, self.rows, input_, output)

    def gemm(self, inputs: list[float], outputs: list[float], batch: int) -> None:
        if batch <= 1:
            self.gemv(inputs, outputs)
            return
        if self.quant is not None:
            q = self.quant
            fn = self._dequant_fn()
            if fn is None:
                raise ValueError("missing dequant fn")
            gemv.gemm_quantized_f32(
                q.bytes,
                fn,
                q.out_dim,
                q.in_dim,
                1,
                batch,
                inputs,
                outputs,
                None,
            )
            return
        gemv.gemm_f32(inputs, self.data, batch, self.cols, self.rows, outputs)

    def row(self, row_idx: int, output: list[float]) -> None:
        if self.quant is not None:
            q = self.quant
            if row_idx >= q.out_dim or len(output) != q.in_dim:
                raise ValueError(
                    f"row bounds mismatch: row={row_idx} out={q.out_dim} want in={q.in_dim}"
                )
            row_bytes = quant.quantized_size(q.q_type, q.in_dim)
            start = row_idx * row_bytes
            end = start + row_bytes
            if end > len(q.bytes):
                raise ValueError("quantized row extends past tensor data")
            dequantize(q.q_type, q.bytes[start:end], output)
            return
        if row_idx >= self.rows or len(output) != self.cols:
            raise ValueError(
                f"row bounds mismatch: row={row_idx} rows={self.rows} "
                f"out={len(output)} cols={self.cols}"
            )
        base = row_idx * self.cols
        output[:] = self.data[base : base + self.cols]


def transpose_f32(data: list[float], gguf_rows: int, gguf_cols: int) -> list[float]:
    result = [0.0] * len(data)
    for r in range(gguf_rows):
        for c in range(gguf_cols):
            result[c * gguf_rows + r] = data[r * gguf_cols + c]
    return result


def quantized_gemv_supported(qtype: quant.Type, in_dim: int) -> bool:
    match qtype:
        case quant.Type.Q4_K_S | quant.Type.Q4_K_M | quant.Type.Q2_K | quant.Type.Q6_K:
            return in_dim % 256 == 0
        case quant.Type.Q8_0:
            return in_dim % 32 == 0
        case _:
            return False


def f32_weight_from_dims(data: list[float], dims: list[int]) -> F32Weight:
    if not dims:
        return F32Weight([], 0, 0)
    if len(dims) == 1:
        n = dims[0]
        return F32Weight.from_slice(data, n, 1)
    r, c = dims[0], dims[1]
    return F32Weight.from_slice(transpose_f32(data, r, c), c, r)


def weight_from_2d(data: list[float], rows: int, cols: int) -> F32Weight:
    if not data:
        return F32Weight([], 0, 0)
    if cols == 0:
        cols = len(data) // rows if rows else 0
    return F32Weight.from_slice(data, rows, cols)


def ones(n: int) -> list[float]:
    return [1.0] * n
