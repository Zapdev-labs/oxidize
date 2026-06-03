"""CPU ComputeBackend mirroring oxidize-golang/core/backends/cpu."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from oxidize_python.core import backend as be
from oxidize_python.core.tensor import (
    apply_rope_f32,
    gemm_f32,
    gemv_quantized_f32,
    rms_norm_f32,
    scaled_dot_product_attention_f32,
    sigmoid,
    softmax_f32,
)


class CpuError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class CpuTensor:
    data: list[float]
    shape: list[int]


@dataclass
class CpuWeightStorage:
    bytes: bytes | bytearray
    dequant: Callable[[bytes | bytearray, list[float]], None]


class Cpu:
    def __init__(self) -> None:
        self._mu = threading.Lock()

    def name(self) -> str:
        return "cpu"

    def tensor_from_f32(self, data: list[float]) -> be.TensorHandle:
        return CpuTensor(data=list(data), shape=[len(data)])

    def tensor_from_f32_2d(self, data: list[float], rows: int, cols: int) -> be.TensorHandle:
        if len(data) != rows * cols:
            raise CpuError("data length != rows*cols")
        return CpuTensor(data=list(data), shape=[rows, cols])

    def tensor_to_f32(self, t: be.TensorHandle, out: list[float]) -> int:
        tc = _as_cpu(t)
        if len(out) < len(tc.data):
            raise CpuError("output too small")
        out[: len(tc.data)] = tc.data
        return len(tc.data)

    def tensor_shape(self, t: be.TensorHandle) -> list[int]:
        return list(_as_cpu(t).shape)

    def tensor_dtype(self, _t: be.TensorHandle) -> be.DType:
        return be.DType.F32

    def rms_norm(self, input_: be.TensorHandle, weight: be.TensorHandle, eps: float) -> be.TensorHandle:
        inp, w = _as_cpu(input_), _as_cpu(weight)
        out = [0.0] * len(inp.data)
        rms_norm_f32(inp.data, w.data, out, eps)
        return CpuTensor(data=out, shape=list(inp.shape))

    def apply_rope(
        self, input_: be.TensorHandle, position: int, head_dim: int, theta: float
    ) -> be.TensorHandle:
        inp = _as_cpu(input_)
        out = [0.0] * len(inp.data)
        apply_rope_f32(inp.data, out, position, head_dim, theta)
        return CpuTensor(data=out, shape=list(inp.shape))

    def attention_decode(
        self,
        query: be.TensorHandle,
        key_cache: be.TensorHandle,
        value_cache: be.TensorHandle,
        seq_len: int,
        head_dim: int,
        scale: float,
    ) -> be.TensorHandle:
        q, k, v = _as_cpu(query), _as_cpu(key_cache), _as_cpu(value_cache)
        out = [0.0] * head_dim
        scaled_dot_product_attention_f32(q.data, k.data, v.data, out, seq_len, head_dim, scale)
        return CpuTensor(data=out, shape=[head_dim])

    def gemv(
        self, matrix: be.WeightStorage, vector: be.TensorHandle, rows: int, cols: int
    ) -> be.TensorHandle:
        ws = matrix
        if isinstance(ws, CpuWeightStorage):
            vec = _as_cpu(vector)
            out = [0.0] * rows
            gemv_quantized_f32(ws.bytes, ws.dequant, rows, cols, vec.data, out, None)
            return CpuTensor(data=out, shape=[rows])
        raise CpuError("unsupported weight storage")

    def gemm(
        self, a: be.TensorHandle, b: be.TensorHandle, rows: int, shared_dim: int, cols: int
    ) -> be.TensorHandle:
        left, right = _as_cpu(a), _as_cpu(b)
        out = [0.0] * (rows * cols)
        gemm_f32(left.data, right.data, rows, shared_dim, cols, out)
        return CpuTensor(data=out, shape=[rows, cols])

    def add(self, a: be.TensorHandle, b: be.TensorHandle) -> be.TensorHandle:
        x, y = _as_cpu(a), _as_cpu(b)
        if len(x.data) != len(y.data):
            raise CpuError("shape mismatch")
        return CpuTensor(data=[i + j for i, j in zip(x.data, y.data)], shape=list(x.shape))

    def mul(self, a: be.TensorHandle, b: be.TensorHandle) -> be.TensorHandle:
        x, y = _as_cpu(a), _as_cpu(b)
        if len(x.data) != len(y.data):
            raise CpuError("shape mismatch")
        return CpuTensor(data=[i * j for i, j in zip(x.data, y.data)], shape=list(x.shape))

    def sigmoid(self, x: be.TensorHandle) -> be.TensorHandle:
        tc = _as_cpu(x)
        out = [0.0] * len(tc.data)
        sigmoid(tc.data, out)
        return CpuTensor(data=out, shape=list(tc.shape))

    def softmax(self, x: be.TensorHandle) -> be.TensorHandle:
        tc = _as_cpu(x)
        dim = tc.shape[-1] if tc.shape else len(tc.data)
        out = [0.0] * len(tc.data)
        softmax_f32(tc.data, out, dim)
        return CpuTensor(data=out, shape=list(tc.shape))

    def synchronize(self) -> None:
        return None


def _as_cpu(t: be.TensorHandle) -> CpuTensor:
    if not isinstance(t, CpuTensor):
        raise CpuError("expected CpuTensor")
    return t
