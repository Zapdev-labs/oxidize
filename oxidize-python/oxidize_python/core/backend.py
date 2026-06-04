"""Compute backend abstraction mirroring oxidize_core::backend."""

from __future__ import annotations

import sys
from enum import IntEnum, auto
from typing import Any, Protocol


class Backend(IntEnum):
    CPU = auto()
    METAL = auto()
    CUDA = auto()
    MLX = auto()
    VULKAN = auto()
    INTEL_ARC = auto()

    def __str__(self) -> str:
        names = {
            Backend.CPU: "cpu",
            Backend.METAL: "metal",
            Backend.CUDA: "cuda",
            Backend.MLX: "mlx",
            Backend.VULKAN: "vulkan",
            Backend.INTEL_ARC: "intel-arc",
        }
        return names.get(self, f"backend({int(self)})")


def parse_backend(name: str) -> Backend:
    key = name.strip().lower()
    table = {
        "cpu": Backend.CPU,
        "metal": Backend.METAL,
        "cuda": Backend.CUDA,
        "mlx": Backend.MLX,
        "vulkan": Backend.VULKAN,
        "intel-arc": Backend.INTEL_ARC,
        "arc": Backend.INTEL_ARC,
    }
    if key not in table:
        raise ValueError(f"unknown backend: {name!r}")
    return table[key]


def _vulkan_detected() -> bool:
    return False


def effective_backend(b: Backend) -> tuple[Backend, str, bool]:
    if b == Backend.MLX and sys.platform != "darwin":
        return (
            Backend.CPU,
            "MLX backend requested but unavailable on Linux; falling back to CPU",
            True,
        )
    if b == Backend.INTEL_ARC:
        if _vulkan_detected():
            return Backend.INTEL_ARC, "", False
        return (
            Backend.VULKAN,
            "Intel Arc backend requested but Vulkan was not detected at build time; "
            "using Vulkan fallback path",
            True,
        )
    return b, "", False


class DType(IntEnum):
    F32 = auto()
    F16 = auto()
    I8 = auto()
    I16 = auto()
    I32 = auto()
    I64 = auto()

    def size_in_bytes(self) -> int:
        match self:
            case DType.F32 | DType.I32:
                return 4
            case DType.F16 | DType.I16:
                return 2
            case DType.I8:
                return 1
            case DType.I64:
                return 8
        return 0


TensorHandle = Any
WeightStorage = Any


class ComputeBackend(Protocol):
    def name(self) -> str: ...

    def tensor_from_f32(self, data: list[float]) -> TensorHandle: ...

    def tensor_from_f32_2d(self, data: list[float], rows: int, cols: int) -> TensorHandle: ...

    def tensor_to_f32(self, tensor: TensorHandle, out: list[float]) -> int: ...

    def tensor_shape(self, tensor: TensorHandle) -> list[int]: ...

    def tensor_dtype(self, tensor: TensorHandle) -> DType: ...

    def rms_norm(self, input_: TensorHandle, weight: TensorHandle, eps: float) -> TensorHandle: ...

    def apply_rope(
        self, input_: TensorHandle, position: int, head_dim: int, theta: float
    ) -> TensorHandle: ...

    def attention_decode(
        self,
        query: TensorHandle,
        key_cache: TensorHandle,
        value_cache: TensorHandle,
        seq_len: int,
        head_dim: int,
        scale: float,
    ) -> TensorHandle: ...

    def gemv(
        self, matrix: WeightStorage, vector: TensorHandle, rows: int, cols: int
    ) -> TensorHandle: ...

    def gemm(
        self, a: TensorHandle, b: TensorHandle, rows: int, shared_dim: int, cols: int
    ) -> TensorHandle: ...

    def add(self, a: TensorHandle, b: TensorHandle) -> TensorHandle: ...

    def mul(self, a: TensorHandle, b: TensorHandle) -> TensorHandle: ...

    def sigmoid(self, x: TensorHandle) -> TensorHandle: ...

    def softmax(self, x: TensorHandle) -> TensorHandle: ...

    def synchronize(self) -> None: ...
