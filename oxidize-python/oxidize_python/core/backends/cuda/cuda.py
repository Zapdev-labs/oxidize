"""CUDA backend stub mirroring oxidize-golang/core/backends/cuda."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


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
    _matrix: list[float],
    _vector: list[float],
    _rows: int,
    _cols: int,
    _output: list[float],
) -> None:
    raise GemvCudaError("cuda backend not linked")


def gemm_f32_cuda(
    _left: list[float],
    _right: list[float],
    _rows: int,
    _shared: int,
    _cols: int,
    _output: list[float],
) -> None:
    raise GemmCudaError("cuda backend not linked")


def gemv_quantized_cuda(
    _qbytes: bytes,
    _ggml_type: int,
    _vector: list[float],
    _rows: int,
    _cols: int,
    _output: list[float],
) -> None:
    raise GemvCudaError("cuda backend not linked")


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
