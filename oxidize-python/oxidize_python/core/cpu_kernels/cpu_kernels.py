"""Fused CPU kernels mirroring oxidize-golang/core/cpu_kernels."""

from __future__ import annotations

from enum import IntEnum

from oxidize_python.core.simd import simd
from oxidize_python.core.tensor import gemm_f32, gemv_f32_transposed, rms_norm_f32


class Kernel(IntEnum):
    OPERATOR_FUSION = 0
    WORKSPACE_REUSE = 1
    AVX2 = 2
    AVX512 = 3

    def __str__(self) -> str:
        names = {
            Kernel.OPERATOR_FUSION: "operator_fusion",
            Kernel.WORKSPACE_REUSE: "workspace_reuse",
            Kernel.AVX2: "avx2",
            Kernel.AVX512: "avx512",
        }
        return names.get(self, "unknown")


def implemented_kernels() -> list[Kernel]:
    k = [Kernel.OPERATOR_FUSION, Kernel.WORKSPACE_REUSE]
    pref = simd.preferred()
    if pref == simd.Backend.AVX2:
        k.append(Kernel.AVX2)
    if pref == simd.Backend.AVX512F:
        k.append(Kernel.AVX512)
    return k


class Workspace:
    def __init__(self, capacity: int = 0) -> None:
        self._scratch: list[float] = []
        self._capacity = capacity

    def get(self, n: int) -> list[float]:
        if len(self._scratch) < n:
            self._scratch = [0.0] * n
        else:
            self._scratch = self._scratch[:n]
        return self._scratch

    def capacity(self) -> int:
        return len(self._scratch)


class FusedError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"fused: {message}")


def fused_rms_norm_gemv_f32_transposed(
    input_: list[float],
    norm_weight: list[float],
    eps: float,
    matrix: list[float],
    rows: int,
    cols: int,
    workspace: Workspace,
    output: list[float],
) -> None:
    normalized = workspace.get(rows)
    rms_norm_f32(input_[:rows], norm_weight[:rows], normalized, eps)
    gemv_f32_transposed(matrix, rows, cols, normalized, output)


def mat_mul_reuse_workspace(
    workspace: Workspace,
    left: list[float],
    right: list[float],
    rows: int,
    shared: int,
    cols: int,
) -> list[float]:
    out = workspace.get(rows * cols)
    gemm_f32(left, right, rows, shared, cols, out)
    return out


def dot_product_avx2_or_scalar(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def dot_product_avx512_or_scalar(a: list[float], b: list[float]) -> float:
    return dot_product_avx2_or_scalar(a, b)
