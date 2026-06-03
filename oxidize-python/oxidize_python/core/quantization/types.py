"""GGUF quantization types mirroring oxidize_core::compute::quantization."""

from __future__ import annotations

from enum import IntEnum

QK8_0 = 32
QK4_0 = 32
QK4_1 = 32
QK5_0 = 32
QK5_1 = 32
QK_K = 256
QK_NVFP4 = 64
QK_NVFP4_SUB = 16
BLOCK_Q4_0_SIZE = 18
BLOCK_Q4_1_SIZE = 20
BLOCK_Q5_0_SIZE = 22
BLOCK_Q5_1_SIZE = 24
BLOCK_Q8_0_SIZE = 34
BLOCK_Q2_K_SIZE = 84
BLOCK_Q3_K_SIZE = 110
BLOCK_Q4_K_SIZE = 144
BLOCK_Q5_K_SIZE = 176
BLOCK_Q6_K_SIZE = 210
BLOCK_Q8_K_SIZE = 292
BLOCK_NVFP4_SIZE = 34
BLOCK_IQ1_S_SIZE = 50
BLOCK_IQ1_M_SIZE = 56
BLOCK_IQ2_XXS_SIZE = 66
BLOCK_IQ2_XS_SIZE = 74
BLOCK_IQ2_S_SIZE = 82
BLOCK_IQ3_XXS_SIZE = 98
BLOCK_IQ3_S_SIZE = 110
BLOCK_IQ4_NL_SIZE = 18
BLOCK_IQ4_XS_SIZE = 34

E2M1_DOUBLED_VALUES: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


class Error(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"quantization: {message}")


class Type(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 4
    Q5_1 = 5
    Q8_0 = 6
    Q2_K = 7
    Q3_K_S = 8
    Q3_K_M = 9
    Q3_K_L = 10
    Q4_K_S = 11
    Q4_K_M = 12
    Q5_K_S = 13
    Q5_K_M = 14
    Q6_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    IQ1_M = 24
    NVFP4 = 25
    Q8_K = 26
    UNKNOWN = 27

    def block_size(self) -> int:
        match self:
            case Type.F32 | Type.F16:
                return 1
            case Type.Q4_0 | Type.Q4_1 | Type.Q5_0 | Type.Q5_1 | Type.Q8_0:
                return QK4_0
            case Type.Q2_K | Type.Q3_K_S | Type.Q3_K_M | Type.Q3_K_L | Type.Q4_K_S | Type.Q4_K_M | Type.Q5_K_S | Type.Q5_K_M | Type.Q6_K | Type.Q8_K:
                return QK_K
            case Type.NVFP4:
                return QK_NVFP4
            case Type.IQ1_S | Type.IQ1_M | Type.IQ2_XXS | Type.IQ2_XS | Type.IQ2_S | Type.IQ3_XXS | Type.IQ3_S | Type.IQ4_NL | Type.IQ4_XS:
                return QK_K
        return 0

    def bytes_per_block(self) -> int:
        m = {
            Type.F32: 4,
            Type.F16: 2,
            Type.Q4_0: BLOCK_Q4_0_SIZE,
            Type.Q4_1: BLOCK_Q4_1_SIZE,
            Type.Q5_0: BLOCK_Q5_0_SIZE,
            Type.Q5_1: BLOCK_Q5_1_SIZE,
            Type.Q8_0: BLOCK_Q8_0_SIZE,
            Type.Q2_K: BLOCK_Q2_K_SIZE,
            Type.Q3_K_S: BLOCK_Q3_K_SIZE,
            Type.Q3_K_M: BLOCK_Q3_K_SIZE,
            Type.Q3_K_L: BLOCK_Q3_K_SIZE,
            Type.Q4_K_S: BLOCK_Q4_K_SIZE,
            Type.Q4_K_M: BLOCK_Q4_K_SIZE,
            Type.Q5_K_S: BLOCK_Q5_K_SIZE,
            Type.Q5_K_M: BLOCK_Q5_K_SIZE,
            Type.Q6_K: BLOCK_Q6_K_SIZE,
            Type.Q8_K: BLOCK_Q8_K_SIZE,
            Type.NVFP4: BLOCK_NVFP4_SIZE,
            Type.IQ1_S: BLOCK_IQ1_S_SIZE,
            Type.IQ1_M: BLOCK_IQ1_M_SIZE,
            Type.IQ2_XXS: BLOCK_IQ2_XXS_SIZE,
            Type.IQ2_XS: BLOCK_IQ2_XS_SIZE,
            Type.IQ2_S: BLOCK_IQ2_S_SIZE,
            Type.IQ3_XXS: BLOCK_IQ3_XXS_SIZE,
            Type.IQ3_S: BLOCK_IQ3_S_SIZE,
            Type.IQ4_NL: BLOCK_IQ4_NL_SIZE,
            Type.IQ4_XS: BLOCK_IQ4_XS_SIZE,
        }
        return m.get(self, 0)


def quantized_size(t: Type, n: int) -> int:
    if n < 0:
        raise Error("negative value count")
    if t == Type.F32:
        return n * 4
    if t == Type.F16:
        return n * 2
    block = t.block_size()
    if block == 0:
        raise Error(f"unsupported type {t.name}")
    blocks = (n + block - 1) // block
    return blocks * t.bytes_per_block()


def parse_type(name: str) -> Type:
    key = name.upper()
    table = {
        "F32": Type.F32,
        "F16": Type.F16,
        "Q4_0": Type.Q4_0,
        "Q4_1": Type.Q4_1,
        "Q5_0": Type.Q5_0,
        "Q5_1": Type.Q5_1,
        "Q8_0": Type.Q8_0,
        "Q2_K": Type.Q2_K,
        "Q3_K_S": Type.Q3_K_S,
        "Q3_K_M": Type.Q3_K_M,
        "Q3_K_L": Type.Q3_K_L,
        "Q4_K_S": Type.Q4_K_S,
        "Q4_K_M": Type.Q4_K_M,
        "Q5_K_S": Type.Q5_K_S,
        "Q5_K_M": Type.Q5_K_M,
        "Q6_K": Type.Q6_K,
        "IQ2_XXS": Type.IQ2_XXS,
        "IQ2_XS": Type.IQ2_XS,
        "IQ3_XXS": Type.IQ3_XXS,
        "IQ1_S": Type.IQ1_S,
        "IQ4_NL": Type.IQ4_NL,
        "IQ3_S": Type.IQ3_S,
        "IQ2_S": Type.IQ2_S,
        "IQ4_XS": Type.IQ4_XS,
        "IQ1_M": Type.IQ1_M,
        "NVFP4": Type.NVFP4,
        "Q8_K": Type.Q8_K,
    }
    if key not in table:
        raise Error(f"unknown type {name!r}")
    return table[key]


def from_llama_ftype(ftype: int) -> Type:
    table = {
        0: Type.F32,
        1: Type.F16,
        2: Type.Q4_0,
        3: Type.Q4_1,
        6: Type.Q5_0,
        7: Type.Q5_1,
        8: Type.Q8_0,
    }
    return table.get(ftype, Type.UNKNOWN)


def from_ggml_type(id_: int) -> Type:
    table = {
        0: Type.F32,
        1: Type.F16,
        2: Type.Q4_0,
        3: Type.Q4_1,
        6: Type.Q5_0,
        7: Type.Q5_1,
        8: Type.Q8_0,
        10: Type.Q2_K,
        11: Type.Q3_K_S,
        12: Type.Q4_K_S,
        13: Type.Q5_K_S,
        14: Type.Q6_K,
        15: Type.IQ2_XXS,
        16: Type.IQ2_XS,
        17: Type.IQ3_XXS,
        18: Type.IQ1_S,
        19: Type.IQ4_NL,
        20: Type.IQ3_S,
        21: Type.IQ2_S,
        22: Type.IQ4_XS,
        23: Type.IQ1_M,
        24: Type.NVFP4,
    }
    return table.get(id_, Type.UNKNOWN)


def max_abs(values: list[float]) -> float:
    return max((abs(v) for v in values), default=0.0)


def min_max(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return min(values), max(values)
