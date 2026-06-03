"""Scalar quantization mirroring oxidize-golang/core/quantization/quantize.go."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from oxidize_python.core.quantization.types import (
    BLOCK_Q2_K_SIZE,
    BLOCK_Q3_K_SIZE,
    BLOCK_Q4_0_SIZE,
    BLOCK_Q4_1_SIZE,
    BLOCK_Q4_K_SIZE,
    BLOCK_Q5_0_SIZE,
    BLOCK_Q5_1_SIZE,
    BLOCK_Q5_K_SIZE,
    BLOCK_Q6_K_SIZE,
    BLOCK_Q8_0_SIZE,
    QK4_0,
    QK4_1,
    QK5_0,
    QK5_1,
    QK8_0,
    QK_K,
    Error,
    Type,
    max_abs,
    min_max,
)

if TYPE_CHECKING:
    from oxidize_python.core.quantization.imatrix import IMatrix


def f32_to_f16_bits(v: float) -> int:

    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    sign = (bits >> 31) & 1
    exp = ((bits >> 23) & 0xFF) - 127 + 15
    mant = bits & 0x7FFFFF
    if exp <= 0:
        if exp < -10:
            return sign << 15
        mant = (mant | 0x800000) >> (1 - exp)
        return (sign << 15) | (mant >> 13)
    if exp >= 0x1F:
        return (sign << 15) | 0x7C00
    return (sign << 15) | (exp << 10) | (mant >> 13)


def _clamp_int(v: float, lo: int, hi: int) -> int:
    iv = int(v)
    return max(lo, min(hi, iv))


def quantize_scalar(
    t: Type,
    input_: list[float],
    output: bytearray | bytes | memoryview,
    imatrix: IMatrix | None,
) -> None:
    out = memoryview(output) if not isinstance(output, memoryview) else output
    match t:
        case Type.F32:
            _quantize_f32(input_, out)
        case Type.F16:
            _quantize_f16(input_, out)
        case Type.Q4_0:
            _quantize_q4_0(input_, out)
        case Type.Q4_1:
            _quantize_q4_1(input_, out)
        case Type.Q5_0:
            _quantize_q5_0(input_, out)
        case Type.Q5_1:
            _quantize_q5_1(input_, out)
        case Type.Q8_0:
            _quantize_q8_0(input_, out)
        case Type.Q2_K:
            _quantize_q2_k(input_, out, imatrix)
        case Type.Q3_K_S | Type.Q3_K_M | Type.Q3_K_L:
            _quantize_q3_k(input_, out, imatrix)
        case Type.Q4_K_S | Type.Q4_K_M:
            _quantize_q4_k(input_, out, imatrix)
        case Type.Q5_K_S | Type.Q5_K_M:
            _quantize_q5_k(input_, out, imatrix)
        case Type.Q6_K:
            _quantize_q6_k(input_, out, imatrix)
        case _:
            raise Error(f"quantize unsupported type {t.name}")


def _quantize_f32(input_: list[float], output: memoryview) -> None:
    if len(output) < len(input_) * 4:
        raise Error("output too small for f32")
    for i, v in enumerate(input_):
        struct.pack_into("<f", output, i * 4, v)


def _quantize_f16(input_: list[float], output: memoryview) -> None:
    if len(output) < len(input_) * 2:
        raise Error("output too small for f16")
    for i, v in enumerate(input_):
        struct.pack_into("<H", output, i * 2, f32_to_f16_bits(v))


def _quantize_q4_0(input_: list[float], output: memoryview) -> None:
    if len(input_) % QK4_0:
        raise Error("input length not aligned to Q4_0 block")
    blocks = len(input_) // QK4_0
    if len(output) < blocks * BLOCK_Q4_0_SIZE:
        raise Error("output too small for Q4_0")
    for b in range(blocks):
        src = input_[b * QK4_0 : (b + 1) * QK4_0]
        base = b * BLOCK_Q4_0_SIZE
        scale = max_abs(src) / 7 or 1.0
        inv = 7 / scale
        struct.pack_into("<H", output, base, f32_to_f16_bits(scale))
        for i in range(QK4_0 // 2):
            v0 = _clamp_int(src[i * 2] * inv + 8, 0, 15)
            v1 = _clamp_int(src[i * 2 + 1] * inv + 8, 0, 15)
            output[base + 2 + i] = v0 | (v1 << 4)


def _quantize_q4_1(input_: list[float], output: memoryview) -> None:
    if len(input_) % QK4_1:
        raise Error("input length not aligned to Q4_1 block")
    blocks = len(input_) // QK4_1
    for b in range(blocks):
        src = input_[b * QK4_1 : (b + 1) * QK4_1]
        base = b * BLOCK_Q4_1_SIZE
        mn, mx = min_max(src)
        scale = (mx - mn) / 15 or 1.0
        inv = 15 / scale
        struct.pack_into("<H", output, base, f32_to_f16_bits(scale))
        struct.pack_into("<H", output, base + 2, f32_to_f16_bits(mn))
        for i in range(QK4_1 // 2):
            v0 = _clamp_int((src[i * 2] - mn) * inv, 0, 15)
            v1 = _clamp_int((src[i * 2 + 1] - mn) * inv, 0, 15)
            output[base + 4 + i] = v0 | (v1 << 4)


def _quantize_q5_0(input_: list[float], output: memoryview) -> None:
    if len(input_) % QK5_0:
        raise Error("input length not aligned to Q5_0 block")
    blocks = len(input_) // QK5_0
    for b in range(blocks):
        src = input_[b * QK5_0 : (b + 1) * QK5_0]
        base = b * BLOCK_Q5_0_SIZE
        scale = max_abs(src) / 15 or 1.0
        inv = 15 / scale
        struct.pack_into("<H", output, base, f32_to_f16_bits(scale))
        qh = 0
        for i in range(QK5_0):
            v = _clamp_int(src[i] * inv + 16, 0, 31)
            if v >= 16:
                qh |= 1 << i
            lo = v & 0x0F
            pos = base + 6 + i // 2
            if i % 2 == 0:
                output[pos] = (output[pos] & 0xF0) | lo
            else:
                output[pos] = (output[pos] & 0x0F) | (lo << 4)
        struct.pack_into("<I", output, base + 2, qh)


def _quantize_q5_1(input_: list[float], output: memoryview) -> None:
    if len(input_) % QK5_1:
        raise Error("input length not aligned to Q5_1 block")
    blocks = len(input_) // QK5_1
    for b in range(blocks):
        src = input_[b * QK5_1 : (b + 1) * QK5_1]
        base = b * BLOCK_Q5_1_SIZE
        mn, mx = min_max(src)
        scale = (mx - mn) / 31 or 1.0
        inv = 31 / scale
        struct.pack_into("<H", output, base, f32_to_f16_bits(scale))
        struct.pack_into("<H", output, base + 2, f32_to_f16_bits(mn))
        qh = 0
        for i in range(QK5_1):
            v = _clamp_int((src[i] - mn) * inv, 0, 31)
            if v >= 16:
                qh |= 1 << i
            lo = v & 0x0F
            pos = base + 8 + i // 2
            if i % 2 == 0:
                output[pos] = (output[pos] & 0xF0) | lo
            else:
                output[pos] = (output[pos] & 0x0F) | (lo << 4)
        struct.pack_into("<I", output, base + 4, qh)


def _quantize_q8_0(input_: list[float], output: memoryview) -> None:
    if len(input_) % QK8_0:
        raise Error("input length not aligned to Q8_0 block")
    blocks = len(input_) // QK8_0
    for b in range(blocks):
        src = input_[b * QK8_0 : (b + 1) * QK8_0]
        base = b * BLOCK_Q8_0_SIZE
        scale = max_abs(src) / 127 or 1.0
        inv = 127 / scale
        struct.pack_into("<H", output, base, f32_to_f16_bits(scale))
        for i, v in enumerate(src):
            q = _clamp_int(v * inv, -128, 127)
            output[base + 2 + i] = q & 0xFF


def _quantize_q2_k(input_: list[float], output: memoryview, _imatrix: IMatrix | None) -> None:
    if len(input_) % QK_K:
        raise Error("input length not aligned to Q2_K block")
    blocks = len(input_) // QK_K
    for b in range(blocks):
        src = input_[b * QK_K : (b + 1) * QK_K]
        base = b * BLOCK_Q2_K_SIZE
        mx = max_abs(src) or 1.0
        scale = mx / 3 or 1.0
        inv = 3 / scale
        struct.pack_into("<H", output, base + 80, f32_to_f16_bits(scale))
        struct.pack_into("<H", output, base + 82, f32_to_f16_bits(0.0))
        for i in range(QK_K // 4):
            packed = 0
            for j in range(4):
                v = _clamp_int(src[i * 4 + j] * inv, 0, 3)
                packed |= v << (j * 2)
            output[base + i] = packed


def _quantize_q3_k(input_: list[float], output: memoryview, _imatrix: IMatrix | None) -> None:
    if len(input_) % QK_K:
        raise Error("input length not aligned to Q3_K block")
    blocks = len(input_) // QK_K
    for b in range(blocks):
        src = input_[b * QK_K : (b + 1) * QK_K]
        base = b * BLOCK_Q3_K_SIZE
        mx = max_abs(src) or 1.0
        scale = mx / 7 or 1.0
        inv = 7 / scale
        struct.pack_into("<H", output, base + 108, f32_to_f16_bits(scale))
        for i in range(QK_K // 2):
            packed = 0
            for j in range(2):
                v = int(src[i * 2 + j] * inv)
                v = max(0, min(7, v))
                packed |= v << (j * 4)
            output[base + 32 + i] = packed
        for i in range(QK_K // 16):
            output[base + 96 + i] = 32


def _quantize_q4_k(input_: list[float], output: memoryview, _imatrix: IMatrix | None) -> None:
    if len(input_) % QK_K:
        raise Error("input length not aligned to Q4_K block")
    blocks = len(input_) // QK_K
    for b in range(blocks):
        src = input_[b * QK_K : (b + 1) * QK_K]
        base = b * BLOCK_Q4_K_SIZE
        mn, mx = min_max(src)
        scale = (mx - mn) / 63 or 1.0
        inv = 63 / scale
        struct.pack_into("<H", output, base, f32_to_f16_bits(scale))
        struct.pack_into("<H", output, base + 2, f32_to_f16_bits(mn))
        for i in range(QK_K // 2):
            v0 = _clamp_int((src[i * 2] - mn) * inv, 0, 63)
            v1 = _clamp_int((src[i * 2 + 1] - mn) * inv, 0, 63)
            output[base + 16 + i] = v0 | (v1 << 4)


def _quantize_q5_k(input_: list[float], output: memoryview, _imatrix: IMatrix | None) -> None:
    if len(input_) % QK_K:
        raise Error("input length not aligned to Q5_K block")
    blocks = len(input_) // QK_K
    for b in range(blocks):
        src = input_[b * QK_K : (b + 1) * QK_K]
        base = b * BLOCK_Q5_K_SIZE
        mn, mx = min_max(src)
        scale = (mx - mn) / 63 or 1.0
        inv = 63 / scale
        struct.pack_into("<H", output, base, f32_to_f16_bits(scale))
        struct.pack_into("<H", output, base + 2, f32_to_f16_bits(mn))
        for i in range(QK_K):
            v = _clamp_int((src[i] - mn) * inv, 0, 63)
            pos = base + 48 + i // 2
            if i % 2 == 0:
                output[pos] = (output[pos] & 0xF0) | (v & 0x0F)
            else:
                output[pos] = (output[pos] & 0x0F) | ((v & 0x0F) << 4)
            if v >= 32:
                output[base + 16 + i // 8] |= 1 << (i % 8)


def _quantize_q6_k(input_: list[float], output: memoryview, _imatrix: IMatrix | None) -> None:
    if len(input_) % QK_K:
        raise Error("input length not aligned to Q6_K block")
    blocks = len(input_) // QK_K
    for b in range(blocks):
        src = input_[b * QK_K : (b + 1) * QK_K]
        base = b * BLOCK_Q6_K_SIZE
        mx = max_abs(src) or 1.0
        scale = mx / 127 or 1.0
        inv = 127 / scale
        struct.pack_into("<H", output, base + 208, f32_to_f16_bits(scale))
        for i in range(QK_K // 16):
            output[base + 192 + i] = 64
        for i in range(QK_K // 2):
            packed = 0
            for j in range(2):
                v = int(src[i * 2 + j] * inv)
                v = max(-32, min(31, v))
                if v < 0:
                    v += 64
                packed |= (v & 0xFF) << (j * 4)
            output[base + i] = packed
        for i in range(QK_K // 4):
            packed = 0
            for j in range(4):
                v = int(src[i * 4 + j] * inv)
                v = max(-32, min(31, v))
                if v < 0:
                    v += 64
                packed |= (v & 0x3) << (j * 2)
            output[base + 128 + i] = packed
