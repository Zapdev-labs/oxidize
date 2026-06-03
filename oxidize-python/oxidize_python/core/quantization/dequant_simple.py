import struct

from oxidize_python.core.quantization._f16 import f16_bits_to_f32
from oxidize_python.core.quantization.types import (
    BLOCK_Q4_0_SIZE,
    BLOCK_Q4_1_SIZE,
    BLOCK_Q5_0_SIZE,
    BLOCK_Q5_1_SIZE,
    BLOCK_Q8_0_SIZE,
    BLOCK_Q8_K_SIZE,
    QK4_0,
    QK4_1,
    QK5_0,
    QK5_1,
    QK8_0,
    QK_K,
    Error,
)


def dequant_f32(input_: bytes | bytearray, output: list[float]) -> None:
    if len(output) * 4 < len(input_):
        raise Error("output too small")
    n = len(input_) // 4
    for i in range(n):
        output[i] = struct.unpack_from("<f", input_, i * 4)[0]


def dequant_f16(input_: bytes | bytearray, output: list[float]) -> None:
    if not input_:
        return
    if len(output) * 2 < len(input_):
        raise Error("output too small")
    n = len(input_) // 2
    for i in range(n):
        bits = struct.unpack_from("<H", input_, i * 2)[0]
        output[i] = f16_bits_to_f32(bits)


def dequant_q4_0(input_: bytes | bytearray, output: list[float]) -> None:
    if not input_:
        return
    if len(input_) % BLOCK_Q4_0_SIZE != 0:
        raise Error("q4_0 input length not aligned")
    blocks = len(input_) // BLOCK_Q4_0_SIZE
    if len(output) < blocks * QK4_0:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_Q4_0_SIZE : (b + 1) * BLOCK_Q4_0_SIZE]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 0)[0])
        qs = blk[2 : 2 + QK4_0 // 2]
        out = output[b * QK4_0 : (b + 1) * QK4_0]
        for i in range(QK4_0 // 2):
            v0 = int((qs[i] & 0x0F) - 8)
            v1 = int(((qs[i] >> 4) & 0x0F) - 8)
            out[i * 2] = float(v0) * d
            out[i * 2 + 1] = float(v1) * d


def dequant_q4_1(input_: bytes | bytearray, output: list[float]) -> None:
    if len(input_) % BLOCK_Q4_1_SIZE != 0:
        raise Error("q4_1 input not aligned")
    blocks = len(input_) // BLOCK_Q4_1_SIZE
    if len(output) < blocks * QK4_1:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_Q4_1_SIZE : (b + 1) * BLOCK_Q4_1_SIZE]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 0)[0])
        m = f16_bits_to_f32(struct.unpack_from("<H", blk, 2)[0])
        qs = blk[4 : 4 + QK4_1 // 2]
        out = output[b * QK4_1 : (b + 1) * QK4_1]
        for i in range(QK4_1 // 2):
            v0 = float(qs[i] & 0x0F)
            v1 = float((qs[i] >> 4) & 0x0F)
            out[i * 2] = v0 * d + m
            out[i * 2 + 1] = v1 * d + m


def dequant_q5_0(input_: bytes | bytearray, output: list[float]) -> None:
    if len(input_) % BLOCK_Q5_0_SIZE != 0:
        raise Error("q5_0 input not aligned")
    blocks = len(input_) // BLOCK_Q5_0_SIZE
    if len(output) < blocks * QK5_0:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_Q5_0_SIZE : (b + 1) * BLOCK_Q5_0_SIZE]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 0)[0])
        qh = struct.unpack_from("<I", blk, 2)[0]
        qs = blk[6 : 6 + QK5_0 // 2]
        out = output[b * QK5_0 : (b + 1) * QK5_0]
        for i in range(QK5_0):
            high = (qh >> i) & 1
            lo = qs[i // 2] & 0x0F if i % 2 == 0 else (qs[i // 2] >> 4) & 0x0F
            v = float(int((high << 4) | lo) - 16)
            out[i] = v * d


def dequant_q5_1(input_: bytes | bytearray, output: list[float]) -> None:
    if len(input_) % BLOCK_Q5_1_SIZE != 0:
        raise Error("q5_1 input not aligned")
    blocks = len(input_) // BLOCK_Q5_1_SIZE
    if len(output) < blocks * QK5_1:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_Q5_1_SIZE : (b + 1) * BLOCK_Q5_1_SIZE]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 0)[0])
        m = f16_bits_to_f32(struct.unpack_from("<H", blk, 2)[0])
        qh = struct.unpack_from("<I", blk, 4)[0]
        qs = blk[8 : 8 + QK5_1 // 2]
        out = output[b * QK5_1 : (b + 1) * QK5_1]
        for i in range(QK5_1):
            high = (qh >> i) & 1
            lo = qs[i // 2] & 0x0F if i % 2 == 0 else (qs[i // 2] >> 4) & 0x0F
            v = float((high << 4) | lo) * d + m
            out[i] = v


def dequant_q8_0(input_: bytes | bytearray, output: list[float]) -> None:
    if len(input_) % BLOCK_Q8_0_SIZE != 0:
        raise Error("q8_0 input not aligned")
    blocks = len(input_) // BLOCK_Q8_0_SIZE
    if len(output) < blocks * QK8_0:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_Q8_0_SIZE : (b + 1) * BLOCK_Q8_0_SIZE]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 0)[0])
        qs = blk[2 : 2 + QK8_0]
        out = output[b * QK8_0 : (b + 1) * QK8_0]
        for i, q in enumerate(qs):
            out[i] = float(int(q if q < 128 else q - 256)) * d


def dequant_q8_k(input_: bytes | bytearray, output: list[float]) -> None:
    if len(input_) % BLOCK_Q8_K_SIZE != 0:
        raise Error("q8_k input not aligned")
    blocks = len(input_) // BLOCK_Q8_K_SIZE
    if len(output) < blocks * QK_K:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_Q8_K_SIZE : (b + 1) * BLOCK_Q8_K_SIZE]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 0)[0])
        qs = blk[2 : 2 + QK_K]
        out = output[b * QK_K : (b + 1) * QK_K]
        for i, q in enumerate(qs):
            out[i] = float(int(q if q < 128 else q - 256)) * d
