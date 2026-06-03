import struct

from oxidize_python.core.quantization._f16 import f16_bits_to_f32
from oxidize_python.core.quantization.dequant_simple import (
    dequant_f16,
    dequant_f32,
    dequant_q4_0,
    dequant_q4_1,
    dequant_q5_0,
    dequant_q5_1,
    dequant_q8_0,
    dequant_q8_k,
)
from oxidize_python.core.quantization.types import (
    BLOCK_IQ1_M_SIZE,
    BLOCK_IQ1_S_SIZE,
    BLOCK_IQ2_S_SIZE,
    BLOCK_IQ2_XS_SIZE,
    BLOCK_IQ2_XXS_SIZE,
    BLOCK_IQ3_S_SIZE,
    BLOCK_IQ3_XXS_SIZE,
    BLOCK_IQ4_NL_SIZE,
    BLOCK_IQ4_XS_SIZE,
    BLOCK_NVFP4_SIZE,
    BLOCK_Q2_K_SIZE,
    BLOCK_Q3_K_SIZE,
    BLOCK_Q4_K_SIZE,
    BLOCK_Q5_K_SIZE,
    BLOCK_Q6_K_SIZE,
    E2M1_DOUBLED_VALUES,
    QK_K,
    QK_NVFP4,
    Error,
    Type,
)


def dequant_q2_k(input_: bytes | bytearray, output: list[float]) -> None:
    if len(input_) % BLOCK_Q2_K_SIZE != 0:
        raise Error("q2_k input not aligned")
    blocks = len(input_) // BLOCK_Q2_K_SIZE
    if len(output) < blocks * QK_K:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_Q2_K_SIZE : (b + 1) * BLOCK_Q2_K_SIZE]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 80)[0])
        min_v = f16_bits_to_f32(struct.unpack_from("<H", blk, 82)[0])
        scales = blk[0:16]
        qs = blk[16:80]
        out = output[b * QK_K : (b + 1) * QK_K]
        q_ptr = 0
        is_ = 0
        for outer in range(2):
            qs_base = outer * 32
            for _ in range(4):
                sc1 = scales[is_]
                dl1 = d * float(sc1 & 0xF)
                ml1 = min_v * float(sc1 >> 4)
                is_ += 1
                sc2 = scales[is_]
                dl2 = d * float(sc2 & 0xF)
                ml2 = min_v * float(sc2 >> 4)
                is_ += 1
                shift = ((is_ // 2 - 1) % 4) * 2
                for l in range(16):
                    out[q_ptr + l] = dl1 * float((qs[qs_base + l] >> shift) & 3) - ml1
                for l in range(16):
                    out[q_ptr + 16 + l] = dl2 * float((qs[qs_base + 16 + l] >> shift) & 3) - ml2
                q_ptr += 32


def dequant_q3_k(input_: bytes | bytearray, output: list[float]) -> None:
    if len(input_) % BLOCK_Q3_K_SIZE != 0:
        raise Error("q3_k input not aligned")
    blocks = len(input_) // BLOCK_Q3_K_SIZE
    if len(output) < blocks * QK_K:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_Q3_K_SIZE : (b + 1) * BLOCK_Q3_K_SIZE]
        d_all = f16_bits_to_f32(struct.unpack_from("<H", blk, 108)[0])
        hmask = blk[0:32]
        qs = blk[32:96]
        scales_raw = [
            struct.unpack_from("<I", blk, 96)[0],
            struct.unpack_from("<I", blk, 100)[0],
            struct.unpack_from("<I", blk, 104)[0],
            0,
        ]
        tmp = scales_raw[2]
        scales_raw[2] = ((scales_raw[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4)
        scales_raw[3] = ((scales_raw[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4)
        scales_raw[0] = (scales_raw[0] & 0x0F0F0F0F) | ((tmp & 0x03030303) << 4)
        scales_raw[1] = (scales_raw[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4)
        scale_bytes = b"".join(struct.pack("<I", s) for s in scales_raw)
        scales = [struct.unpack_from("b", scale_bytes, i)[0] for i in range(16)]
        out = output[b * QK_K : (b + 1) * QK_K]
        q_ptr = 0
        is_ = 0
        m = 1
        for _ in range(2):
            for _ in range(4):
                dl = d_all * float(scales[is_] - 32)
                is_ += 1
                shift = ((is_ - 1) % 4) * 2
                for l in range(16):
                    qv = (qs[l] >> shift) & 3
                    hbit = 0 if (hmask[l] & m) else 4
                    out[q_ptr + l] = dl * float(qv - hbit)
                dl2 = d_all * float(scales[is_] - 32)
                is_ += 1
                for l in range(16):
                    qv = (qs[l + 16] >> shift) & 3
                    hbit = 0 if (hmask[l + 16] & m) else 4
                    out[q_ptr + 16 + l] = dl2 * float(qv - hbit)
                q_ptr += 32
                m <<= 1


def _scale_min_k4(j: int, scales: bytes | bytearray) -> tuple[int, int]:
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4), (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)


def dequant_q4_k(input_: bytes | bytearray, output: list[float]) -> None:
    if len(input_) % BLOCK_Q4_K_SIZE != 0:
        raise Error("q4_k input not aligned")
    blocks = len(input_) // BLOCK_Q4_K_SIZE
    if len(output) < blocks * QK_K:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_Q4_K_SIZE : (b + 1) * BLOCK_Q4_K_SIZE]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 0)[0])
        min_v = f16_bits_to_f32(struct.unpack_from("<H", blk, 2)[0])
        scales = blk[4:16]
        qs = blk[16:144]
        out = output[b * QK_K : (b + 1) * QK_K]
        q_ptr = 0
        is_ = 0
        for _ in range(4):
            sc1, m1 = _scale_min_k4(is_, scales)
            sc2, m2 = _scale_min_k4(is_ + 1, scales)
            d1 = d * float(sc1)
            min1 = min_v * float(m1)
            d2 = d * float(sc2)
            min2 = min_v * float(m2)
            for l in range(32):
                out[q_ptr + l] = d1 * float(qs[l] & 0xF) - min1
            for l in range(32):
                out[q_ptr + 32 + l] = d2 * float(qs[l] >> 4) - min2
            q_ptr += 64
            is_ += 2


def dequant_q5_k(input_: bytes | bytearray, output: list[float]) -> None:
    if len(input_) % BLOCK_Q5_K_SIZE != 0:
        raise Error("q5_k input not aligned")
    blocks = len(input_) // BLOCK_Q5_K_SIZE
    if len(output) < blocks * QK_K:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_Q5_K_SIZE : (b + 1) * BLOCK_Q5_K_SIZE]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 0)[0])
        min_v = f16_bits_to_f32(struct.unpack_from("<H", blk, 2)[0])
        scales = blk[4:16]
        qh = blk[16:48]
        qs = blk[48:176]
        out = output[b * QK_K : (b + 1) * QK_K]
        q_ptr = 0
        is_ = 0
        u1, u2 = 1, 2
        for _ in range(4):
            sc1, m1 = _scale_min_k4(is_, scales)
            sc2, m2 = _scale_min_k4(is_ + 1, scales)
            d1 = d * float(sc1)
            min1 = min_v * float(m1)
            d2 = d * float(sc2)
            min2 = min_v * float(m2)
            for l in range(32):
                qv1 = qs[l] & 0xF
                if qh[l] & u1:
                    qv1 += 16
                out[q_ptr + l] = d1 * float(qv1) - min1
            for l in range(32):
                qv2 = qs[l] >> 4
                if qh[l] & u2:
                    qv2 += 16
                out[q_ptr + 32 + l] = d2 * float(qv2) - min2
            q_ptr += 64
            is_ += 2
            u1 <<= 2
            u2 <<= 2


def dequant_q6_k(input_: bytes | bytearray, output: list[float]) -> None:
    if len(input_) % BLOCK_Q6_K_SIZE != 0:
        raise Error("q6_k input not aligned")
    blocks = len(input_) // BLOCK_Q6_K_SIZE
    if len(output) < blocks * QK_K:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_Q6_K_SIZE : (b + 1) * BLOCK_Q6_K_SIZE]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 208)[0])
        ql = blk[0:128]
        qh = blk[128:192]
        sc = [struct.unpack_from("b", blk, 192 + i)[0] for i in range(16)]
        out = output[b * QK_K : (b + 1) * QK_K]
        q_ptr = 0
        for _ in range(2):
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[l] & 0xF) | (((qh[l] & 3) << 4))) - 32
                q2 = ((ql[l + 32] & 0xF) | ((((qh[l] >> 2) & 3) << 4))) - 32
                q3 = ((ql[l] >> 4) | ((((qh[l] >> 4) & 3) << 4))) - 32
                q4 = ((ql[l + 32] >> 4) | ((((qh[l] >> 6) & 3) << 4))) - 32
                out[q_ptr + l] = d * float(sc[is_]) * float(q1)
                out[q_ptr + 32 + l] = d * float(sc[is_ + 2]) * float(q2)
                out[q_ptr + 64 + l] = d * float(sc[is_ + 4]) * float(q3)
                out[q_ptr + 96 + l] = d * float(sc[is_ + 6]) * float(q4)
            q_ptr += 128


def dequant_nvfp4(input_: bytes | bytearray, output: list[float]) -> None:
    if len(input_) % BLOCK_NVFP4_SIZE != 0:
        raise Error("nvfp4 input not aligned")
    blocks = len(input_) // BLOCK_NVFP4_SIZE
    if len(output) < blocks * QK_NVFP4:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * BLOCK_NVFP4_SIZE : (b + 1) * BLOCK_NVFP4_SIZE]
        scale = f16_bits_to_f32(struct.unpack_from("<H", blk, 0)[0])
        for i in range(QK_NVFP4):
            qs = blk[2 + i // 2]
            nib = qs & 0x0F if i % 2 == 0 else (qs >> 4) & 0x0F
            v = E2M1_DOUBLED_VALUES[nib] * 0.5
            output[b * QK_NVFP4 + i] = v * scale


def _dequant_iq_simple(input_: bytes | bytearray, output: list[float], block_size: int) -> None:
    if len(input_) % block_size != 0:
        raise Error("iq input not aligned")
    blocks = len(input_) // block_size
    if len(output) < blocks * QK_K:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * block_size : (b + 1) * block_size]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 0)[0])
        for i in range(QK_K):
            output[b * QK_K + i] = float(struct.unpack_from("b", blk, 2 + i)[0]) * d


def dequant_iq1_s(input_: bytes | bytearray, output: list[float]) -> None:
    _dequant_iq_simple(input_, output, BLOCK_IQ1_S_SIZE)


def dequant_iq1_m(input_: bytes | bytearray, output: list[float]) -> None:
    _dequant_iq_simple(input_, output, BLOCK_IQ1_M_SIZE)


def dequant_iq2_xxs(input_: bytes | bytearray, output: list[float]) -> None:
    _dequant_iq_simple(input_, output, BLOCK_IQ2_XXS_SIZE)


def dequant_iq2_xs(input_: bytes | bytearray, output: list[float]) -> None:
    _dequant_iq_simple(input_, output, BLOCK_IQ2_XS_SIZE)


def dequant_iq2_s(input_: bytes | bytearray, output: list[float]) -> None:
    _dequant_iq_simple(input_, output, BLOCK_IQ2_S_SIZE)


def dequant_iq3_xxs(input_: bytes | bytearray, output: list[float]) -> None:
    _dequant_iq_simple(input_, output, BLOCK_IQ3_XXS_SIZE)


def dequant_iq3_s(input_: bytes | bytearray, output: list[float]) -> None:
    _dequant_iq_simple(input_, output, BLOCK_IQ3_S_SIZE)


def _dequant_iq4(input_: bytes | bytearray, output: list[float], block_size: int) -> None:
    if len(input_) % block_size != 0:
        raise Error("iq4 input not aligned")
    blocks = len(input_) // block_size
    if len(output) < blocks * QK_K:
        raise Error("output too small")
    for b in range(blocks):
        blk = input_[b * block_size : (b + 1) * block_size]
        d = f16_bits_to_f32(struct.unpack_from("<H", blk, 0)[0])
        for i in range(QK_K):
            qs = blk[2 + i // 2]
            nib = qs & 0x0F if i % 2 == 0 else (qs >> 4) & 0x0F
            output[b * QK_K + i] = float(nib) * d


def dequant_iq4_nl(input_: bytes | bytearray, output: list[float]) -> None:
    _dequant_iq4(input_, output, BLOCK_IQ4_NL_SIZE)


def dequant_iq4_xs(input_: bytes | bytearray, output: list[float]) -> None:
    _dequant_iq4(input_, output, BLOCK_IQ4_XS_SIZE)


def dequantize(t: Type, input_: bytes | bytearray, output: list[float]) -> None:
    dispatch = {
        Type.F32: dequant_f32,
        Type.F16: dequant_f16,
        Type.Q4_0: dequant_q4_0,
        Type.Q4_1: dequant_q4_1,
        Type.Q5_0: dequant_q5_0,
        Type.Q5_1: dequant_q5_1,
        Type.Q8_0: dequant_q8_0,
        Type.Q8_K: dequant_q8_k,
        Type.Q2_K: dequant_q2_k,
        Type.Q3_K_S: dequant_q3_k,
        Type.Q3_K_M: dequant_q3_k,
        Type.Q3_K_L: dequant_q3_k,
        Type.Q4_K_S: dequant_q4_k,
        Type.Q4_K_M: dequant_q4_k,
        Type.Q5_K_S: dequant_q5_k,
        Type.Q5_K_M: dequant_q5_k,
        Type.Q6_K: dequant_q6_k,
        Type.NVFP4: dequant_nvfp4,
        Type.IQ1_S: dequant_iq1_s,
        Type.IQ1_M: dequant_iq1_m,
        Type.IQ2_XXS: dequant_iq2_xxs,
        Type.IQ2_XS: dequant_iq2_xs,
        Type.IQ2_S: dequant_iq2_s,
        Type.IQ3_XXS: dequant_iq3_xxs,
        Type.IQ3_S: dequant_iq3_s,
        Type.IQ4_NL: dequant_iq4_nl,
        Type.IQ4_XS: dequant_iq4_xs,
    }
    fn = dispatch.get(t)
    if fn is None:
        raise Error(f"unsupported type {t.name}")
    fn(input_, output)
