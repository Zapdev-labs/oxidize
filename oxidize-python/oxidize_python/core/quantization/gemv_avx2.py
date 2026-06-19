"""Fused Q4_K x Q8_K integer GEMV kernels.

Mirrors oxidize-golang/core/quantization/gemv_avx2.go, which in turn mirrors the
AVX2 kernels ``q4_k_q8_k_row_dot_avx2`` / ``q4_k_q8_k_row_dot_x4_avx2`` in
oxidize-core. The results are bit-equivalent to dequantizing the Q4_K weight row
and the Q8_K input vector and taking a float dot product, but the heavy work
stays in integer arithmetic per super-block (the perf win that the AVX2 paths
exploit).

Pure Python has no AVX2 intrinsics; these helpers implement the same
integer-domain math the Go/Rust kernels use, so the numerical results match.
"""

from __future__ import annotations

import struct

from oxidize_python.core.quantization._f16 import f16_bits_to_f32
from oxidize_python.core.quantization.dequant_k import _scale_min_k4
from oxidize_python.core.quantization.types import (
    BLOCK_Q4_K_SIZE,
    BLOCK_Q8_K_SIZE,
    QK_K,
)


def _read_q8k_bsum(block: bytes | bytearray | memoryview, base: int, idx: int) -> int:
    return struct.unpack_from("<h", block, base + 4 + QK_K + idx * 2)[0]


def _i8(b: int) -> int:
    return b - 256 if b >= 128 else b


def q4_k_q8k_row_dot(
    row: bytes | bytearray | memoryview,
    blocks_per_row: int,
    q8k: bytes | bytearray | memoryview,
) -> float:
    """Dot one Q4_K weight row against a Q8_K-quantized input vector.

    Mirrors ``Q4KQ8KRowDot`` (Go) / ``q4_k_q8_k_row_dot_avx2`` (Rust).

      row            : ``blocks_per_row`` Q4_K blocks (144 bytes each)
      blocks_per_row : number of 256-element super-blocks in the row
      q8k            : ``blocks_per_row`` Q8_K blocks (292 bytes each)
    """
    acc = 0.0
    for block_idx in range(blocks_per_row):
        w_off = block_idx * BLOCK_Q4_K_SIZE
        q8_off = block_idx * BLOCK_Q8_K_SIZE
        d_w = f16_bits_to_f32(struct.unpack_from("<H", row, w_off)[0])
        dmin_w = f16_bits_to_f32(struct.unpack_from("<H", row, w_off + 2)[0])
        d_q8 = struct.unpack_from("<f", q8k, q8_off)[0]
        scales = row[w_off + 4 : w_off + 16]
        qs = row[w_off + 16 : w_off + 144]
        q8_base = q8_off + 4

        pos_acc = 0
        min_acc = 0
        for gp in range(4):
            g1 = gp * 2
            g2 = g1 + 1
            s1, ms1 = _scale_min_k4(g1, scales)
            s2, ms2 = _scale_min_k4(g2, scales)
            base = gp * 32
            p1 = 0
            p2 = 0
            for l in range(32):
                packed = qs[base + l]
                p1 += (packed & 0x0F) * _i8(q8k[q8_base + g1 * 32 + l])
                p2 += (packed >> 4) * _i8(q8k[q8_base + g2 * 32 + l])
            pos_acc += s1 * p1
            pos_acc += s2 * p2

            bs1 = _read_q8k_bsum(q8k, q8_off, g1 * 2) + _read_q8k_bsum(q8k, q8_off, g1 * 2 + 1)
            bs2 = _read_q8k_bsum(q8k, q8_off, g2 * 2) + _read_q8k_bsum(q8k, q8_off, g2 * 2 + 1)
            min_acc += ms1 * bs1
            min_acc += ms2 * bs2
        acc += d_w * d_q8 * float(pos_acc) - dmin_w * d_q8 * float(min_acc)
    return acc


def q4_k_q8k_row_dot_x4(
    rows: bytes | bytearray | memoryview,
    row_bytes: int,
    blocks_per_row: int,
    q8k: bytes | bytearray | memoryview,
) -> list[float]:
    """Dot 4 consecutive Q4_K weight rows against one shared Q8_K input vector.

    The Q8_K sub-group loads and bsum pair-sums are computed once per block and
    reused across all four rows (cache-locality win), mirroring
    ``Q4KQ8KRowDotX4`` (Go) / ``q4_k_q8_k_row_dot_x4_avx2`` (Rust).

      rows         : at least 4*row_bytes bytes (4 Q4_K rows row_bytes apart)
      row_bytes    : stride between rows (blocks_per_row*144 typically)
      blocks_per_row : super-blocks per row
      q8k          : ``blocks_per_row`` Q8_K blocks
    """
    pos = [0.0, 0.0, 0.0, 0.0]
    neg = [0.0, 0.0, 0.0, 0.0]
    for block_idx in range(blocks_per_row):
        q8_off = block_idx * BLOCK_Q8_K_SIZE
        d_q8 = struct.unpack_from("<f", q8k, q8_off)[0]
        q8_base = q8_off + 4

        # Shared bsum pair-sums for all 8 groups.
        bs = [
            _read_q8k_bsum(q8k, q8_off, g * 2) + _read_q8k_bsum(q8k, q8_off, g * 2 + 1)
            for g in range(8)
        ]
        # Shared int8 view of this block's quants.
        q8 = [_i8(q8k[q8_base + i]) for i in range(QK_K)]

        for r in range(4):
            w_off = r * row_bytes + block_idx * BLOCK_Q4_K_SIZE
            d_w = f16_bits_to_f32(struct.unpack_from("<H", rows, w_off)[0])
            dmin_w = f16_bits_to_f32(struct.unpack_from("<H", rows, w_off + 2)[0])
            scales = rows[w_off + 4 : w_off + 16]
            qs = rows[w_off + 16 : w_off + 144]

            pos_acc = 0
            min_acc = 0
            for gp in range(4):
                g1 = gp * 2
                g2 = g1 + 1
                s1, ms1 = _scale_min_k4(g1, scales)
                s2, ms2 = _scale_min_k4(g2, scales)
                base = gp * 32
                p1 = 0
                p2 = 0
                for l in range(32):
                    packed = qs[base + l]
                    p1 += (packed & 0x0F) * q8[g1 * 32 + l]
                    p2 += (packed >> 4) * q8[g2 * 32 + l]
                pos_acc += s1 * p1
                pos_acc += s2 * p2
                min_acc += ms1 * bs[g1]
                min_acc += ms2 * bs[g2]
            pos[r] += d_w * d_q8 * float(pos_acc)
            neg[r] += dmin_w * d_q8 * float(min_acc)
    return [pos[r] - neg[r] for r in range(4)]
