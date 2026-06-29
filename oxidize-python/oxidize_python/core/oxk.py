"""OXK: custom Oxidize CPU kernels for quantized GEMV.

Phase 1 scope: Q4_K x Q8_K row dots (scalar reference + NumPy fast path) and a
contiguous-range GEMV helper. The per-row math is bit-identical to the legacy
kernels in oxidize-core/src/compute/tensor.rs and the Rust oxidize-kernels crate.

This module is self-contained (no deps on other oxidize packages) so it can be
benchmarked and tested in isolation.
"""

from __future__ import annotations

import os
import struct
import sys
import threading
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Constants (match GGUF K-quants)
# ---------------------------------------------------------------------------

QK_K: Final = 256
BLOCK_Q4_K_SIZE: Final = 144
BLOCK_Q8_K_BYTES: Final = 4 + 256 + 32

# ---------------------------------------------------------------------------
# CPU vendor / ISA detection and tuning
# ---------------------------------------------------------------------------


class CpuVendor:
    INTEL = 0
    AMD = 1
    OTHER = 2


class CpuInfo:
    def __init__(
        self,
        vendor: int = CpuVendor.OTHER,
        family: int = 0,
        model: int = 0,
        stepping: int = 0,
        has_avx2: bool = False,
        has_fma: bool = False,
        has_avx512f: bool = False,
        has_avx512bw: bool = False,
        has_avx512vnni: bool = False,
        has_avxvnni: bool = False,
        use_avx512: bool = False,
    ) -> None:
        self.vendor = vendor
        self.family = family
        self.model = model
        self.stepping = stepping
        self.has_avx2 = has_avx2
        self.has_fma = has_fma
        self.has_avx512f = has_avx512f
        self.has_avx512bw = has_avx512bw
        self.has_avx512vnni = has_avx512vnni
        self.has_avxvnni = has_avxvnni
        self.use_avx512 = use_avx512


class OxkTune:
    def __init__(self, pf_bytes: int = 0, pf_nta: bool = False) -> None:
        self.pf_bytes = pf_bytes
        self.pf_nta = pf_nta


_cpu_info_once = threading.Lock()
_cpu_info_val: CpuInfo | None = None

_tune_once = threading.Lock()
_tune_val: OxkTune | None = None


def cpu_info() -> CpuInfo:
    global _cpu_info_val
    if _cpu_info_val is None:
        with _cpu_info_once:
            if _cpu_info_val is None:
                _cpu_info_val = _detect_cpuinfo()
    return _cpu_info_val


def tune() -> OxkTune:
    global _tune_val
    if _tune_val is None:
        with _tune_once:
            if _tune_val is None:
                info = cpu_info()
                default_blocks = 2
                if info.vendor == CpuVendor.INTEL:
                    default_blocks = 1
                elif info.vendor == CpuVendor.AMD:
                    default_blocks = 2
                blocks = default_blocks
                if v := os.environ.get("OXIDIZE_OXK_PF"):
                    try:
                        blocks = int(v)
                    except ValueError:
                        print(
                            f"OXIDIZE_OXK_PF={v!r} invalid; using default {default_blocks}",
                            file=sys.stderr,
                        )
                pf_nta = False
                hint = os.environ.get("OXIDIZE_OXK_PF_HINT", "")
                if hint == "nta":
                    pf_nta = True
                elif hint == "t0" or hint == "":
                    pf_nta = False
                else:
                    print(
                        f"OXIDIZE_OXK_PF_HINT={hint} unknown (use t0|nta); using t0",
                        file=sys.stderr,
                    )
                _tune_val = OxkTune(pf_bytes=blocks * BLOCK_Q4_K_SIZE, pf_nta=pf_nta)
    return _tune_val


def oxk_cpu_summary() -> str:
    info = cpu_info()
    vendor_map = {CpuVendor.INTEL: "intel", CpuVendor.AMD: "amd", CpuVendor.OTHER: "other"}
    vendor = vendor_map.get(info.vendor, "other")
    t = tune()
    pf_hint = "nta" if t.pf_nta else "t0"
    return (
        f"vendor={vendor} fam={info.family} model={info.model} step={info.stepping} "
        f"avx2={info.has_avx2} fma={info.has_fma} avx512f={info.has_avx512f} "
        f"avx512bw={info.has_avx512bw} avx512vnni={info.has_avx512vnni} "
        f"avxvnni={info.has_avxvnni} use_avx512={info.use_avx512} "
        f"pf_blocks={t.pf_bytes // BLOCK_Q4_K_SIZE} pf_hint={pf_hint}"
    )


def _detect_cpuinfo() -> CpuInfo:
    return CpuInfo()


# ---------------------------------------------------------------------------
# Tile width selection (default 16 = widest, enabled by default)
# ---------------------------------------------------------------------------

_max_tile_once = threading.Lock()
_max_tile_val: int | None = None


def max_tile() -> int:
    global _max_tile_val
    if _max_tile_val is None:
        with _max_tile_once:
            if _max_tile_val is None:
                _max_tile_val = 16
                if v := os.environ.get("OXIDIZE_OXK_TILE"):
                    try:
                        t = int(v)
                        if t in (1, 4, 8, 16):
                            _max_tile_val = t
                    except ValueError:
                        print(
                            f"OXIDIZE_OXK_TILE={v!r} invalid (use 1|4|8|16); using default 16",
                            file=sys.stderr,
                        )
    return _max_tile_val


# ---------------------------------------------------------------------------
# f16 helpers
# ---------------------------------------------------------------------------


def _f16_le_to_f32(bytes_: bytes | bytearray) -> float:
    bits = struct.unpack_from("<H", bytes_, 0)[0]
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x03FF
    if exp == 0:
        if frac == 0:
            f32_bits = sign << 31
        else:
            frac_norm = frac
            e = -14
            while (frac_norm & 0x0400) == 0:
                frac_norm <<= 1
                e -= 1
            frac_norm &= 0x03FF
            f32_bits = (sign << 31) | ((e + 127) << 23) | (frac_norm << 13)
    elif exp == 0x1F:
        f32_bits = (sign << 31) | (0xFF << 23) | (frac << 13)
    else:
        f32_bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13)
    return struct.unpack("<f", struct.pack("<I", f32_bits))[0]


# ---------------------------------------------------------------------------
# Scale/min decoding
# ---------------------------------------------------------------------------


def get_scale_min_k4(j: int, scales: bytes | bytearray) -> tuple[int, int]:
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return (
        (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
        (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
    )


# ---------------------------------------------------------------------------
# Q8_K activation quantization
# ---------------------------------------------------------------------------


def quantize_q8_k_into(vector: list[float], n_blocks: int, out: bytearray | memoryview) -> None:
    out_mv = memoryview(out) if not isinstance(out, memoryview) else out
    if len(vector) != n_blocks * QK_K:
        raise ValueError("vector length mismatch")
    if len(out_mv) < n_blocks * BLOCK_Q8_K_BYTES:
        raise ValueError("output buffer too small")
    for b in range(n_blocks):
        block_in = vector[b * QK_K : (b + 1) * QK_K]
        block_out = out_mv[b * BLOCK_Q8_K_BYTES : (b + 1) * BLOCK_Q8_K_BYTES]
        _quantize_q8_k_block(block_in, block_out)


def _quantize_q8_k_block(block_in: list[float], block_out: memoryview) -> None:
    amax = 0.0
    max_v = 0.0
    for v in block_in:
        av = abs(v)
        if av > amax:
            amax = av
            max_v = v
    if amax == 0.0:
        block_out[:4] = b"\x00\x00\x00\x00"
        for i in range(4, len(block_out)):
            block_out[i] = 0
        return
    iscale = -128.0 / max_v
    d = 1.0 / iscale
    struct.pack_into("<f", block_out, 0, d)
    qs_off = 4
    for i, v in enumerate(block_in):
        q = int(round(iscale * v))
        q = max(-128, min(127, q))
        block_out[qs_off + i] = q & 0xFF
    bsums_off = qs_off + QK_K
    for g in range(QK_K // 16):
        s = 0
        for i in range(16):
            s += int(int8(block_out[qs_off + g * 16 + i]))
        s = max(-32768, min(32767, s))
        struct.pack_into("<h", block_out, bsums_off + g * 2, s)


# ---------------------------------------------------------------------------
# NumPy-accelerated Q4_K x Q8_K row dot
# ---------------------------------------------------------------------------


def _int8_array(buf: bytes | bytearray) -> np.ndarray:
    """Convert bytes to signed int8 numpy array."""
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int8)


def q4k_q8k_row_dot_scalar(
    row: bytes | bytearray,
    blocks_per_row: int,
    q8k: bytes | bytearray,
) -> float:
    """Dot one Q4_K row against a Q8_K vector.

    Uses NumPy vectorization for the inner loops while keeping the exact same
    math as the Rust scalar reference.
    """
    if len(row) < blocks_per_row * BLOCK_Q4_K_SIZE:
        raise ValueError("row too small")
    if len(q8k) < blocks_per_row * BLOCK_Q8_K_BYTES:
        raise ValueError("q8k too small")

    acc = 0.0
    for block_idx in range(blocks_per_row):
        w = row[block_idx * BLOCK_Q4_K_SIZE : (block_idx + 1) * BLOCK_Q4_K_SIZE]
        q8b = q8k[block_idx * BLOCK_Q8_K_BYTES : (block_idx + 1) * BLOCK_Q8_K_BYTES]
        d_w = _f16_le_to_f32(w[0:2])
        dmin_w = _f16_le_to_f32(w[2:4])
        d_q8 = struct.unpack_from("<f", q8b, 0)[0]
        scales = w[4:16]
        qs = np.frombuffer(w[16 : 16 + QK_K // 2], dtype=np.uint8)
        q8 = _int8_array(q8b[4 : 4 + QK_K])
        bsums = np.frombuffer(q8b[4 + QK_K :], dtype=np.int16)

        # Unpack nibbles: low and high
        low = qs & 0x0F
        high = qs >> 4

        # Process 8 groups of 32 elements each (4 outer groups * 2 inner)
        pos = 0
        min_acc = 0
        for gp in range(4):
            g1 = gp * 2
            g2 = g1 + 1
            s1, ms1 = get_scale_min_k4(g1, scales)
            s2, ms2 = get_scale_min_k4(g2, scales)

            base = gp * 32
            # Vectorized sum1: low nibbles * q8[g1*32:(g1+1)*32]
            sum1 = int(
                np.dot(
                    low[base : base + 32].astype(np.int32),
                    q8[g1 * 32 : (g1 + 1) * 32].astype(np.int32),
                )
            )
            # Vectorized sum2: high nibbles * q8[g2*32:(g2+1)*32]
            sum2 = int(
                np.dot(
                    high[base : base + 32].astype(np.int32),
                    q8[g2 * 32 : (g2 + 1) * 32].astype(np.int32),
                )
            )

            pos += s1 * sum1 + s2 * sum2
            bs1 = int(bsums[g1 * 2]) + int(bsums[g1 * 2 + 1])
            bs2 = int(bsums[g2 * 2]) + int(bsums[g2 * 2 + 1])
            min_acc += ms1 * bs1
            min_acc += ms2 * bs2

        acc += d_w * d_q8 * pos - dmin_w * d_q8 * min_acc
    return acc


# ---------------------------------------------------------------------------
# Tile runners (multi-row variants) — enabled by default, scalar fallback
# ---------------------------------------------------------------------------


def q4k_q8k_row_dot_x1_scalar(
    row: bytes | bytearray,
    blocks_per_row: int,
    q8k: bytes | bytearray,
) -> float:
    return q4k_q8k_row_dot_scalar(row, blocks_per_row, q8k)


def q4k_q8k_row_dot_x4_scalar(
    rows: bytes | bytearray,
    row_bytes: int,
    blocks_per_row: int,
    q8k: bytes | bytearray,
    out: list[float],
) -> None:
    if len(out) < 4:
        raise ValueError("out too small")
    for r in range(4):
        row = rows[r * row_bytes : (r + 1) * row_bytes]
        out[r] = q4k_q8k_row_dot_scalar(row, blocks_per_row, q8k)


def q4k_q8k_row_dot_x8_scalar(
    rows: bytes | bytearray,
    row_bytes: int,
    blocks_per_row: int,
    q8k: bytes | bytearray,
    out: list[float],
) -> None:
    if len(out) < 8:
        raise ValueError("out too small")
    for r in range(8):
        row = rows[r * row_bytes : (r + 1) * row_bytes]
        out[r] = q4k_q8k_row_dot_scalar(row, blocks_per_row, q8k)


def q4k_q8k_row_dot_x16_scalar(
    rows: bytes | bytearray,
    row_bytes: int,
    blocks_per_row: int,
    q8k: bytes | bytearray,
    out: list[float],
) -> None:
    if len(out) < 16:
        raise ValueError("out too small")
    for r in range(16):
        row = rows[r * row_bytes : (r + 1) * row_bytes]
        out[r] = q4k_q8k_row_dot_scalar(row, blocks_per_row, q8k)


# ---------------------------------------------------------------------------
# Range GEMV — main entry point with tile runner enabled by default
# ---------------------------------------------------------------------------


def gemv_q4k_range(
    rows: bytes | bytearray,
    blocks_per_row: int,
    q8k: bytes | bytearray,
    out: list[float],
) -> None:
    row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE
    if len(rows) < len(out) * row_bytes:
        raise ValueError("rows buffer too small")
    if len(q8k) < blocks_per_row * BLOCK_Q8_K_BYTES:
        raise ValueError("q8k buffer too small")

    tile = max_tile()
    n = len(out)
    r = 0

    # x16 tile (widest, enabled by default)
    while tile >= 16 and r + 16 <= n:
        base = rows[r * row_bytes :]
        hex_ = [0.0] * 16
        q4k_q8k_row_dot_x16_scalar(base, row_bytes, blocks_per_row, q8k, hex_)
        out[r : r + 16] = hex_
        r += 16

    # x8 tile
    while tile >= 8 and r + 8 <= n:
        base = rows[r * row_bytes :]
        octet = [0.0] * 8
        q4k_q8k_row_dot_x8_scalar(base, row_bytes, blocks_per_row, q8k, octet)
        out[r : r + 8] = octet
        r += 8

    # x4 tile
    while tile >= 4 and r + 4 <= n:
        base = rows[r * row_bytes :]
        quad = [0.0] * 4
        q4k_q8k_row_dot_x4_scalar(base, row_bytes, blocks_per_row, q8k, quad)
        out[r : r + 4] = quad
        r += 4

    # x1 tail
    while r < n:
        row = rows[r * row_bytes : (r + 1) * row_bytes]
        out[r] = q4k_q8k_row_dot_scalar(row, blocks_per_row, q8k)
        r += 1


def int8(b: int) -> int:
    """Convert unsigned byte to signed int8."""
    return b if b < 128 else b - 256
