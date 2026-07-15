#!/usr/bin/env python3
"""Emits tests/golden.h: known quantized blocks + their expected dequantized
values. The reference below is written from the ggml block layouts
(llama.cpp ggml-quants.c) and the AL5_XS spec, NOT from src/quant.c -- that is
the whole point: it must be able to disagree with the kernels.

  python3 tools/gen_golden.py > tests/golden.h

The IQ (grid-codebook) generators re-derive the ggml dequant here in Python,
INDEPENDENTLY of src/quant.c. They share only the fixed codebook grids, which
are parsed out of src/quant_iq_grids.h (itself a verbatim copy of ggml's
tables) -- a hand-tuned codebook has nothing to "independently derive", so one
verified copy is the honest single source; what these golden values actually
cross-check is the packing + scale LOGIC, coded separately here and in quant.c.
"""
import os
import re
import struct

QK, QK_K = 32, 256

# ---- IQ codebook grids (parsed from src/quant_iq_grids.h) -------------------
_GRID_H = os.path.join(os.path.dirname(__file__), '..', 'src', 'quant_iq_grids.h')


def _load_grids(path):
    text = open(path).read()
    out = {}
    for m in re.finditer(r'oc_(\w+)\[(\d+)\]\s*=\s*\{([^}]*)\}', text):
        name, n, body = m.group(1), int(m.group(2)), m.group(3)
        vals = [int(x, 0) for x in re.findall(r'0x[0-9a-fA-F]+|\d+', body)]
        assert len(vals) == n, (name, len(vals), n)
        out[name] = vals
    return out


GRIDS = _load_grids(_GRID_H)
KMASK = GRIDS['kmask_iq2xs']
KSIGNS = GRIDS['ksigns_iq2xs']
IQ1S_DELTA = 0.125


def grid_bytes(val, n, signed=False):
    """The n reconstruction bytes of a codebook entry, little-endian; the IQ1
    grids are read as signed int8 (ggml casts iq1s_grid to const int8_t*)."""
    r = []
    for j in range(n):
        b = (val >> (8 * j)) & 0xFF
        r.append(b - 256 if (signed and b >= 128) else b)
    return r


def sgn(signs, j):
    return -1.0 if (signs & KMASK[j]) else 1.0


def lcg(seed):
    s = seed
    while True:
        s = (s * 1103515245 + 12345) & 0xFFFFFFFF
        yield (s >> 16) & 0xFF


def f16(x):  # float -> raw f16 bits
    return struct.unpack('<H', struct.pack('<e', x))[0]


def unf16(b):  # raw f16 bits -> float
    return struct.unpack('<e', struct.pack('<H', b))[0]


def put16(dst, off, bits):
    dst[off] = bits & 0xFF
    dst[off + 1] = bits >> 8


def get_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4))


def q4_0():
    r = lcg(0xC0FFEE)
    blk = bytearray(18)
    d = 0.0140380859375  # exactly representable in f16
    put16(blk, 0, f16(d))
    for i in range(16):
        blk[2 + i] = next(r)
    out = [0.0] * 32
    for i in range(16):
        out[i] = ((blk[2 + i] & 0xF) - 8) * d
        out[i + 16] = ((blk[2 + i] >> 4) - 8) * d
    return blk, out


def q8_0():
    r = lcg(0xBEEF)
    blk = bytearray(34)
    d = 0.0234375
    put16(blk, 0, f16(d))
    for i in range(32):
        blk[2 + i] = next(r)
    out = [struct.unpack('b', bytes([blk[2 + i]]))[0] * d for i in range(32)]
    return blk, out


def q4_k():
    r = lcg(0x1234)
    blk = bytearray(144)
    d, mn = 0.0234375, 0.01171875
    put16(blk, 0, f16(d))
    put16(blk, 2, f16(mn))
    for i in range(12):
        blk[4 + i] = next(r)
    for i in range(128):
        blk[16 + i] = next(r)
    sc, qs = blk[4:16], blk[16:144]
    out, is_ = [], 0
    for gp in range(4):
        q = qs[gp * 32:gp * 32 + 32]
        s1, m1 = get_scale_min_k4(is_, sc)
        s2, m2 = get_scale_min_k4(is_ + 1, sc)
        out += [d * s1 * (q[l] & 0xF) - mn * m1 for l in range(32)]
        out += [d * s2 * (q[l] >> 4) - mn * m2 for l in range(32)]
        is_ += 2
    return blk, out


def q5_k():
    r = lcg(0x5A5A)
    blk = bytearray(176)
    d, mn = 0.0185546875, 0.005859375
    put16(blk, 0, f16(d))
    put16(blk, 2, f16(mn))
    for i in range(12):
        blk[4 + i] = next(r)
    for i in range(32):
        blk[16 + i] = next(r)   # qh
    for i in range(128):
        blk[48 + i] = next(r)   # ql
    sc, qh, ql = blk[4:16], blk[16:48], blk[48:176]
    out, is_, u1, u2 = [], 0, 1, 2
    for gp in range(4):
        q = ql[gp * 32:gp * 32 + 32]
        s1, m1 = get_scale_min_k4(is_, sc)
        s2, m2 = get_scale_min_k4(is_ + 1, sc)
        out += [d * s1 * ((q[l] & 0xF) + (16 if qh[l] & u1 else 0)) - mn * m1
                for l in range(32)]
        out += [d * s2 * ((q[l] >> 4) + (16 if qh[l] & u2 else 0)) - mn * m2
                for l in range(32)]
        is_ += 2
        u1 = (u1 << 2) & 0xFF
        u2 = (u2 << 2) & 0xFF
    return blk, out


def q6_k():
    r = lcg(0x77)
    blk = bytearray(210)
    d = 0.0068359375
    for i in range(128):
        blk[i] = next(r)        # ql
    for i in range(64):
        blk[128 + i] = next(r)  # qh
    for i in range(16):
        blk[192 + i] = next(r)  # scales (int8)
    put16(blk, 208, f16(d))
    sc = [struct.unpack('b', bytes([blk[192 + i]]))[0] for i in range(16)]
    out = [0.0] * 256
    for g in range(2):
        ql, qh = blk[g * 64:g * 64 + 64], blk[128 + g * 32:128 + g * 32 + 32]
        s, base = sc[g * 8:g * 8 + 8], g * 128
        for l in range(32):
            i = l // 16
            q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32
            q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
            out[base + l] = d * s[i] * q1
            out[base + l + 32] = d * s[i + 2] * q2
            out[base + l + 64] = d * s[i + 4] * q3
            out[base + l + 96] = d * s[i + 6] * q4
    return blk, out


KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10,
                 1, 13, 25, 38, 53, 69, 89, 113]


def q4_1():
    """d, m, 16 nibble bytes; ggml split order out[j], out[j+16]."""
    r = lcg(0x0401)
    blk = bytearray(20)
    d, m = 0.015625, -0.125
    put16(blk, 0, f16(d))
    put16(blk, 2, f16(m))
    for i in range(16):
        blk[4 + i] = next(r)
    out = [0.0] * 32
    for j in range(16):
        out[j] = (blk[4 + j] & 0xF) * d + m
        out[j + 16] = (blk[4 + j] >> 4) * d + m
    return blk, out


def q5_0():
    """d, 4-byte qh, 16 nibble bytes; 5th bit from qh bit i, split order."""
    r = lcg(0x0500)
    blk = bytearray(22)
    d = 0.0234375
    put16(blk, 0, f16(d))
    for i in range(4):
        blk[2 + i] = next(r)   # qh
    for i in range(16):
        blk[6 + i] = next(r)   # qs
    qh = int.from_bytes(blk[2:6], 'little')
    out = [0.0] * 32
    for j in range(16):
        xh0 = ((qh >> j) & 1) << 4
        xh1 = ((qh >> (j + 16)) & 1) << 4
        out[j] = (((blk[6 + j] & 0xF) | xh0) - 16) * d
        out[j + 16] = (((blk[6 + j] >> 4) | xh1) - 16) * d
    return blk, out


def q5_1():
    """d, m, 4-byte qh, 16 nibble bytes; split order, min offset (no -16)."""
    r = lcg(0x0501)
    blk = bytearray(24)
    d, m = 0.01953125, -0.0625
    put16(blk, 0, f16(d))
    put16(blk, 2, f16(m))
    for i in range(4):
        blk[4 + i] = next(r)   # qh
    for i in range(16):
        blk[8 + i] = next(r)   # qs
    qh = int.from_bytes(blk[4:8], 'little')
    out = [0.0] * 32
    for j in range(16):
        xh0 = ((qh >> j) & 1) << 4
        xh1 = ((qh >> (j + 16)) & 1) << 4
        out[j] = ((blk[8 + j] & 0xF) | xh0) * d + m
        out[j + 16] = ((blk[8 + j] >> 4) | xh1) * d + m
    return blk, out


def q2_k():
    """16 scale bytes, 64 qs, f16 d, f16 dmin (ggml dequantize_row_q2_K)."""
    r = lcg(0x2C0)
    blk = bytearray(84)
    d, dmin = 0.015625, 0.00390625
    for i in range(16):
        blk[i] = next(r)        # scales
    for i in range(64):
        blk[16 + i] = next(r)   # qs
    put16(blk, 80, f16(d))
    put16(blk, 82, f16(dmin))
    scales, qs = blk[0:16], blk[16:80]
    out = [0.0] * 256
    y, is_ = 0, 0
    for n in range(0, 256, 128):
        qbase = (n // 128) * 32
        shift = 0
        for _ in range(4):
            sc = scales[is_]
            is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            for l in range(16):
                out[y + l] = dl * ((qs[qbase + l] >> shift) & 3) - ml
            sc = scales[is_]
            is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            for l in range(16):
                out[y + 16 + l] = dl * ((qs[qbase + 16 + l] >> shift) & 3) - ml
            y += 32
            shift += 2
    return blk, out


def q3_k():
    """32 hmask, 64 qs, 12 packed scales, f16 d (ggml dequantize_row_q3_K)."""
    r = lcg(0x3C0)
    blk = bytearray(110)
    d = 0.0068359375
    for i in range(32):
        blk[i] = next(r)        # hmask
    for i in range(64):
        blk[32 + i] = next(r)   # qs
    for i in range(12):
        blk[96 + i] = next(r)   # packed 6-bit scales
    put16(blk, 108, f16(d))
    hm, qs = blk[0:32], blk[32:96]
    aux = [int.from_bytes(blk[96 + 4 * k:100 + 4 * k], 'little') for k in range(3)]
    aux.append(0)
    tmp = aux[2]
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4)
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4)
    aux[0] = (aux[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4)
    aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4)
    sb = bytearray()
    for k in range(4):
        sb += (aux[k] & 0xFFFFFFFF).to_bytes(4, 'little')

    def s8(b):
        return b - 256 if b >= 128 else b
    out = [0.0] * 256
    y, is_, m = 0, 0, 1
    for n in range(0, 256, 128):
        qbase = (n // 128) * 32
        shift = 0
        for _ in range(4):
            dl = d * (s8(sb[is_]) - 32)
            is_ += 1
            for l in range(16):
                qv = (qs[qbase + l] >> shift) & 3
                hbit = 0 if (hm[l] & m) else 4
                out[y + l] = dl * (qv - hbit)
            dl = d * (s8(sb[is_]) - 32)
            is_ += 1
            for l in range(16):
                qv = (qs[qbase + 16 + l] >> shift) & 3
                hbit = 0 if (hm[l + 16] & m) else 4
                out[y + 16 + l] = dl * (qv - hbit)
            y += 32
            shift += 2
            m <<= 1
    return blk, out


def iq4_xs():
    """f16 d, u16 scales_h, 4 scales_l, 128 qs (ggml dequantize_row_iq4_xs)."""
    r = lcg(0x4C5)
    blk = bytearray(136)
    d = 0.0234375
    put16(blk, 0, f16(d))
    for i in range(2):
        blk[2 + i] = next(r)     # scales_h (u16)
    for i in range(4):
        blk[4 + i] = next(r)     # scales_l
    for i in range(128):
        blk[8 + i] = next(r)     # qs
    scales_h = int.from_bytes(blk[2:4], 'little')
    scales_l, qs = blk[4:8], blk[8:136]
    out = [0.0] * 256
    for ib in range(8):
        ls_l = (scales_l[ib // 2] >> (4 * (ib % 2))) & 0xf
        ls_h = ((scales_h >> (2 * ib)) & 3) << 4
        dl = d * ((ls_l | ls_h) - 32)
        for j in range(16):
            b = qs[ib * 16 + j]
            out[ib * 32 + j] = dl * KVALUES_IQ4NL[b & 0xf]
            out[ib * 32 + j + 16] = dl * KVALUES_IQ4NL[b >> 4]
    return blk, out


def bf16():
    """High 16 bits of f32; hand-picked finite patterns."""
    bits = [0x3f80, 0xc000, 0x4049, 0x3e80, 0xbf00, 0x4120, 0x0000, 0x3dcc]
    blk = bytearray()
    for b in bits:
        blk += b.to_bytes(2, 'little')
    out = [struct.unpack('<f', struct.pack('<I', b << 16))[0] for b in bits]
    return blk, out


def al5_xs():
    """3-bit codes, LSB-first in a 96-bit LE stream; w = (code - 4) * scale."""
    blk = bytearray(14)
    scale = 0.0625
    put16(blk, 0, f16(scale))
    codes = [(i * 5 + 3) & 7 for i in range(32)]
    bits = 0
    for i, c in enumerate(codes):
        bits |= c << (3 * i)
    for i in range(12):
        blk[2 + i] = (bits >> (8 * i)) & 0xFF
    return blk, [(c - 4) * scale for c in codes]


def iq2_xxs():
    """f16 d + 32 u16 qs (ggml dequantize_row_iq2_xxs)."""
    r = lcg(0x2222)
    blk = bytearray(66)
    d = 0.0234375
    put16(blk, 0, f16(d))
    for i in range(64):
        blk[2 + i] = next(r)
    qs = blk[2:66]
    out = []
    for ib32 in range(8):
        base = 8 * ib32
        a1 = int.from_bytes(qs[base + 4:base + 8], 'little')
        db = d * (0.5 + (a1 >> 28)) * 0.25
        for l in range(4):
            grid = grid_bytes(GRIDS['iq2xxs_grid'][qs[base + l]], 8)
            signs = KSIGNS[(a1 >> (7 * l)) & 127]
            out += [db * grid[j] * sgn(signs, j) for j in range(8)]
    return blk, out


def iq2_xs():
    """f16 d + 32 u16 qs + 8 scales (ggml dequantize_row_iq2_xs)."""
    r = lcg(0x1717)
    blk = bytearray(74)
    d = 0.0234375
    put16(blk, 0, f16(d))
    for i in range(72):
        blk[2 + i] = next(r)
    qs, scales = blk[2:66], blk[66:74]
    out = []
    for ib32 in range(8):
        db = [d * (0.5 + (scales[ib32] & 0xf)) * 0.25,
              d * (0.5 + (scales[ib32] >> 4)) * 0.25]
        for l in range(4):
            q = int.from_bytes(qs[2 * (4 * ib32 + l):2 * (4 * ib32 + l) + 2], 'little')
            grid = grid_bytes(GRIDS['iq2xs_grid'][q & 511], 8)
            signs = KSIGNS[q >> 9]
            dl = db[l // 2]
            out += [dl * grid[j] * sgn(signs, j) for j in range(8)]
    return blk, out


def iq2_s():
    """f16 d + 64 qs (grid-lo + signs) + 8 qh + 8 scales (dequantize_row_iq2_s)."""
    r = lcg(0x2255)
    blk = bytearray(82)
    d = 0.0234375
    put16(blk, 0, f16(d))
    for i in range(80):
        blk[2 + i] = next(r)
    qs, qh, scales = blk[2:66], blk[66:74], blk[74:82]
    signs = qs[32:64]
    out = []
    for ib32 in range(8):
        db = [d * (0.5 + (scales[ib32] & 0xf)) * 0.25,
              d * (0.5 + (scales[ib32] >> 4)) * 0.25]
        for l in range(4):
            dl = db[l // 2]
            idx = qs[4 * ib32 + l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)
            grid = grid_bytes(GRIDS['iq2s_grid'][idx], 8)
            s = signs[4 * ib32 + l]
            out += [dl * grid[j] * sgn(s, j) for j in range(8)]
    return blk, out


def iq3_xxs():
    """f16 d + 64 qs + 32 scales/signs (ggml dequantize_row_iq3_xxs)."""
    r = lcg(0x1818)
    blk = bytearray(98)
    d = 0.0234375
    put16(blk, 0, f16(d))
    for i in range(96):
        blk[2 + i] = next(r)
    qs, sas = blk[2:66], blk[66:98]
    out = []
    for ib32 in range(8):
        a = int.from_bytes(sas[4 * ib32:4 * ib32 + 4], 'little')
        db = d * (0.5 + (a >> 28)) * 0.5
        for l in range(4):
            signs = KSIGNS[(a >> (7 * l)) & 127]
            g1 = grid_bytes(GRIDS['iq3xxs_grid'][qs[8 * ib32 + 2 * l + 0]], 4)
            g2 = grid_bytes(GRIDS['iq3xxs_grid'][qs[8 * ib32 + 2 * l + 1]], 4)
            row = [0.0] * 8
            for j in range(4):
                row[j + 0] = db * g1[j] * sgn(signs, j + 0)
                row[j + 4] = db * g2[j] * sgn(signs, j + 4)
            out += row
    return blk, out


def iq3_s():
    """f16 d + 64 qs + 8 qh + 32 signs + 4 scales (ggml dequantize_row_iq3_s)."""
    r = lcg(0x2121)
    blk = bytearray(110)
    d = 0.0234375
    put16(blk, 0, f16(d))
    for i in range(108):
        blk[2 + i] = next(r)
    qs, qh, signs, scales = blk[2:66], blk[66:74], blk[74:106], blk[106:110]
    out = []
    qpos, spos = 0, 0
    for ib32 in range(0, 8, 2):
        db1 = d * (1 + 2 * (scales[ib32 // 2] & 0xf))
        db2 = d * (1 + 2 * (scales[ib32 // 2] >> 4))
        h0, h1 = qh[ib32], qh[ib32 + 1]
        for db, h in ((db1, h0), (db2, h1)):
            for l in range(4):
                i1 = qs[qpos + 2 * l + 0] | ((h << (8 - 2 * l)) & 256)
                i2 = qs[qpos + 2 * l + 1] | ((h << (7 - 2 * l)) & 256)
                g1 = grid_bytes(GRIDS['iq3s_grid'][i1], 4)
                g2 = grid_bytes(GRIDS['iq3s_grid'][i2], 4)
                s = signs[spos + l]
                row = [0.0] * 8
                for j in range(4):
                    row[j + 0] = db * g1[j] * sgn(s, j + 0)
                    row[j + 4] = db * g2[j] * sgn(s, j + 4)
                out += row
            qpos += 8
            spos += 4
    return blk, out


def iq1_s():
    """f16 d + 32 qs + 8 u16 qh (ggml dequantize_row_iq1_s)."""
    r = lcg(0x1919)
    blk = bytearray(50)
    d = 0.0234375
    put16(blk, 0, f16(d))
    for i in range(48):
        blk[2 + i] = next(r)
    qs, qh = blk[2:34], blk[34:50]
    out = []
    for ib in range(8):
        h = int.from_bytes(qh[2 * ib:2 * ib + 2], 'little')
        dl = d * (2 * ((h >> 12) & 7) + 1)
        delta = -IQ1S_DELTA if (h & 0x8000) else IQ1S_DELTA
        for l in range(4):
            idx = qs[4 * ib + l] | (((h >> (3 * l)) & 7) << 8)
            grid = grid_bytes(GRIDS['iq1s_grid'][idx], 8, signed=True)
            out += [dl * (grid[j] + delta) for j in range(8)]
    return blk, out


def iq1_m():
    """32 qs + 16 qh + 8 scales; no d field -- f16 scale packed in the top
    nibbles of the 4 u16 scales (ggml dequantize_row_iq1_m). We fix those 4
    nibbles so the assembled scale is a sane finite f16; the low 12 bits (the
    per-32 sub-scales) stay random."""
    r = lcg(0x1D1D)
    blk = bytearray(56)
    for i in range(56):
        blk[i] = next(r)
    H = f16(0.0625)  # assembled scale; nibbles go into the scale u16 tops
    for k in range(4):
        blk[48 + 2 * k + 1] = (blk[48 + 2 * k + 1] & 0x0f) | (((H >> (4 * k)) & 0xf) << 4)
    qs, qh = blk[0:32], blk[32:48]
    sc = [int.from_bytes(blk[48 + 2 * k:48 + 2 * k + 2], 'little') for k in range(4)]
    s = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
    d = unf16(s)
    out = []
    for ib in range(8):
        dl1 = d * (2 * ((sc[ib // 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1)
        dl2 = d * (2 * ((sc[ib // 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1)
        idx = [qs[4 * ib + 0] | ((qh[2 * ib + 0] << 8) & 0x700),
               qs[4 * ib + 1] | ((qh[2 * ib + 0] << 4) & 0x700),
               qs[4 * ib + 2] | ((qh[2 * ib + 1] << 8) & 0x700),
               qs[4 * ib + 3] | ((qh[2 * ib + 1] << 4) & 0x700)]
        delta = [-IQ1S_DELTA if qh[2 * ib + 0] & 0x08 else IQ1S_DELTA,
                 -IQ1S_DELTA if qh[2 * ib + 0] & 0x80 else IQ1S_DELTA,
                 -IQ1S_DELTA if qh[2 * ib + 1] & 0x08 else IQ1S_DELTA,
                 -IQ1S_DELTA if qh[2 * ib + 1] & 0x80 else IQ1S_DELTA]
        for l in range(4):
            grid = grid_bytes(GRIDS['iq1s_grid'][idx[l]], 8, signed=True)
            dl = dl1 if l < 2 else dl2
            out += [dl * (grid[j] + delta[l]) for j in range(8)]
    return blk, out


def cf(v):  # C float literal (%g alone emits "3", and "3f" is not a float)
    s = "%.9g" % v
    if "." not in s and "e" not in s:
        s += ".0"
    return s + "f"


def emit(name, blk, exp):
    print("static const uint8_t GOLD_%s[%d] = {" % (name, len(blk)))
    for i in range(0, len(blk), 12):
        print("    " + " ".join("0x%02x," % b for b in blk[i:i + 12]))
    print("};")
    print("static const float GOLD_%s_EXP[%d] = {" % (name, len(exp)))
    for i in range(0, len(exp), 4):
        print("    " + " ".join(cf(v) + "," for v in exp[i:i + 4]))
    print("};")
    print()


print("""/* GENERATED by tools/gen_golden.py -- do not edit.
 * Known blocks + expected dequantized values, derived from the ggml block
 * layouts and the AL5_XS spec, independently of src/quant.c. */
#ifndef OC_GOLDEN_H
#define OC_GOLDEN_H

#include <stdint.h>
""")
for n, f in (("Q4_0", q4_0), ("Q4_1", q4_1), ("Q5_0", q5_0), ("Q5_1", q5_1),
             ("Q8_0", q8_0), ("Q2_K", q2_k), ("Q3_K", q3_k), ("Q4_K", q4_k),
             ("Q5_K", q5_k), ("Q6_K", q6_k), ("IQ4_XS", iq4_xs),
             ("IQ2_XXS", iq2_xxs), ("IQ2_XS", iq2_xs), ("IQ2_S", iq2_s),
             ("IQ3_XXS", iq3_xxs), ("IQ3_S", iq3_s), ("IQ1_S", iq1_s),
             ("IQ1_M", iq1_m),
             ("BF16", bf16), ("AL5_XS", al5_xs)):
    emit(n, *f())
print("#endif")
