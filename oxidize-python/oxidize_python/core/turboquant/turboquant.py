"""TurboQuant blockwise quantization mirroring oxidize-golang/core/turboquant."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Type(IntEnum):
    INT4 = 0
    INT8 = 1


@dataclass
class Block:
    scale: float = 0.0
    values: list[int] = field(default_factory=list)


@dataclass
class Data:
    qtype: Type = Type.INT4
    cols: int = 0
    rows: int = 0
    blocks: list[Block] = field(default_factory=list)

    def quantize_f32(self, src: list[float], rows: int, cols: int, qtype: Type) -> None:
        self.qtype = qtype
        self.rows = rows
        self.cols = cols
        if rows * cols == 0:
            self.blocks = []
            return
        block = 32
        stride = min(block, cols)
        num_blocks = (cols + stride - 1) // stride
        self.blocks = []
        for r in range(rows):
            for b in range(num_blocks):
                start = b * stride
                end = min(start + stride, cols)
                values = src[r * cols + start : r * cols + end]
                self.blocks.append(_quantize_block(values, qtype))

    def dequantize_f32(self, out: list[float]) -> None:
        if len(out) < self.rows * self.cols:
            return
        block = 32
        stride = min(block, self.cols)
        num_blocks = (self.cols + stride - 1) // stride
        idx = 0
        for r in range(self.rows):
            for b in range(num_blocks):
                if idx >= len(self.blocks):
                    return
                start = b * stride
                end = min(start + stride, self.cols)
                blk = self.blocks[idx]
                idx += 1
                _dequantize_block(blk, out[r * self.cols + start : r * self.cols + end], self.qtype)

    def gemv(self, vector: list[float], output: list[float]) -> None:
        if len(vector) < self.cols or len(output) < self.rows:
            return
        row = [0.0] * (self.rows * self.cols)
        self.dequantize_f32(row)
        for r in range(self.rows):
            s = 0.0
            for c in range(self.cols):
                s += row[r * self.cols + c] * vector[c]
            output[r] = s


def _quantize_block(values: list[float], qtype: Type) -> Block:
    if not values:
        return Block()
    max_v = max(abs(v) for v in values)
    if qtype == Type.INT4:
        levels = 7
    else:
        levels = 127
    scale = max_v / levels if max_v else 1.0
    if scale == 0:
        scale = 1.0
    inv = 1.0 / scale
    out: list[int] = []
    for v in values:
        if qtype == Type.INT4:
            q = _clamp_int(v * inv, -8, 7)
            out.append(_nibble_pack(q))
        else:
            q = _clamp_int(v * inv, -128, 127)
            out.append(q & 0xFF)
    return Block(scale=scale, values=out)


def _dequantize_block(blk: Block, dst: list[float], qtype: Type) -> None:
    if not dst:
        return
    for i, v in enumerate(blk.values):
        if qtype == Type.INT4:
            q = _nibble_unpack(v)
            dst[i] = float(q) * blk.scale
        else:
            q = v if v < 128 else v - 256
            dst[i] = float(q) * blk.scale
    for i in range(len(blk.values), len(dst)):
        dst[i] = 0.0


def _clamp_int(v: float, lo: int, hi: int) -> int:
    iv = int(v)
    return max(lo, min(hi, iv))


def _nibble_pack(v: int) -> int:
    return v & 0x0F


def _nibble_unpack(v: int) -> int:
    if v >= 8:
        return v - 16
    return v
