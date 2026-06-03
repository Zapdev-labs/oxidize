"""DType and Tensor mirroring oxidize-golang/core/tensor/dtype.go."""

from __future__ import annotations

import struct
import threading
from dataclasses import dataclass
from enum import IntEnum, auto


class DType(IntEnum):
    F32 = 0
    F16 = 1
    I8 = 2
    I16 = 3
    I32 = 4
    I64 = 5

    def size_in_bytes(self) -> int:
        match self:
            case DType.F32 | DType.I32:
                return 4
            case DType.F16 | DType.I16:
                return 2
            case DType.I8:
                return 1
            case DType.I64:
                return 8
        return 0

    def __str__(self) -> str:
        return self.name.lower()


class ActivationFn(IntEnum):
    NONE = auto()
    SILU = auto()
    GELU = auto()
    RELU = auto()


@dataclass
class Tensor:
    _mu: threading.RLock
    _data: list[float]
    _bytes: bytes | bytearray
    shape: list[int]
    strides: list[int]
    dtype: DType
    owned: bool

    @classmethod
    def new(cls, data: list[float], shape: list[int]) -> Tensor | None:
        total = 1
        for s in shape:
            if s <= 0:
                return None
            total *= s
        if total != len(data):
            return None
        t = cls(
            _mu=threading.RLock(),
            _data=list(data),
            _bytes=b"",
            shape=list(shape),
            strides=_default_strides(shape),
            dtype=DType.F32,
            owned=True,
        )
        return t

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, shape: list[int], dt: DType) -> Tensor:
        total = 1
        for s in shape:
            if s <= 0:
                raise ValueError(f"invalid shape dimension: {s}")
            total *= s
        if total * dt.size_in_bytes() != len(data):
            raise ValueError(f"shape/dtype mismatch: shape={shape} dtype={dt} bytes={len(data)}")
        f32 = list(_bytes_as_f32(data)) if dt == DType.F32 else []
        return cls(
            _mu=threading.RLock(),
            _data=f32,
            _bytes=data,
            shape=list(shape),
            strides=_default_strides(shape),
            dtype=dt,
            owned=dt == DType.F32,
        )

    def data(self) -> list[float]:
        with self._mu:
            if self.dtype == DType.F32:
                return list(self._data)
            n = self.num_elements()
            return [self._f32_at(i) for i in range(n)]

    def bytes(self) -> bytes:
        with self._mu:
            if self.dtype == DType.F32:
                buf = bytearray(len(self._data) * 4)
                for i, v in enumerate(self._data):
                    struct.pack_into("<f", buf, i * 4, v)
                return bytes(buf)
            return bytes(self._bytes)

    def num_elements(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def at(self, i: int) -> float:
        with self._mu:
            return self._f32_at(i)

    def set(self, i: int, v: float) -> None:
        with self._mu:
            match self.dtype:
                case DType.F32:
                    self._data[i] = v
                case DType.F16:
                    bits = f32_to_f16_bits(v)
                    self._bytes[i * 2] = bits & 0xFF
                    self._bytes[i * 2 + 1] = (bits >> 8) & 0xFF
                case DType.I8:
                    self._bytes[i] = int(max(-128, min(127, int(v)))) & 0xFF
                case DType.I16:
                    struct.pack_into("<h", self._bytes, i * 2, int(max(-32768, min(32767, int(v)))))
                case DType.I32:
                    struct.pack_into("<i", self._bytes, i * 4, int(v))
                case DType.I64:
                    struct.pack_into("<q", self._bytes, i * 8, int(v))

    def _f32_at(self, i: int) -> float:
        match self.dtype:
            case DType.F32:
                return self._data[i]
            case DType.F16:
                bits = self._bytes[i * 2] | (self._bytes[i * 2 + 1] << 8)
                return f16_bits_to_f32(bits)
            case DType.I8:
                b = self._bytes[i]
                return float(b - 256 if b > 127 else b)
            case DType.I16:
                return float(struct.unpack_from("<h", self._bytes, i * 2)[0])
            case DType.I32:
                return float(struct.unpack_from("<i", self._bytes, i * 4)[0])
            case DType.I64:
                return float(struct.unpack_from("<q", self._bytes, i * 8)[0])
        return 0.0


def _default_strides(shape: list[int]) -> list[int]:
    strides = [0] * len(shape)
    if not shape:
        return strides
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return strides


def _bytes_as_f32(b: bytes | bytearray) -> list[float]:
    return [struct.unpack_from("<f", b, i * 4)[0] for i in range(len(b) // 4)]


def f16_le_to_f32(b: tuple[int, int] | bytes) -> float:
    if isinstance(b, bytes):
        bits = b[0] | (b[1] << 8)
    else:
        bits = b[0] | (b[1] << 8)
    return f16_bits_to_f32(bits)


def f16_bits_to_f32(bits: int) -> float:
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0:
        if mant == 0:
            out = sign << 31
        else:
            while mant & 0x400 == 0:
                mant <<= 1
                exp -= 1
            exp += 1
            mant &= 0x3FF
            out = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    elif exp == 0x1F:
        out = (sign << 31) | (0xFF << 23) | (mant << 13)
    else:
        out = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    return struct.unpack("<f", struct.pack("<I", out))[0]


def f32_to_f16_bits(f: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", f))[0]
    sign = (bits >> 31) & 1
    exp = ((bits >> 23) & 0xFF) - 127
    mant = bits & 0x7FFFFF
    if exp == 128:
        return (sign << 15) | (0x7C00 if mant == 0 else 0x7C00 | (mant >> 13))
    if exp > 15:
        return (sign << 15) | 0x7C00
    if exp < -24:
        return sign << 15
    if exp < -13:
        mant |= 0x800000
        shift = -10 - exp
        mant = (mant + (1 << (shift - 1))) >> shift
        return (sign << 15) | mant
    return (sign << 15) | ((exp + 15) << 10) | (mant >> 13)


def extract_bits(bitstream: bytes | bytearray, index: int, bits: int) -> int:
    if bits == 0 or bits > 32:
        return 0
    bit_pos = index * bits
    byte_pos = bit_pos // 8
    bit_offset = bit_pos % 8
    value = 0
    remaining = bits
    shift = 0
    while remaining > 0:
        available = 8 - bit_offset
        take = min(remaining, available)
        mask = (1 << take) - 1
        chunk = (bitstream[byte_pos] >> bit_offset) & mask
        value |= chunk << shift
        remaining -= take
        shift += take
        bit_offset = 0
        byte_pos += 1
    return value
