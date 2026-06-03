"""Binary GGUF reader mirroring oxidize-golang/internal/gguf/reader.go."""

from __future__ import annotations

import struct
from io import BufferedIOBase, BytesIO

from oxidize_python.internal.gguf.errors import (
    MAX_GGUF_STRING_BYTES,
    err_integer_overflow,
    err_string_too_long,
    err_unexpected_eof,
)


class BinaryReader:
    def __init__(self, src: BufferedIOBase | BytesIO) -> None:
        self._src = src
        self._cursor = 0

    def position(self) -> int:
        return self._cursor

    def read_exact(self, length: int) -> bytes:
        if length < 0:
            raise err_integer_overflow()
        out = self._src.read(length)
        if len(out) != length:
            raise err_unexpected_eof()
        self._cursor += length
        return out

    def read_u8(self) -> int:
        return self.read_exact(1)[0]

    def read_u16(self) -> int:
        return struct.unpack("<H", self.read_exact(2))[0]

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self.read_exact(8))[0]

    def read_i8(self) -> int:
        return struct.unpack("<b", self.read_exact(1))[0]

    def read_i16(self) -> int:
        return struct.unpack("<h", self.read_exact(2))[0]

    def read_i32(self) -> int:
        return struct.unpack("<i", self.read_exact(4))[0]

    def read_i64(self) -> int:
        return struct.unpack("<q", self.read_exact(8))[0]

    def read_f32(self) -> float:
        return struct.unpack("<f", self.read_exact(4))[0]

    def read_f64(self) -> float:
        return struct.unpack("<d", self.read_exact(8))[0]

    def read_string(self) -> str:
        length = self.read_u64()
        if length > (1 << 63) - 1:
            raise err_integer_overflow()
        if length > MAX_GGUF_STRING_BYTES:
            raise err_string_too_long(length)
        return self.read_exact(int(length)).decode("utf-8", errors="replace")
