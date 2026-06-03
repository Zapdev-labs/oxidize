"""SafeTensors reader mirroring oxidize-golang/core/safetensors."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from io import BufferedIOBase
from pathlib import Path


class DType(IntEnum):
    BOOL = 0
    I8 = 1
    I16 = 2
    I32 = 3
    I64 = 4
    U8 = 5
    F16 = 6
    BF16 = 7
    F32 = 8
    F64 = 9
    UNKNOWN = 10

    def __str__(self) -> str:
        names = {
            DType.BOOL: "bool",
            DType.I8: "i8",
            DType.I16: "i16",
            DType.I32: "i32",
            DType.I64: "i64",
            DType.U8: "u8",
            DType.F16: "f16",
            DType.BF16: "bf16",
            DType.F32: "f32",
            DType.F64: "f64",
        }
        return names.get(self, f"dtype({int(self)})")

    def size_in_bytes(self) -> int:
        match self:
            case DType.BOOL | DType.I8 | DType.U8:
                return 1
            case DType.I16 | DType.F16 | DType.BF16:
                return 2
            case DType.I32 | DType.F32:
                return 4
            case DType.I64 | DType.F64:
                return 8
            case _:
                return 0


class Error(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"safetensors: {message}")


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    dtype: DType
    absolute_offset: int
    size_bytes: int


class MappedFile:
    def __init__(self, path: str = "", raw: bytes = b"", tensors: list[TensorInfo] | None = None) -> None:
        self.path = path
        self._bytes = raw
        self._tensors = tensors or []

    def tensors(self) -> list[TensorInfo]:
        return self._tensors

    def bytes(self) -> bytes:
        return self._bytes

    def tensor_data(self, name: str) -> bytes:
        for info in self._tensors:
            if info.name == name:
                end = info.absolute_offset + info.size_bytes
                if end > len(self._bytes):
                    raise Error(f"{name} out of bounds")
                return bytes(self._bytes[info.absolute_offset : end])
        raise Error(f"tensor not found: {name}")


def load(path: str) -> MappedFile:
    return parse(Path(path).read_bytes())


def load_reader(r: BufferedIOBase) -> MappedFile:
    return parse(r.read())


def parse(raw: bytes) -> MappedFile:
    if len(raw) < 8:
        raise Error("file too small")
    header_len = struct.unpack_from("<Q", raw, 0)[0]
    if len(raw) < 8 + header_len:
        raise Error("header truncated")
    header_bytes = raw[8 : 8 + header_len]
    header = json.loads(header_bytes)
    infos: list[TensorInfo] = []
    data_start = int(8 + header_len)
    for name, meta_json in header.items():
        if name == "__metadata__":
            continue
        meta = json.loads(meta_json) if isinstance(meta_json, (str, bytes)) else meta_json
        dt = _parse_dtype(meta["dtype"])
        shape = meta["shape"]
        elements = 1
        for d in shape:
            elements *= d
        size_bytes = elements * dt.size_in_bytes()
        abs_offset = data_start + meta["data_offsets"][0]
        if abs_offset + size_bytes > len(raw):
            raise Error(f"tensor {name} out of bounds")
        infos.append(
            TensorInfo(
                name=name,
                shape=shape,
                dtype=dt,
                absolute_offset=abs_offset,
                size_bytes=size_bytes,
            )
        )
    return MappedFile(raw=raw, tensors=infos)


def _parse_dtype(s: str) -> DType:
    table = {
        "bool": DType.BOOL,
        "i8": DType.I8,
        "i16": DType.I16,
        "i32": DType.I32,
        "i64": DType.I64,
        "u8": DType.U8,
        "f16": DType.F16,
        "bf16": DType.BF16,
        "f32": DType.F32,
        "f64": DType.F64,
    }
    if s not in table:
        raise Error(f"unsupported dtype {s!r}")
    return table[s]
