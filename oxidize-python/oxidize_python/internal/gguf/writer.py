"""GGUF writer mirroring oxidize-golang/internal/gguf/writer.go."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from io import BytesIO

from oxidize_python.internal.gguf.parse import align_up
from oxidize_python.internal.gguf.types import MetadataType, MetadataValue, TensorInfo


@dataclass
class WriterHeader:
    version: int = 3
    tensor_count: int = 0
    metadata_count: int = 0
    metadata: dict[str, MetadataValue] | None = None
    tensors: list[TensorInfo] | None = None
    alignment: int = 32
    data_section_start: int = 0


def encode(header: WriterHeader, body: bytes) -> bytes:
    h = header
    if h.version == 0:
        h.version = 3
    if h.alignment == 0:
        h.alignment = 32
    metadata = h.metadata or {}
    tensors = h.tensors or []
    if h.metadata_count == 0:
        h.metadata_count = len(metadata)
    if h.tensor_count == 0:
        h.tensor_count = len(tensors)

    buf = BytesIO()
    buf.write(b"GGUF")
    buf.write(struct.pack("<I", h.version))
    buf.write(struct.pack("<Q", h.tensor_count))
    buf.write(struct.pack("<Q", h.metadata_count))
    for key, value in metadata.items():
        _write_string(buf, key)
        _write_value(buf, value)
    for tensor in tensors:
        _write_string(buf, tensor.name)
        buf.write(struct.pack("<I", len(tensor.dimensions)))
        for dim in tensor.dimensions:
            buf.write(struct.pack("<Q", dim))
        buf.write(struct.pack("<I", tensor.ggml_type))
        buf.write(struct.pack("<Q", tensor.relative_offset))

    header_end = buf.tell()
    data_start = align_up(header_end, h.alignment)
    while buf.tell() < data_start:
        buf.write(b"\x00")
    if buf.tell() != data_start:
        raise ValueError("alignment mismatch")
    buf.write(body)
    return buf.getvalue()


def _write_string(buf: BytesIO, s: str) -> None:
    data = s.encode("utf-8")
    buf.write(struct.pack("<Q", len(data)))
    buf.write(data)


def _write_value(buf: BytesIO, value: MetadataValue) -> None:
    buf.write(struct.pack("<I", int(value.type)))
    _write_value_body(buf, value)


def _write_value_body(buf: BytesIO, value: MetadataValue) -> None:
    match value.type:
        case MetadataType.UINT8:
            buf.write(struct.pack("<B", value.uint64 & 0xFF))
        case MetadataType.INT8:
            buf.write(struct.pack("<b", value.int64))
        case MetadataType.UINT16:
            buf.write(struct.pack("<H", value.uint64 & 0xFFFF))
        case MetadataType.INT16:
            buf.write(struct.pack("<h", value.int64))
        case MetadataType.UINT32:
            buf.write(struct.pack("<I", value.uint64 & 0xFFFFFFFF))
        case MetadataType.INT32:
            buf.write(struct.pack("<i", value.int64))
        case MetadataType.FLOAT32:
            buf.write(struct.pack("<f", value.float64))
        case MetadataType.BOOL:
            buf.write(struct.pack("<B", 1 if value.bool else 0))
        case MetadataType.STRING:
            _write_string(buf, value.string)
        case MetadataType.ARRAY:
            if not value.array:
                buf.write(struct.pack("<I", int(MetadataType.UINT8)))
                buf.write(struct.pack("<Q", 0))
            else:
                buf.write(struct.pack("<I", int(value.array[0].type)))
                buf.write(struct.pack("<Q", len(value.array)))
                for element in value.array:
                    _write_value_body(buf, element)
        case MetadataType.UINT64:
            buf.write(struct.pack("<Q", value.uint64))
        case MetadataType.INT64:
            buf.write(struct.pack("<q", value.int64))
        case MetadataType.FLOAT64:
            buf.write(struct.pack("<d", value.float64))
        case _:
            raise ValueError(f"unsupported metadata type {value.type}")
