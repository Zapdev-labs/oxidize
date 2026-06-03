"""GGUF parse/load mirroring oxidize-golang/internal/gguf/parse.go."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from oxidize_python.internal.gguf.errors import (
    err_integer_overflow,
    err_invalid_alignment,
    err_invalid_magic,
    err_unexpected_eof,
    err_unsupported_version,
)
from oxidize_python.internal.gguf.metadata_io import read_tensor, read_value
from oxidize_python.internal.gguf.reader import BinaryReader
from oxidize_python.internal.gguf.tensor_size import tensor_byte_size, tensor_element_count
from oxidize_python.internal.gguf.types import File, MetadataType, MetadataValue, TensorInfo

DEFAULT_ALIGNMENT = 32


@dataclass
class Header:
    version: int
    metadata: dict[str, MetadataValue]


def load_file(path: str | Path) -> File:
    raw = Path(path).read_bytes()
    return parse(raw)


def load_metadata(path: str | Path) -> Header:
    with Path(path).open("rb") as fh:
        r = BinaryReader(fh)
        version, _, metadata, _ = parse_header(r)
    return Header(version=version, metadata=metadata)


def validate_file(path: str | Path) -> None:
    p = Path(path)
    stat_size = p.stat().st_size
    with p.open("rb") as fh:
        r = BinaryReader(fh)
        version, tensor_count, metadata, _ = parse_header(r)
        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            tensors.append(read_tensor(r))
        alignment = _alignment_from_metadata(metadata)
        data_start = align_up(r.position(), alignment)
        limit = stat_size
        if data_start > limit:
            raise err_unexpected_eof()
        for tensor in tensors:
            abs_off = data_start + tensor.relative_offset
            if abs_off < data_start:
                raise err_integer_overflow()
            element_count = tensor_element_count(tensor.dimensions)
            byte_size = tensor_byte_size(tensor.ggml_type, element_count)
            if abs_off > limit or byte_size > limit - abs_off:
                raise err_unexpected_eof()
        if r.position() > limit:
            raise err_unexpected_eof()
        _ = version


def parse(raw: bytes) -> File:
    r = BinaryReader(BytesIO(raw))
    version, tensor_count, metadata, _ = parse_header(r)
    tensors: list[TensorInfo] = []
    for _ in range(tensor_count):
        tensors.append(read_tensor(r))
    alignment = _alignment_from_metadata(metadata)
    data_start = align_up(r.position(), alignment)
    limit = len(raw)
    if data_start > limit:
        raise err_unexpected_eof()
    for tensor in tensors:
        abs_off = data_start + tensor.relative_offset
        if abs_off < data_start:
            raise err_integer_overflow()
        element_count = tensor_element_count(tensor.dimensions)
        byte_size = tensor_byte_size(tensor.ggml_type, element_count)
        if abs_off > limit or byte_size > limit - abs_off:
            raise err_unexpected_eof()
        tensor.absolute_offset = abs_off
    return File(
        version=version,
        tensor_count=tensor_count,
        metadata=metadata,
        tensor_infos=tensors,
        alignment=alignment,
        data_section_start=data_start,
    )


def parse_header(
    r: BinaryReader,
) -> tuple[int, int, dict[str, MetadataValue], None]:
    magic = r.read_exact(4)
    if magic != b"GGUF":
        raise err_invalid_magic()
    version = r.read_u32()
    if version not in (2, 3):
        raise err_unsupported_version(version)
    tensor_count = r.read_u64()
    metadata_count = r.read_u64()
    metadata: dict[str, MetadataValue] = {}
    for _ in range(metadata_count):
        key = r.read_string()
        value_type = MetadataType(r.read_u32())
        metadata[key] = read_value(r, value_type)
    return version, tensor_count, metadata, None


def align_up(value: int, alignment: int) -> int:
    mask = alignment - 1
    total = value + mask
    if total < value:
        raise err_integer_overflow()
    return total & ~mask


def _alignment_from_metadata(metadata: dict[str, MetadataValue]) -> int:
    alignment = DEFAULT_ALIGNMENT
    if value := metadata.get("general.alignment"):
        number, ok = value.as_uint64()
        if not ok or number == 0 or (number & (number - 1)) != 0:
            raise err_invalid_alignment(number)
        alignment = number
    return alignment
