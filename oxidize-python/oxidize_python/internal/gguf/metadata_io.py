"""Tensor and metadata value I/O mirroring oxidize-golang/internal/gguf/metadata.go."""

from __future__ import annotations

from oxidize_python.internal.gguf.errors import err_unknown_metadata_type
from oxidize_python.internal.gguf.reader import BinaryReader
from oxidize_python.internal.gguf.types import MetadataType, MetadataValue, TensorInfo


def read_tensor(r: BinaryReader) -> TensorInfo:
    name = r.read_string()
    dimension_count = r.read_u32()
    dimensions: list[int] = []
    for _ in range(dimension_count):
        dimensions.append(r.read_u64())
    ggml_type = r.read_u32()
    relative_offset = r.read_u64()
    return TensorInfo(
        name=name,
        dimensions=dimensions,
        ggml_type=ggml_type,
        relative_offset=relative_offset,
    )


def read_value(r: BinaryReader, kind: MetadataType) -> MetadataValue:
    match kind:
        case MetadataType.UINT8:
            value = r.read_u8()
            return MetadataValue(type=kind, uint64=value)
        case MetadataType.INT8:
            value = r.read_i8()
            return MetadataValue(type=kind, int64=value)
        case MetadataType.UINT16:
            value = r.read_u16()
            return MetadataValue(type=kind, uint64=value)
        case MetadataType.INT16:
            value = r.read_i16()
            return MetadataValue(type=kind, int64=value)
        case MetadataType.UINT32:
            value = r.read_u32()
            return MetadataValue(type=kind, uint64=value)
        case MetadataType.INT32:
            value = r.read_i32()
            return MetadataValue(type=kind, int64=value)
        case MetadataType.FLOAT32:
            value = r.read_f32()
            return MetadataValue(type=kind, float64=value)
        case MetadataType.BOOL:
            value = r.read_u8()
            return MetadataValue(type=kind, bool=value != 0)
        case MetadataType.STRING:
            value = r.read_string()
            return MetadataValue(type=kind, string=value)
        case MetadataType.ARRAY:
            element_type = MetadataType(r.read_u32())
            length = r.read_u64()
            values: list[MetadataValue] = []
            for _ in range(length):
                values.append(read_value(r, element_type))
            return MetadataValue(type=kind, array=values)
        case MetadataType.UINT64:
            value = r.read_u64()
            return MetadataValue(type=kind, uint64=value)
        case MetadataType.INT64:
            value = r.read_i64()
            return MetadataValue(type=kind, int64=value)
        case MetadataType.FLOAT64:
            value = r.read_f64()
            return MetadataValue(type=kind, float64=value)
        case _:
            raise err_unknown_metadata_type(int(kind))
