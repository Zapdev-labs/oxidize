from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class MetadataType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


@dataclass
class MetadataValue:
    type: MetadataType
    uint64: int = 0
    int64: int = 0
    float64: float = 0.0
    bool: bool = False
    string: str = ""
    array: list[MetadataValue] = field(default_factory=list)

    def as_uint64(self) -> tuple[int, bool]:
        if self.type in (
            MetadataType.UINT8,
            MetadataType.UINT16,
            MetadataType.UINT32,
            MetadataType.UINT64,
        ):
            return self.uint64, True
        if self.type in (
            MetadataType.INT8,
            MetadataType.INT16,
            MetadataType.INT32,
            MetadataType.INT64,
        ):
            if self.int64 >= 0:
                return self.int64, True
        return 0, False

    def as_float32(self) -> tuple[float, bool]:
        if self.type in (MetadataType.FLOAT32, MetadataType.FLOAT64):
            return float(self.float64), True
        n, ok = self.as_uint64()
        if ok:
            return float(n), True
        return 0.0, False


@dataclass
class TensorInfo:
    name: str
    dimensions: list[int]
    ggml_type: int
    relative_offset: int
    absolute_offset: int = 0


@dataclass
class File:
    version: int
    tensor_count: int
    metadata: dict[str, MetadataValue]
    tensor_infos: list[TensorInfo]
    alignment: int = 32
    data_section_start: int = 0
