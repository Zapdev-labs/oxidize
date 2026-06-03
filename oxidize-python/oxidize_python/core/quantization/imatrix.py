"""Importance matrix for mixed quantization."""

from __future__ import annotations

from dataclasses import dataclass

from oxidize_python.core.quantization.quantize import quantize_scalar
from oxidize_python.core.quantization.types import Error, Type, quantized_size


@dataclass
class IMatrix:
    values: list[float]

    @classmethod
    def new(cls, values: list[float]) -> IMatrix:
        return cls(values=list(values))

    def values_copy(self) -> list[float]:
        return list(self.values)

    def importance_at(self, index: int) -> float:
        if index >= len(self.values):
            return 1.0
        v = self.values[index]
        return 1.0 if v <= 0 else v


@dataclass
class MixedLayerPlan:
    name: str
    value_count: int
    target: Type


@dataclass
class QuantizedLayer:
    name: str
    target: Type
    bytes: bytes


def quantize_mixed_scalar(
    plans: list[MixedLayerPlan],
    tensors: dict[str, list[float]],
) -> list[QuantizedLayer]:
    out: list[QuantizedLayer] = []
    for p in plans:
        src = tensors.get(p.name)
        if src is None:
            raise Error(f"missing tensor {p.name}")
        if len(src) != p.value_count:
            raise Error(f"value count mismatch for {p.name}")
        size = quantized_size(p.target, p.value_count)
        buf = bytearray(size)
        quantize_scalar(p.target, src, buf, None)
        out.append(QuantizedLayer(name=p.name, target=p.target, bytes=bytes(buf)))
    return out
