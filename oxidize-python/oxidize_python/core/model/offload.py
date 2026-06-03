"""Multi-GPU offload planning mirroring oxidize-golang/core/model/offload.go."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class OffloadPolicy(IntEnum):
    LAYER_MAJOR = 0
    ROW_MAJOR = 1
    PIPELINE = 2

    def __str__(self) -> str:
        if self == OffloadPolicy.LAYER_MAJOR:
            return "layer-major"
        if self == OffloadPolicy.ROW_MAJOR:
            return "row-major"
        if self == OffloadPolicy.PIPELINE:
            return "pipeline"
        return "unknown"


@dataclass
class DeviceMemory:
    device_id: int
    backend: str
    bytes: int


@dataclass
class LayerAssignment:
    layer_index: int
    device_id: int
    backend: str
    bytes: int


@dataclass
class LayerOffloadPlan:
    layers: list[LayerAssignment] = field(default_factory=list)

    def total_bytes(self) -> int:
        return sum(layer.bytes for layer in self.layers)

    def by_device(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for layer in self.layers:
            out[layer.device_id] = out.get(layer.device_id, 0) + layer.bytes
        return out


@dataclass
class PipelineStage:
    stage_id: int
    layer_range: tuple[int, int]
    device_id: int
    backend: str
    bytes: int
    micro_batch: int = 1


@dataclass
class MultiGpuOffloadPlan:
    stages: list[PipelineStage] = field(default_factory=list)
    policy: OffloadPolicy = OffloadPolicy.LAYER_MAJOR
    total_bytes: int = 0

    def add_stage(self, stage: PipelineStage) -> None:
        if stage.micro_batch <= 0:
            stage.micro_batch = 1
        self.stages.append(stage)
        self.total_bytes += stage.bytes

    def validate(self, total_layers: int) -> None:
        if not self.stages:
            raise OffloadError("no stages")
        sorted_stages = sorted(self.stages, key=lambda s: s.layer_range[0])
        for i, s in enumerate(sorted_stages):
            lo, hi = s.layer_range
            if lo < 0 or hi > total_layers:
                raise OffloadError("layer range out of bounds")
            if lo >= hi:
                raise OffloadError("empty layer range")
            if i > 0 and lo != sorted_stages[i - 1].layer_range[1]:
                raise OffloadError("gap or overlap between stages")


class OffloadError(Exception):
    pass


class LayerOffloadPlanner:
    def __init__(self, devices: list[DeviceMemory], policy: OffloadPolicy) -> None:
        self.devices = devices
        self.policy = policy

    def plan(self, layer_count: int, bytes_per_layer: int) -> LayerOffloadPlan:
        if not self.devices or layer_count == 0:
            return LayerOffloadPlan()
        plan = LayerOffloadPlan()
        if self.policy == OffloadPolicy.LAYER_MAJOR:
            per = layer_count // len(self.devices)
            rem = layer_count % len(self.devices)
            idx = 0
            for d, dev in enumerate(self.devices):
                end = idx + per + (1 if d < rem else 0)
                for i in range(idx, end):
                    plan.layers.append(
                        LayerAssignment(i, dev.device_id, dev.backend, bytes_per_layer)
                    )
                idx = end
        else:
            for i in range(layer_count):
                dev = self.devices[i % len(self.devices)]
                plan.layers.append(
                    LayerAssignment(i, dev.device_id, dev.backend, bytes_per_layer)
                )
        return plan


@dataclass
class GpuOffloadConfig:
    enabled: bool = False
    max_layers_on_gpu: int = 0
    fallback_to_cpu: bool = True
    use_async_copy: bool = False


@dataclass
class OffloadMetrics:
    total_transfers: int = 0
    bytes_transferred: int = 0
    avg_latency_micros: float = 0.0

    def avg_bytes_per_transfer(self) -> float:
        if self.total_transfers == 0:
            return 0.0
        return self.bytes_transferred / self.total_transfers
