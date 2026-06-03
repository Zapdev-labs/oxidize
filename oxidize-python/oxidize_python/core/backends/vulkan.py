"""Vulkan backend stub mirroring oxidize-golang/core/backends/vulkan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


@dataclass
class BuildInfo:
    detected_at_build: bool = False
    loader_path: str = ""


def info() -> BuildInfo:
    return BuildInfo()


class DeviceClass(IntEnum):
    INTEL_ARC = 0
    INTEL_INTEGRATED = 1
    NVIDIA = 2
    AMD = 3
    OTHER = 4

    def __str__(self) -> str:
        names = {
            DeviceClass.INTEL_ARC: "intel-arc",
            DeviceClass.INTEL_INTEGRATED: "intel-integrated",
            DeviceClass.NVIDIA: "nvidia",
            DeviceClass.AMD: "amd",
        }
        return names.get(self, "other")


@dataclass
class DeviceInfo:
    vendor_id: int
    device_id: int
    device_name: str
    device_class: DeviceClass
    compute_queue_family: int = 0


def classify_device(vendor_id: int, device_id: int, name: str) -> DeviceClass:
    if vendor_id == 0x8086:
        if "Arc" in name:
            return DeviceClass.INTEL_ARC
        return DeviceClass.INTEL_INTEGRATED
    if vendor_id == 0x10DE:
        return DeviceClass.NVIDIA
    if vendor_id == 0x1002:
        return DeviceClass.AMD
    _ = device_id
    return DeviceClass.OTHER
