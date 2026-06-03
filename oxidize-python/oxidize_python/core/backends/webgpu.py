"""WebGPU backend stub mirroring oxidize-golang/core/backends/webgpu."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BuildInfo:
    detected_at_build: bool = False


def info() -> BuildInfo:
    return BuildInfo()
