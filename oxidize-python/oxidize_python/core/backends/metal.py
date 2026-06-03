"""Metal backend stub mirroring oxidize-golang/core/backends/metal."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BuildInfo:
    detected_at_build: bool = False
    sdk_path: str = ""


def info() -> BuildInfo:
    return BuildInfo(detected_at_build=False)


def available() -> bool:
    return info().detected_at_build
