"""Video helpers mirroring oxidize-golang/core/video."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class FrameSamplingStrategy(IntEnum):
    UNIFORM = 0
    DENSE = 1
    ADAPTIVE = 2


@dataclass
class Config:
    target_frames: int = 8
    strategy: FrameSamplingStrategy = FrameSamplingStrategy.UNIFORM
    dense_stride: int = 1


@dataclass
class DecodedFrame:
    width: int
    height: int
    data: bytes


class VideoError(Exception):
    pass


def sample_indices(total_frames: int, target_frames: int, strategy: FrameSamplingStrategy) -> list[int]:
    if total_frames <= 0 or target_frames <= 0:
        raise VideoError("frame count out of range")
    if total_frames <= target_frames:
        return list(range(total_frames))
    step = (total_frames - 1) / max(target_frames - 1, 1)
    out: list[int] = []
    seen: set[int] = set()
    for i in range(target_frames):
        idx = min(total_frames - 1, int(round(i * step)))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return sorted(out)


def luma_histogram_rgb(data: bytes) -> list[float]:
    hist = [0.0] * 16
    total = 0.0
    for i in range(0, len(data) - 2, 3):
        luma = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
        bin_idx = min(15, int(luma / 16))
        hist[bin_idx] += 1
        total += 1
    if total:
        hist = [v / total for v in hist]
    return hist
