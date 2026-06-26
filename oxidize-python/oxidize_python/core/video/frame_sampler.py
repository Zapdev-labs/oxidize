"""Frame sampling mirroring oxidize-golang/core/video/frame_sampler.go."""

from __future__ import annotations

from enum import IntEnum

from oxidize_python.core.video.errors import (
    EmptySampleError,
    FrameCountOutOfRangeError,
)


class FrameSamplingStrategy(IntEnum):
    UNIFORM = 0
    DENSE = 1
    ADAPTIVE = 2


def sample_indices(
    total_frames: int,
    target_frames: int,
    strategy: FrameSamplingStrategy = FrameSamplingStrategy.UNIFORM,
) -> list[int]:
    """Pick frame indices from [0, total_frames) using strategy."""
    if total_frames <= 0 or target_frames <= 0:
        raise FrameCountOutOfRangeError()
    if strategy == FrameSamplingStrategy.DENSE:
        indices = _dense(total_frames, target_frames, 1)
    else:
        indices = _uniform(total_frames, target_frames)
    if not indices:
        raise EmptySampleError()
    return indices


def luma_histogram_rgb(data: bytes) -> list[float]:
    """Build a 16-bin normalized luma histogram for an RGB frame."""
    hist = [0.0] * 16
    total = 0.0
    for i in range(0, len(data) - 2, 3):
        luma = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
        bin_idx = min(15, int(luma / 16))
        hist[bin_idx] += 1.0
        total += 1.0
    if total:
        hist = [v / total for v in hist]
    return hist


def sample_indices_adaptive(
    total_frames: int,
    target_frames: int,
    luma_hists: list[float],
) -> list[int]:
    """Keep first/last frames and fill remaining slots by histogram distance.

    `luma_hists` is a flattened array of `total_frames` 16-bin histograms
    (row-major). Falls back to uniform sampling when it is too short.
    """
    if total_frames <= 0 or target_frames <= 0:
        raise FrameCountOutOfRangeError()
    if len(luma_hists) < total_frames * 16:
        return sample_indices(total_frames, target_frames, FrameSamplingStrategy.ADAPTIVE)
    if total_frames <= target_frames:
        return list(range(total_frames))

    chosen: set[int] = {0, total_frames - 1}
    out: list[int] = [0, total_frames - 1]
    while len(out) < target_frames:
        best_idx = -1
        best_score = 0.0
        for cand in range(total_frames):
            if cand in chosen:
                continue
            score = _min_hist_distance(cand, out, luma_hists)
            if best_idx < 0 or score > best_score:
                best_idx = cand
                best_score = score
        if best_idx < 0:
            break
        chosen.add(best_idx)
        out.append(best_idx)
    out.sort()
    if not out:
        raise EmptySampleError()
    return out


def _uniform(total: int, target: int) -> list[int]:
    if total <= target:
        return list(range(total))
    step = (total - 1) / max(target - 1, 1)
    out: list[int] = []
    seen: set[int] = set()
    for i in range(target):
        idx = min(total - 1, int(i * step + 0.5))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    out.sort()
    return out


def _dense(total: int, target: int, stride: int) -> list[int]:
    if stride <= 0:
        stride = 1
    out: list[int] = []
    i = 0
    while i < total and len(out) < target:
        out.append(i)
        i += stride
    return out


def _min_hist_distance(cand: int, chosen: list[int], hists: list[float]) -> float:
    cand_hist = hists[cand * 16 : (cand + 1) * 16]
    best = 0.0
    for idx in chosen:
        other = hists[idx * 16 : (idx + 1) * 16]
        d = _l1(cand_hist, other)
        if best == 0.0 or d < best:
            best = d
    return best


def _l1(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))
