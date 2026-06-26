"""Video frame preprocessing mirroring oxidize-golang/core/video/preprocess.go."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from oxidize_python.core.video.config import VisionConfig
from oxidize_python.core.video.decoder import DecodedFrame, resize_rgb_nearest
from oxidize_python.core.video.errors import VideoError


@dataclass
class ImagePatches:
    """Flattened patch tensor for a single frame.

    `num_patches` rows of length `patch_dim`, row-major. Mirrors the relevant
    fields of oxidize-core's vision::ImagePatches.
    """

    data: list[float]
    num_patches: int
    patch_dim: int
    original_width: int = 0
    original_height: int = 0


@dataclass
class VideoFrames:
    """Container of preprocessed per-frame patches consumed by the VideoEncoder.

    Mirrors preprocess.rs:VideoFrames.
    """

    frames: list[ImagePatches] = field(default_factory=list)
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)

    def frame_count(self) -> int:
        return len(self.frames)

    def total_patches(self) -> int:
        return sum(f.num_patches for f in self.frames)


# Extracts patch tensors from a single decoded RGB frame. Mirrors
# ImagePreprocessor::preprocess_rgb; supplied by the caller so the video package
# stays decoupled from a specific vision implementation.
FramePreprocessFn = Callable[[bytes, int, int], ImagePatches]


class VideoPreprocessor:
    """Validates frame consistency and runs per-frame preprocessing in parallel.

    Mirrors preprocess.rs:VideoPreprocessor.
    """

    def __init__(
        self,
        config: VisionConfig,
        preprocess: FramePreprocessFn | None = None,
    ) -> None:
        self.config = config
        self.preprocess: FramePreprocessFn = preprocess or self._default_preprocess

    def preprocess_frames(self, frames: list[DecodedFrame]) -> VideoFrames:
        """Preprocess a sequence of decoded RGB frames in parallel.

        All frames must share the same resolution. Mirrors
        VideoPreprocessor::preprocess.
        """
        if not frames:
            return VideoFrames()
        first = frames[0]
        for idx in range(1, len(frames)):
            if frames[idx].width != first.width or frames[idx].height != first.height:
                raise VideoError(
                    f"frame {idx} has dims {frames[idx].width}x{frames[idx].height}, "
                    f"expected {first.width}x{first.height}"
                )

        workers = max(1, min(os.cpu_count() or 1, len(frames)))

        def work(frame: DecodedFrame) -> ImagePatches:
            return self.preprocess(frame.data, frame.width, frame.height)

        if workers == 1:
            out = [work(f) for f in frames]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                out = list(pool.map(work, frames))

        widths = [f.width for f in frames]
        heights = [f.height for f in frames]
        return VideoFrames(frames=out, widths=widths, heights=heights)

    def _default_preprocess(self, data: bytes, width: int, height: int) -> ImagePatches:
        """Resize to image_size then extract a grid of normalized RGB patches."""
        cfg = self.config
        cfg.validate()
        size = cfg.image_size
        resized = data
        if width != size or height != size:
            resized = resize_rgb_nearest(data, width, height, size, size)
        side = cfg.num_patches_per_side()
        ps = cfg.patch_size
        num_patches = side * side
        patch_dim = cfg.patch_dim()
        out = [0.0] * (num_patches * patch_dim)

        mean = cfg.image_mean
        std = cfg.image_std
        for py in range(side):
            for px in range(side):
                patch_idx = py * side + px
                base = patch_idx * patch_dim
                d = 0
                for yy in range(ps):
                    src_y = py * ps + yy
                    for xx in range(ps):
                        src_x = px * ps + xx
                        src_idx = (src_y * size + src_x) * 3
                        for c in range(3):
                            v = resized[src_idx + c] / 255.0
                            s = std[c] if std[c] != 0 else 1.0
                            out[base + d] = (v - mean[c]) / s
                            d += 1
        return ImagePatches(
            data=out,
            num_patches=num_patches,
            patch_dim=patch_dim,
            original_width=width,
            original_height=height,
        )


def new_video_preprocessor(config: VisionConfig) -> VideoPreprocessor:
    """Build a preprocessor with the default config-derived patch extractor."""
    return VideoPreprocessor(config)
