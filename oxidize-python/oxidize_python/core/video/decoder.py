"""Video decoders mirroring oxidize-golang/core/video/{video,decoder}.go."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from oxidize_python.core.video.errors import (
    FrameCountOutOfRangeError,
    InvalidFrameError,
    VideoError,
)


@dataclass
class DecodedFrame:
    """A single RGB frame in row-major layout (3 bytes per pixel)."""

    width: int
    height: int
    data: bytes

    @classmethod
    def new(cls, width: int, height: int, data: bytes) -> DecodedFrame:
        """Validate dimensions and payload length, copying the data."""
        expected = width * height * 3
        if width <= 0 or height <= 0 or len(data) != expected:
            raise InvalidFrameError(
                f"invalid frame {width}x{height} bytes={len(data)}"
            )
        return cls(width=width, height=height, data=bytes(data))


@dataclass
class VideoSource:
    """Identifies input to a decoder."""

    frames: list[DecodedFrame] = field(default_factory=list)
    single_image: DecodedFrame | None = None


class VideoDecoder(ABC):
    """Decodes a source into RGB frames."""

    @abstractmethod
    def decode(self, source: VideoSource) -> list[DecodedFrame]:
        ...


class RawFrameDecoder(VideoDecoder):
    """Returns pre-decoded frames unchanged."""

    def decode(self, source: VideoSource) -> list[DecodedFrame]:
        if source.frames:
            return list(source.frames)
        if source.single_image is not None:
            return [source.single_image]
        raise FrameCountOutOfRangeError()


class RepetitiveFrameDecoder(VideoDecoder):
    """Repeats a single image n times (CLI --video-frame mode)."""

    def __init__(self, count: int = 1) -> None:
        self.count = count

    def decode(self, source: VideoSource) -> list[DecodedFrame]:
        n = self.count if self.count > 0 else 1
        img = source.single_image
        if img is None and len(source.frames) == 1:
            img = source.frames[0]
        if img is None:
            raise FrameCountOutOfRangeError()
        return [
            DecodedFrame(width=img.width, height=img.height, data=bytes(img.data))
            for _ in range(n)
        ]


class ResizingDecoder(VideoDecoder):
    """Wraps another decoder and resizes every frame with nearest-neighbor.

    Mirrors oxidize-core/src/video/decoder.rs:ResizingDecoder.
    """

    def __init__(
        self,
        inner: VideoDecoder,
        target_width: int,
        target_height: int,
    ) -> None:
        self.inner = inner
        self.target_width = target_width
        self.target_height = target_height

    def decode(self, source: VideoSource) -> list[DecodedFrame]:
        if self.inner is None:
            raise VideoError("ResizingDecoder requires an inner decoder")
        if self.target_width <= 0 or self.target_height <= 0:
            raise VideoError("ResizingDecoder target dimensions must be positive")
        frames = self.inner.decode(source)
        out: list[DecodedFrame] = []
        for f in frames:
            if f.width == self.target_width and f.height == self.target_height:
                out.append(
                    DecodedFrame(width=f.width, height=f.height, data=bytes(f.data))
                )
                continue
            resized = resize_rgb_nearest(
                f.data, f.width, f.height, self.target_width, self.target_height
            )
            out.append(
                DecodedFrame(
                    width=self.target_width,
                    height=self.target_height,
                    data=resized,
                )
            )
        return out


def resize_rgb_nearest(
    src: bytes,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> bytes:
    """Resize a row-major RGB image (3 bytes/pixel) via integer nearest-neighbor.

    Matches the Rust/Go reference exactly.
    """
    dst = bytearray(dst_w * dst_h * 3)
    if src_w <= 0 or src_h <= 0:
        return bytes(dst)
    for dy in range(dst_h):
        sy = dy * src_h // dst_h
        for dx in range(dst_w):
            sx = dx * src_w // dst_w
            src_idx = (sy * src_w + sx) * 3
            dst_idx = (dy * dst_w + dx) * 3
            if src_idx + 3 <= len(src):
                dst[dst_idx : dst_idx + 3] = src[src_idx : src_idx + 3]
    return bytes(dst)
