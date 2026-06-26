"""Video error hierarchy mirroring oxidize-core/src/video/error.rs:VideoError."""

from __future__ import annotations


class VideoError(Exception):
    """Base error for all video subsystem failures."""


class EmptySampleError(VideoError):
    """Raised when frame sampling produces no indices."""

    def __init__(self) -> None:
        super().__init__("empty frame sample")


class FrameCountOutOfRangeError(VideoError):
    """Raised when a frame count falls outside the valid range."""

    def __init__(self, message: str = "frame count out of range") -> None:
        super().__init__(message)


class InvalidFrameError(VideoError):
    """Raised when a decoded frame has invalid dimensions or payload."""


class ShapeMismatchError(VideoError):
    """Raised when a weight or buffer has an unexpected length."""
