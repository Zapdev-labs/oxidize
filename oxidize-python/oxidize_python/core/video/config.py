"""Video model configuration mirroring oxidize-golang/core/video/config.go."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from oxidize_python.core.video.errors import VideoError
from oxidize_python.core.video.frame_sampler import FrameSamplingStrategy


@dataclass
class VisionConfig:
    """Subset of vision-encoder configuration the video stack needs.

    Mirrors the relevant fields of oxidize-core's VisionConfig without coupling
    the video package to a specific vision implementation.
    """

    image_size: int = 224
    patch_size: int = 14
    hidden_size: int = 768
    num_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-5
    projection_dim: int = 2048
    image_mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    num_image_tokens: int = 256

    def num_patches_per_side(self) -> int:
        if self.patch_size == 0:
            return 0
        return self.image_size // self.patch_size

    def num_patches(self) -> int:
        side = self.num_patches_per_side()
        return side * side

    def patch_dim(self) -> int:
        return 3 * self.patch_size * self.patch_size

    def validate(self) -> None:
        if self.image_size == 0 or self.patch_size == 0:
            raise VideoError("image_size and patch_size must be non-zero")
        if self.image_size % self.patch_size != 0:
            raise VideoError("image_size must be divisible by patch_size")
        if self.projection_dim == 0:
            raise VideoError("projection_dim must be non-zero")


def clip_base_vision() -> VisionConfig:
    """Return a small CLIP-base style vision config."""
    return VisionConfig()


class TemporalPool(IntEnum):
    """How per-frame patch embeddings are aggregated into one vector per frame."""

    MEAN = 0
    CLS_TOKEN = 1
    LAST_TOKEN = 2


@dataclass
class TemporalConfig:
    """Temporal encoder configuration.

    Mirrors oxidize-core/src/video/config.rs:TemporalConfig.
    """

    hidden_size: int = 1024
    num_layers: int = 2
    num_heads: int = 8
    intermediate_size: int = 4096
    rms_norm_eps: float = 1e-5
    max_frames: int = 32
    rope_theta: float = 10000.0
    use_cls_token: bool = True
    # Stored for checkpoint compatibility but unused at inference time.
    layer_dropout: float = 0.0

    def head_dim(self) -> int:
        if self.num_heads == 0:
            return 0
        return self.hidden_size // self.num_heads

    def validate(self) -> None:
        if self.hidden_size == 0:
            raise VideoError("hidden_size must be non-zero")
        if self.num_heads == 0:
            raise VideoError("num_heads must be non-zero")
        if self.hidden_size % self.num_heads != 0:
            raise VideoError("hidden_size must be divisible by num_heads")
        if self.num_layers == 0:
            raise VideoError("num_layers must be non-zero")
        if self.intermediate_size == 0:
            raise VideoError("intermediate_size must be non-zero")
        if self.max_frames == 0:
            raise VideoError("max_frames must be non-zero")
        if self.rms_norm_eps <= 0.0:
            raise VideoError("rms_norm_eps must be positive")
        if self.rope_theta <= 0.0:
            raise VideoError("rope_theta must be positive")


def default_temporal_config() -> TemporalConfig:
    """Mirror Rust TemporalConfig::default()."""
    return TemporalConfig()


@dataclass
class VideoConfig:
    """Top-level video model configuration.

    Mirrors oxidize-core/src/video/config.rs:VideoConfig.
    """

    vision: VisionConfig = field(default_factory=clip_base_vision)
    temporal: TemporalConfig = field(default_factory=default_temporal_config)
    sampling: FrameSamplingStrategy = FrameSamplingStrategy.UNIFORM
    # Number of frames the sampler produces; must be <= temporal.max_frames.
    target_frames: int = 8
    # Output projection dim. When 0 the temporal hidden size is reused.
    llm_hidden_size: int = 0
    pool: TemporalPool = TemporalPool.MEAN
    video_start_token_id: int = 0
    video_end_token_id: int = 0

    def effective_llm_hidden(self) -> int:
        if self.llm_hidden_size == 0:
            return self.temporal.hidden_size
        return self.llm_hidden_size

    def validate(self) -> None:
        try:
            self.vision.validate()
        except VideoError as exc:
            raise VideoError(f"vision: {exc}") from exc
        if self.vision.projection_dim != self.temporal.hidden_size:
            raise VideoError(
                f"temporal.hidden_size ({self.temporal.hidden_size}) must equal "
                f"vision.projection_dim ({self.vision.projection_dim})"
            )
        self.temporal.validate()
        if self.target_frames == 0:
            raise VideoError("target_frames must be non-zero")
        if self.target_frames > self.temporal.max_frames:
            raise VideoError(
                f"target_frames ({self.target_frames}) exceeds "
                f"temporal.max_frames ({self.temporal.max_frames})"
            )


def default_video_config() -> VideoConfig:
    """Return a small CPU-friendly video config."""
    vision = clip_base_vision()
    temporal = default_temporal_config()
    temporal.hidden_size = vision.projection_dim
    temporal.num_layers = 2
    temporal.num_heads = 4
    temporal.intermediate_size = 2048
    temporal.max_frames = 16
    return VideoConfig(
        vision=vision,
        temporal=temporal,
        sampling=FrameSamplingStrategy.UNIFORM,
        target_frames=8,
        llm_hidden_size=0,
        pool=TemporalPool.MEAN,
    )
