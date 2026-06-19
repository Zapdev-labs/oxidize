"""Video understanding helpers mirroring oxidize-golang/core/video.

CPU-first port of oxidize-core/src/video: frame decoding/resizing, parallel
preprocessing, a temporal self-attention encoder with RoPE + SwiGLU FFN, the
full VideoEncoder pipeline, and multimodal prompt assembly.
"""

from __future__ import annotations

from dataclasses import dataclass

from oxidize_python.core.video.config import (
    TemporalConfig,
    TemporalPool,
    VideoConfig,
    VisionConfig,
    clip_base_vision,
    default_temporal_config,
    default_video_config,
)
from oxidize_python.core.video.decoder import (
    DecodedFrame,
    RawFrameDecoder,
    RepetitiveFrameDecoder,
    ResizingDecoder,
    VideoDecoder,
    VideoSource,
    resize_rgb_nearest,
)
from oxidize_python.core.video.encoder import (
    FrameEncoder,
    VideoEncoder,
    VideoEncoderWeights,
    VideoEncoderWorkspace,
    new_video_encoder_workspace,
    zero_video_encoder_weights,
)
from oxidize_python.core.video.errors import (
    EmptySampleError,
    FrameCountOutOfRangeError,
    InvalidFrameError,
    ShapeMismatchError,
    VideoError,
)
from oxidize_python.core.video.frame_sampler import (
    FrameSamplingStrategy,
    luma_histogram_rgb,
    sample_indices,
    sample_indices_adaptive,
)
from oxidize_python.core.video.preprocess import (
    FramePreprocessFn,
    ImagePatches,
    VideoFrames,
    VideoPreprocessor,
    new_video_preprocessor,
)
from oxidize_python.core.video.prompt import (
    PromptSegment,
    VideoPrompt,
    VideoSegment,
)
from oxidize_python.core.video.temporal import (
    TemporalLayerWeights,
    TemporalWeights,
    TemporalWorkspace,
    forward_temporal,
    new_temporal_workspace,
    zero_temporal_layer_weights,
    zero_temporal_weights,
)


@dataclass
class Config:
    """Video preprocessing defaults (legacy top-level config)."""

    target_frames: int = 8
    strategy: FrameSamplingStrategy = FrameSamplingStrategy.UNIFORM
    dense_stride: int = 1


__all__ = [
    # config
    "VisionConfig",
    "TemporalConfig",
    "TemporalPool",
    "VideoConfig",
    "Config",
    "clip_base_vision",
    "default_temporal_config",
    "default_video_config",
    # decoder
    "DecodedFrame",
    "VideoSource",
    "VideoDecoder",
    "RawFrameDecoder",
    "RepetitiveFrameDecoder",
    "ResizingDecoder",
    "resize_rgb_nearest",
    # preprocess
    "ImagePatches",
    "VideoFrames",
    "VideoPreprocessor",
    "FramePreprocessFn",
    "new_video_preprocessor",
    # temporal
    "TemporalLayerWeights",
    "TemporalWeights",
    "TemporalWorkspace",
    "forward_temporal",
    "new_temporal_workspace",
    "zero_temporal_layer_weights",
    "zero_temporal_weights",
    # encoder
    "FrameEncoder",
    "VideoEncoder",
    "VideoEncoderWeights",
    "VideoEncoderWorkspace",
    "new_video_encoder_workspace",
    "zero_video_encoder_weights",
    # prompt
    "VideoPrompt",
    "PromptSegment",
    "VideoSegment",
    # sampling
    "FrameSamplingStrategy",
    "sample_indices",
    "sample_indices_adaptive",
    "luma_histogram_rgb",
    # errors
    "VideoError",
    "EmptySampleError",
    "FrameCountOutOfRangeError",
    "InvalidFrameError",
    "ShapeMismatchError",
]
