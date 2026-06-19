"""Video encoder mirroring oxidize-golang/core/video/encoder.go."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from oxidize_python.core.tensor.gemv import gemm_f32
from oxidize_python.core.tensor.ops import rms_norm_f32
from oxidize_python.core.video.config import TemporalPool, VideoConfig
from oxidize_python.core.video.errors import ShapeMismatchError, VideoError
from oxidize_python.core.video.preprocess import ImagePatches, VideoFrames
from oxidize_python.core.video.temporal import (
    TemporalWeights,
    TemporalWorkspace,
    forward_temporal,
    new_temporal_workspace,
    zero_temporal_weights,
)


class FrameEncoder(ABC):
    """Encodes one frame's patch tensor into a flattened
    [num_patches, projection_dim] embedding matrix.

    Abstracts the per-frame vision encoder so the video stack stays decoupled
    from a concrete implementation. Mirrors the VisionEncoder::encode call used
    in encoder.rs.
    """

    @abstractmethod
    def encode_frame(self, patches: ImagePatches) -> list[float]:
        ...


@dataclass
class VideoEncoderWeights:
    """Temporal encoder weights plus the LLM projection.

    Mirrors encoder.rs:VideoEncoderWeights.
    """

    temporal: TemporalWeights
    # Applied before the LLM projection (len hidden_size).
    pre_projection_norm: list[float]
    # Maps hidden_size -> llm_hidden_size (len hidden*llm).
    projection: list[float]
    # Learnable frame-position embedding (len max_frames*hidden). When empty, no
    # positional info is added.
    frame_pos_embedding: list[float] = field(default_factory=list)


def zero_video_encoder_weights(cfg: VideoConfig) -> VideoEncoderWeights:
    """Build zero-initialized weights for cfg."""
    h = cfg.temporal.hidden_size
    llm = cfg.effective_llm_hidden()
    return VideoEncoderWeights(
        temporal=zero_temporal_weights(cfg.temporal),
        pre_projection_norm=[1.0] * h,
        projection=[0.0] * (h * llm),
        frame_pos_embedding=[0.0] * (cfg.temporal.max_frames * h),
    )


@dataclass
class VideoEncoderWorkspace:
    """Reusable scratch buffers for VideoEncoder.encode.

    Mirrors encoder.rs:VideoEncoderWorkspace.
    """

    frame_temporal: list[float]
    projected: list[float]
    temporal_ws: TemporalWorkspace


def new_video_encoder_workspace(cfg: VideoConfig) -> VideoEncoderWorkspace:
    """Allocate scratch buffers for cfg's worst case."""
    llm = cfg.effective_llm_hidden()
    return VideoEncoderWorkspace(
        frame_temporal=[0.0] * (cfg.temporal.max_frames * cfg.temporal.hidden_size),
        projected=[0.0] * (cfg.temporal.max_frames * llm),
        temporal_ws=new_temporal_workspace(cfg.temporal),
    )


def _check_len(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ShapeMismatchError(
            f"{name} shape mismatch: expected {expected} got {actual}"
        )


class VideoEncoder:
    """Runs the vision encoder per frame, pools each frame to a vector, runs the
    temporal encoder over the frame axis, and projects to the LLM hidden size.

    Mirrors encoder.rs:VideoEncoder.
    """

    def __init__(self, config: VideoConfig, vision: FrameEncoder) -> None:
        config.validate()
        if vision is None:
            raise VideoError("vision frame encoder must not be nil")
        self._config = config
        self._vision = vision
        self._weights = zero_video_encoder_weights(config)

    @property
    def config(self) -> VideoConfig:
        return self._config

    @property
    def weights(self) -> VideoEncoderWeights:
        return self._weights

    def load_weights(self, w: VideoEncoderWeights) -> None:
        """Validate and install new temporal/projection weights.

        Mirrors encoder.rs:VideoEncoder::load_weights.
        """
        cfg = self._config
        h = cfg.temporal.hidden_size
        llm = cfg.effective_llm_hidden()
        inter = cfg.temporal.intermediate_size
        _check_len("pre_projection_norm", len(w.pre_projection_norm), h)
        _check_len("projection", len(w.projection), h * llm)
        _check_len("frame_pos_embedding", len(w.frame_pos_embedding), cfg.temporal.max_frames * h)
        if len(w.temporal.layers) != cfg.temporal.num_layers:
            raise ShapeMismatchError(
                f"temporal_layers: expected {cfg.temporal.num_layers} "
                f"got {len(w.temporal.layers)}"
            )
        for layer in w.temporal.layers:
            _check_len("temporal_layer.attn_norm", len(layer.attn_norm), h)
            _check_len("temporal_layer.q_proj", len(layer.q_proj), h * h)
            _check_len("temporal_layer.k_proj", len(layer.k_proj), h * h)
            _check_len("temporal_layer.v_proj", len(layer.v_proj), h * h)
            _check_len("temporal_layer.o_proj", len(layer.o_proj), h * h)
            _check_len("temporal_layer.ffn_gate", len(layer.ffn_gate), h * inter)
            _check_len("temporal_layer.ffn_up", len(layer.ffn_up), h * inter)
            _check_len("temporal_layer.ffn_down", len(layer.ffn_down), inter * h)
        _check_len("temporal_final_norm", len(w.temporal.final_norm), h)
        if cfg.temporal.use_cls_token and len(w.temporal.cls_token) != h:
            raise ShapeMismatchError(
                f"temporal_cls_token: expected {h} got {len(w.temporal.cls_token)}"
            )
        if not cfg.temporal.use_cls_token and len(w.temporal.cls_token) != 0:
            raise ShapeMismatchError(
                f"temporal_cls_token: expected 0 got {len(w.temporal.cls_token)}"
            )
        self._weights = w

    def encode(
        self,
        frames: VideoFrames,
        ws: VideoEncoderWorkspace | None = None,
    ) -> list[float]:
        """Produce a [num_frames, llm_hidden_size] token matrix from
        preprocessed frames. Mirrors encoder.rs:VideoEncoder::encode.
        """
        cfg = self._config
        cfg.validate()
        n_frames = frames.frame_count()
        if n_frames == 0:
            return []
        if n_frames > cfg.temporal.max_frames:
            raise VideoError(
                f"frame count {n_frames} out of range [1, {cfg.temporal.max_frames}]"
            )
        if ws is None:
            ws = new_video_encoder_workspace(cfg)

        projection_dim = cfg.vision.projection_dim
        hidden = cfg.temporal.hidden_size
        llm = cfg.effective_llm_hidden()
        num_patches = cfg.vision.num_patches()

        # ---- 1. Vision encoder per frame (parallel) ----
        workers = max(1, min(os.cpu_count() or 1, n_frames))
        if workers == 1:
            pooled = [self._vision.encode_frame(f) for f in frames.frames]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pooled = list(pool.map(self._vision.encode_frame, frames.frames))

        # ---- Pool each frame to a single hidden-sized vector ----
        for i in range(n_frames):
            patches = pooled[i]
            if len(patches) != num_patches * projection_dim:
                raise VideoError(
                    f"vision embedding length {len(patches)} != "
                    f"num_patches*projection_dim ({num_patches}*{projection_dim})"
                )
            dst = ws.frame_temporal
            base = i * hidden
            if cfg.pool == TemporalPool.MEAN:
                for d in range(projection_dim):
                    acc = 0.0
                    for p in range(num_patches):
                        acc += patches[p * projection_dim + d]
                    dst[base + d] = acc / num_patches
            elif cfg.pool == TemporalPool.CLS_TOKEN:
                dst[base : base + projection_dim] = patches[:projection_dim]
            elif cfg.pool == TemporalPool.LAST_TOKEN:
                start = (num_patches - 1) * projection_dim
                dst[base : base + projection_dim] = patches[start : start + projection_dim]
            for d in range(projection_dim, hidden):
                dst[base + d] = 0.0
            if self._weights.frame_pos_embedding:
                pos = self._weights.frame_pos_embedding
                for d in range(hidden):
                    dst[base + d] += pos[base + d]

        # ---- 2. Temporal self-attention ----
        temporal_input = ws.frame_temporal[: n_frames * hidden]
        temporal_out = forward_temporal(
            cfg.temporal, self._weights.temporal, temporal_input, n_frames, ws.temporal_ws
        )

        # ---- 3. Pre-projection norm + LLM projection ----
        offset = 1 if cfg.temporal.use_cls_token else 0
        normalized = [0.0] * hidden
        for i in range(n_frames):
            src = temporal_out[(i + offset) * hidden : (i + offset + 1) * hidden]
            rms_norm_f32(
                src, self._weights.pre_projection_norm, normalized, cfg.temporal.rms_norm_eps
            )
            dst = [0.0] * llm
            gemm_f32(normalized, self._weights.projection, 1, hidden, llm, dst)
            ws.projected[i * llm : (i + 1) * llm] = dst

        return ws.projected[: n_frames * llm]
