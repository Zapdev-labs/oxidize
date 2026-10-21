//! Video encoder.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//!
//! Runs the per-frame vision encoder over a [`VideoFrames`] tensor, pools
//! each frame to a single vector, then runs the temporal encoder across the
//! frame axis. The result is a `[num_frames, llm_hidden_size]` matrix of
//! video tokens ready to be dropped into a language model prompt.

use rayon::prelude::*;

use super::config::{TemporalPool, VideoConfig};
use super::error::VideoError;
use super::preprocess::VideoFrames;
use super::temporal::{TemporalWeights, TemporalWorkspace, forward_temporal};
use crate::tensor::{gemm_f32, rms_norm_f32};
use crate::vision::{VisionEncoder, VisionError};

/// Per-call scratch space for the video encoder. Reusing this across
/// invocations avoids per-frame allocations in the hot path.
#[derive(Debug, Clone)]
pub struct VideoEncoderWorkspace {
    /// Per-frame temporal input: `[num_frames * hidden_size]`.
    pub frame_temporal: Vec<f32>,
    /// Final projected tokens: `[num_frames * llm_hidden_size]`.
    pub projected: Vec<f32>,
    pub temporal_ws: TemporalWorkspace,
}

impl VideoEncoderWorkspace {
    pub fn for_config(config: &VideoConfig) -> Self {
        let llm_hidden = if config.llm_hidden_size == 0 {
            config.temporal.hidden_size
        } else {
            config.llm_hidden_size
        };
        Self {
            frame_temporal: vec![0.0_f32; config.temporal.max_frames * config.temporal.hidden_size],
            projected: vec![0.0_f32; config.temporal.max_frames * llm_hidden],
            temporal_ws: TemporalWorkspace::for_config(&config.temporal),
        }
    }
}

/// Temporal encoder weights + optional LLM projection.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoEncoderWeights {
    pub temporal: TemporalWeights,
    /// Layer norm applied before the LLM projection. Length
    /// `temporal.hidden_size`.
    pub pre_projection_norm: Vec<f32>,
    /// Linear projection from `temporal.hidden_size` to `llm_hidden_size`
    /// (or `temporal.hidden_size` when `llm_hidden_size == 0`).
    /// Length `temporal.hidden_size * llm_hidden_size`.
    pub projection: Vec<f32>,
    /// Learnable frame-position embeddings. Length
    /// `max_frames * temporal.hidden_size`. When empty, position 0 is used
    /// for every frame (effectively disabling temporal positional info).
    pub frame_pos_embedding: Vec<f32>,
}

impl VideoEncoderWeights {
    pub fn zeros(config: &VideoConfig) -> Self {
        let h = config.temporal.hidden_size;
        let llm = if config.llm_hidden_size == 0 {
            h
        } else {
            config.llm_hidden_size
        };
        Self {
            temporal: TemporalWeights::zeros(&config.temporal),
            pre_projection_norm: vec![1.0; h],
            projection: vec![0.0; h * llm],
            frame_pos_embedding: vec![0.0; config.temporal.max_frames * h],
        }
    }
}

/// Video encoder: vision encoder + temporal encoder + optional LLM
/// projection.
#[derive(Debug, Clone)]
pub struct VideoEncoder {
    config: VideoConfig,
    vision: VisionEncoder,
    weights: VideoEncoderWeights,
}

impl VideoEncoder {
    pub fn new(config: VideoConfig, vision: VisionEncoder) -> Result<Self, VideoError> {
        config.validate()?;
        let weights = VideoEncoderWeights::zeros(&config);
        Ok(Self {
            config,
            vision,
            weights,
        })
    }

    pub fn config(&self) -> &VideoConfig {
        &self.config
    }

    pub fn vision(&self) -> &VisionEncoder {
        &self.vision
    }

    pub fn weights(&self) -> &VideoEncoderWeights {
        &self.weights
    }

    /// Replace the temporal / projection weights. Useful for tests and
    /// when loading from a serialized checkpoint.
    pub fn load_weights(&mut self, weights: VideoEncoderWeights) -> Result<(), VideoError> {
        let cfg = &self.config;
        let h = cfg.temporal.hidden_size;
        let llm = if cfg.llm_hidden_size == 0 {
            h
        } else {
            cfg.llm_hidden_size
        };
        check_len("pre_projection_norm", weights.pre_projection_norm.len(), h)?;
        check_len("projection", weights.projection.len(), h * llm)?;
        let expected_pos = cfg.temporal.max_frames * h;
        check_len(
            "frame_pos_embedding",
            weights.frame_pos_embedding.len(),
            expected_pos,
        )?;
        if weights.temporal.layers.len() != cfg.temporal.num_layers {
            return Err(VideoError::WeightShapeMismatch {
                name: "temporal_layers",
                expected: cfg.temporal.num_layers,
                actual: weights.temporal.layers.len(),
            });
        }
        // Validate every layer's Q/K/V/O/FFN shapes.
        let inter = cfg.temporal.intermediate_size;
        for layer in &weights.temporal.layers {
            check_len("temporal_layer.attn_norm", layer.attn_norm.len(), h)?;
            check_len("temporal_layer.q_proj", layer.q_proj.len(), h * h)?;
            check_len("temporal_layer.k_proj", layer.k_proj.len(), h * h)?;
            check_len("temporal_layer.v_proj", layer.v_proj.len(), h * h)?;
            check_len("temporal_layer.o_proj", layer.o_proj.len(), h * h)?;
            check_len("temporal_layer.ffn_gate", layer.ffn_gate.len(), h * inter)?;
            check_len("temporal_layer.ffn_up", layer.ffn_up.len(), h * inter)?;
            check_len("temporal_layer.ffn_down", layer.ffn_down.len(), inter * h)?;
        }
        check_len("temporal_final_norm", weights.temporal.final_norm.len(), h)?;
        if cfg.temporal.use_cls_token && weights.temporal.cls_token.len() != h {
            return Err(VideoError::WeightShapeMismatch {
                name: "temporal_cls_token",
                expected: h,
                actual: weights.temporal.cls_token.len(),
            });
        }
        if !cfg.temporal.use_cls_token && !weights.temporal.cls_token.is_empty() {
            return Err(VideoError::WeightShapeMismatch {
                name: "temporal_cls_token",
                expected: 0,
                actual: weights.temporal.cls_token.len(),
            });
        }
        self.weights = weights;
        Ok(())
    }

    /// Encode a video to `[num_frames, llm_hidden_size]` tokens.
    pub fn encode(
        &self,
        frames: &VideoFrames,
        workspace: &mut VideoEncoderWorkspace,
    ) -> Result<Vec<f32>, VideoError> {
        self.config.validate()?;
        if frames.frame_count() == 0 {
            return Ok(Vec::new());
        }
        if frames.frame_count() > self.config.temporal.max_frames {
            return Err(VideoError::FrameCountOutOfRange {
                requested: frames.frame_count(),
                min: 1,
                max: self.config.temporal.max_frames,
            });
        }

        let n_frames = frames.frame_count();
        let projection_dim = self.config.vision.projection_dim;
        let hidden = self.config.temporal.hidden_size;
        let llm = if self.config.llm_hidden_size == 0 {
            hidden
        } else {
            self.config.llm_hidden_size
        };
        let num_patches = self.config.vision.num_patches();

        // ---- 1. Vision encoder per frame (parallel via rayon) ----
        // The vision encoder returns `[num_patches, projection_dim]` per
        // frame. We pool each frame to a single `[projection_dim]` vector
        // before the temporal encoder.
        let pooled_per_frame: Vec<Vec<f32>> = frames
            .frames
            .par_iter()
            .map(|patches| self.vision.encode(patches))
            .collect::<Result<Vec<_>, VisionError>>()?;

        for i in 0..n_frames {
            let patches = &pooled_per_frame[i];
            if patches.len() != num_patches * projection_dim {
                return Err(VideoError::InferenceFailed(format!(
                    "vision embedding length {} != num_patches * projection_dim ({} * {})",
                    patches.len(),
                    num_patches,
                    projection_dim
                )));
            }
            let dst = &mut workspace.frame_temporal[i * hidden..(i + 1) * hidden];
            // Pool patches to a single vector of length `projection_dim`,
            // then copy into the hidden-sized slot. (hidden == projection_dim
            // is enforced by VideoConfig::validate.)
            match self.config.pool {
                TemporalPool::Mean => {
                    for d in 0..projection_dim {
                        let mut sum = 0.0_f32;
                        for p in 0..num_patches {
                            sum += patches[p * projection_dim + d];
                        }
                        dst[d] = sum / num_patches as f32;
                    }
                }
                TemporalPool::ClsToken => {
                    dst[..projection_dim].copy_from_slice(&patches[..projection_dim]);
                }
                TemporalPool::LastToken => {
                    let start = (num_patches - 1) * projection_dim;
                    dst[..projection_dim].copy_from_slice(&patches[start..start + projection_dim]);
                }
            }
            // Zero-pad any tail if hidden > projection_dim.
            if hidden > projection_dim {
                for d in projection_dim..hidden {
                    dst[d] = 0.0;
                }
            }
            // Add positional embedding.
            if !self.weights.frame_pos_embedding.is_empty() {
                let pos = &self.weights.frame_pos_embedding[i * hidden..(i + 1) * hidden];
                for d in 0..hidden {
                    dst[d] += pos[d];
                }
            }
        }

        // ---- 2. Temporal self-attention ----
        let temporal_input = &workspace.frame_temporal[..n_frames * hidden];
        let temporal_out = forward_temporal(
            &self.config.temporal,
            &self.weights.temporal,
            temporal_input,
            n_frames,
            &mut workspace.temporal_ws,
        )?;

        // ---- 3. Pre-projection norm + LLM projection ----
        // `temporal_out` layout: [seq, hidden] where seq = n_frames + 1
        // (with cls) or n_frames (without). The cls row is dropped from
        // the output: only the per-frame rows are useful tokens.
        let offset = if self.config.temporal.use_cls_token {
            1
        } else {
            0
        };
        for i in 0..n_frames {
            let src = &temporal_out[(i + offset) * hidden..(i + offset + 1) * hidden];
            // Pre-projection norm
            let mut normalized = vec![0.0_f32; hidden];
            rms_norm_f32(
                src,
                &self.weights.pre_projection_norm,
                self.config.temporal.rms_norm_eps,
                &mut normalized,
            )
            .map_err(|e| VideoError::InferenceFailed(format!("pre_proj norm: {e:?}")))?;
            // Project to llm_hidden
            let mut proj = vec![0.0_f32; llm];
            gemm_f32(
                &normalized,
                1,
                hidden,
                &self.weights.projection,
                llm,
                &mut proj,
            )
            .map_err(|e| VideoError::InferenceFailed(format!("projection: {e:?}")))?;
            let dst = &mut workspace.projected[i * llm..(i + 1) * llm];
            dst.copy_from_slice(&proj);
        }

        Ok(workspace.projected[..n_frames * llm].to_vec())
    }
}

fn check_len(name: &'static str, actual: usize, expected: usize) -> Result<(), VideoError> {
    if actual == expected {
        Ok(())
    } else {
        Err(VideoError::WeightShapeMismatch {
            name,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::config::TemporalConfig;
    use crate::vision::VisionConfig;

    fn tiny_video_config() -> VideoConfig {
        let vision = VisionConfig {
            image_size: 4,
            patch_size: 2,
            hidden_size: 4,
            num_attention_heads: 1,
            num_hidden_layers: 1,
            intermediate_size: 8,
            layer_norm_eps: 1e-5,
            projection_dim: 4,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            num_image_tokens: 4,
        };
        let temporal = TemporalConfig {
            hidden_size: vision.projection_dim,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 8,
            max_frames: 4,
            use_cls_token: false,
            ..Default::default()
        };
        VideoConfig {
            vision,
            temporal,
            sampling: Default::default(),
            target_frames: 2,
            llm_hidden_size: 0,
            pool: TemporalPool::Mean,
            video_start_token_id: 0,
            video_end_token_id: 0,
        }
    }

    #[test]
    fn encode_empty_frames_returns_empty_tokens() {
        let cfg = tiny_video_config();
        let vision = VisionEncoder::new(cfg.vision.clone());
        let encoder = VideoEncoder::new(cfg.clone(), vision).unwrap();
        let frames = VideoFrames {
            frames: Vec::new(),
            widths: Vec::new(),
            heights: Vec::new(),
        };
        let mut ws = VideoEncoderWorkspace::for_config(&cfg);
        let tokens = encoder.encode(&frames, &mut ws).unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn encode_too_many_frames_is_rejected() {
        let mut cfg = tiny_video_config();
        cfg.target_frames = 2;
        cfg.temporal.max_frames = 2;
        let vision = VisionEncoder::new(cfg.vision.clone());
        let encoder = VideoEncoder::new(cfg.clone(), vision).unwrap();
        let frames = VideoFrames {
            frames: vec![
                crate::vision::ImagePatches {
                    data: vec![0.0; cfg.vision.num_patches() * cfg.vision.patch_dim()],
                    num_patches: cfg.vision.num_patches(),
                    patch_dim: cfg.vision.patch_dim(),
                    original_width: 4,
                    original_height: 4,
                };
                3
            ],
            widths: vec![4; 3],
            heights: vec![4; 3],
        };
        let mut ws = VideoEncoderWorkspace::for_config(&cfg);
        let err = encoder.encode(&frames, &mut ws).unwrap_err();
        assert!(matches!(err, VideoError::FrameCountOutOfRange { .. }));
    }

    #[test]
    fn load_weights_rejects_wrong_projection() {
        let cfg = tiny_video_config();
        let vision = VisionEncoder::new(cfg.vision.clone());
        let mut encoder = VideoEncoder::new(cfg.clone(), vision).unwrap();
        let mut bad = VideoEncoderWeights::zeros(&cfg);
        bad.projection = vec![0.0; 3]; // wrong size
        let err = encoder.load_weights(bad).unwrap_err();
        assert!(matches!(err, VideoError::WeightShapeMismatch { .. }));
    }
}
