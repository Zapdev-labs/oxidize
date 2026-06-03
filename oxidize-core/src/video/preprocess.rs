//! Video preprocessing.
//!
//! Wraps the per-frame [`ImagePreprocessor`](crate::vision::ImagePreprocessor)
//! and adds video-level concerns: validating that all frames share a common
//! resolution, running per-frame preprocessing in parallel via rayon, and
//! producing a [`VideoFrames`] container the encoder can consume without
//! re-allocating.

use rayon::prelude::*;

use super::decoder::DecodedFrame;
use super::error::VideoError;
use crate::vision::{ImagePatches, ImagePreprocessor, VisionConfig};

/// All frames of a video after preprocessing, ready to be fed into a
/// [`VideoEncoder`](super::encoder::VideoEncoder).
#[derive(Debug, Clone, PartialEq)]
pub struct VideoFrames {
    /// Preprocessed patches for every frame in the input order.
    pub frames: Vec<ImagePatches>,
    /// Source dimensions of every frame (after any decoder-side resize).
    pub widths: Vec<usize>,
    pub heights: Vec<usize>,
}

impl VideoFrames {
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn total_patches(&self) -> usize {
        self.frames.iter().map(|p| p.num_patches).sum()
    }
}

/// Preprocesses a sequence of decoded RGB frames into patch tensors that the
/// vision encoder can consume directly.
///
/// The preprocessor validates that every frame shares the same resolution,
/// then preprocesses all frames in parallel using rayon. A single
/// [`ImagePreprocessor`] is reused across frames to avoid per-frame
/// allocation.
#[derive(Debug, Clone)]
pub struct VideoPreprocessor {
    inner: ImagePreprocessor,
}

impl VideoPreprocessor {
    pub fn new(config: VisionConfig) -> Self {
        Self {
            inner: ImagePreprocessor::new(config),
        }
    }

    pub fn config(&self) -> &VisionConfig {
        self.inner.config()
    }

    /// Preprocess a sequence of decoded RGB frames. All frames must share
    /// the same `width x height`.
    pub fn preprocess(&self, frames: &[DecodedFrame]) -> Result<VideoFrames, VideoError> {
        if frames.is_empty() {
            return Ok(VideoFrames {
                frames: Vec::new(),
                widths: Vec::new(),
                heights: Vec::new(),
            });
        }
        let first = &frames[0];
        for (idx, frame) in frames.iter().enumerate().skip(1) {
            if frame.width != first.width || frame.height != first.height {
                return Err(VideoError::InvalidConfig(format!(
                    "frame {idx} has dims {}x{}, expected {}x{}",
                    frame.width, frame.height, first.width, first.height
                )));
            }
        }

        // rayon parallel map: independent work per frame, no shared state.
        let preprocessed: Result<Vec<ImagePatches>, VideoError> = frames
            .par_iter()
            .map(|frame| {
                self.inner
                    .preprocess_rgb(&frame.data, frame.width, frame.height)
                    .map_err(VideoError::from)
            })
            .collect();

        let widths = frames.iter().map(|f| f.width).collect();
        let heights = frames.iter().map(|f| f.height).collect();

        Ok(VideoFrames {
            frames: preprocessed?,
            widths,
            heights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, fill: u8) -> DecodedFrame {
        DecodedFrame::new(w, h, vec![fill; w * h * 3]).unwrap()
    }

    fn tiny_config() -> VisionConfig {
        VisionConfig {
            image_size: 4,
            patch_size: 2,
            hidden_size: 4,
            num_attention_heads: 1,
            num_hidden_layers: 1,
            intermediate_size: 8,
            layer_norm_eps: 1e-5,
            projection_dim: 3,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            num_image_tokens: 4,
        }
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let prep = VideoPreprocessor::new(tiny_config());
        let out = prep.preprocess(&[]).unwrap();
        assert_eq!(out.frame_count(), 0);
    }

    #[test]
    fn mixed_resolutions_are_rejected() {
        let prep = VideoPreprocessor::new(tiny_config());
        let frames = vec![frame(4, 4, 0), frame(6, 4, 0)];
        let err = prep.preprocess(&frames).unwrap_err();
        assert!(matches!(err, VideoError::InvalidConfig(_)));
    }

    #[test]
    fn preprocessed_patches_match_per_frame_shape() {
        let prep = VideoPreprocessor::new(tiny_config());
        let frames = vec![frame(4, 4, 128), frame(4, 4, 64), frame(4, 4, 0)];
        let out = prep.preprocess(&frames).unwrap();
        assert_eq!(out.frame_count(), 3);
        let cfg = tiny_config();
        for patches in &out.frames {
            assert_eq!(patches.num_patches, cfg.num_patches());
            assert_eq!(patches.patch_dim, cfg.patch_dim());
        }
    }
}
