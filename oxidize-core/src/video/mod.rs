//! CPU-first video understanding support.
//!
//! The video stack intentionally builds on the existing image/vision support:
//! decoded RGB frames are sampled, preprocessed into image patches, encoded
//! frame-by-frame, then passed through a lightweight temporal encoder that runs
//! entirely on CPU.

mod config;
mod decoder;
mod encoder;
mod error;
mod frame_sampler;
mod preprocess;
mod prompt;
mod temporal;

pub use config::{FrameSamplingStrategy, TemporalConfig, TemporalPool, VideoConfig};
pub use decoder::{
    DecodedFrame, RawFrameDecoder, RepetitiveFrameDecoder, ResizingDecoder, VideoDecoder,
    VideoSource,
};
pub use encoder::{VideoEncoder, VideoEncoderWeights, VideoEncoderWorkspace};
pub use error::VideoError;
pub use frame_sampler::{luma_histogram_rgb, sample_indices, sample_indices_adaptive};
pub use preprocess::{VideoFrames, VideoPreprocessor};
pub use prompt::{PromptSegment, VideoPrompt};
pub use temporal::{TemporalLayerWeights, TemporalWeights, TemporalWorkspace, forward_temporal};
