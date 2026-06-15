use super::error::VideoError;
use crate::vision::VisionConfig;

/// Strategy for selecting which frames of an input video to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameSamplingStrategy {
    /// Pick `target_frames` indices evenly spaced across `[0, total_frames)`.
    #[default]
    Uniform,
    /// Keep every `stride`-th frame (clipped to `target_frames` if available).
    /// Stride 0 is treated as 1 (i.e. dense sampling).
    Dense { stride: usize },
    /// Pick frames that maximize pairwise histogram distance — a cheap stand-in
    /// for keyframe detection. Falls back to uniform when fewer than the
    /// target number of candidate frames are available.
    Adaptive,
}

/// Configuration for the temporal encoder that runs on top of the
/// per-frame vision encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalConfig {
    /// Hidden size fed into the temporal encoder (must equal the vision
    /// encoder's `projection_dim`).
    pub hidden_size: usize,
    /// Number of causal self-attention layers over the time axis.
    pub num_layers: usize,
    /// Number of attention heads per layer.
    pub num_heads: usize,
    /// Feed-forward intermediate size.
    pub intermediate_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Maximum number of frames the model can attend to at once. Frames beyond
    /// this limit are dropped (sampling strategies honor this bound).
    pub max_frames: usize,
    /// RoPE base frequency used by the temporal attention (1D, time axis).
    pub rope_theta: f32,
    /// Whether to use a learnable `temporal_cls` token prepended to the time
    /// axis (similar to BERT `[CLS]` / ViT `class_token`). When false the
    /// temporal encoder pools the time axis by simple mean.
    pub use_cls_token: bool,
    /// Dropout-style stochastic depth probability for temporal layers. 0.0 =
    /// no stochastic depth. Stored but not used at inference time — kept so
    /// weights from training checkpoints can be loaded without shape
    /// changes.
    pub layer_dropout: f32,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_layers: 2,
            num_heads: 8,
            intermediate_size: 4096,
            rms_norm_eps: 1e-5,
            max_frames: 32,
            rope_theta: 10000.0,
            use_cls_token: true,
            layer_dropout: 0.0,
        }
    }
}

impl TemporalConfig {
    pub fn head_dim(&self) -> usize {
        if self.num_heads == 0 {
            return 0;
        }
        self.hidden_size / self.num_heads
    }

    pub fn validate(&self) -> Result<(), VideoError> {
        if self.hidden_size == 0 {
            return invalid_config("hidden_size must be non-zero");
        }
        if self.num_heads == 0 {
            return invalid_config("num_heads must be non-zero");
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return invalid_config("hidden_size must be divisible by num_heads");
        }
        if self.num_layers == 0 {
            return invalid_config("num_layers must be non-zero");
        }
        if self.intermediate_size == 0 {
            return invalid_config("intermediate_size must be non-zero");
        }
        if self.max_frames == 0 {
            return invalid_config("max_frames must be non-zero");
        }
        if self.rms_norm_eps <= 0.0 {
            return invalid_config("rms_norm_eps must be positive");
        }
        if self.rope_theta <= 0.0 {
            return invalid_config("rope_theta must be positive");
        }
        Ok(())
    }
}

/// How pooled per-frame video tokens should be aggregated into a single
/// representation per frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TemporalPool {
    /// Mean of all patch embeddings per frame.
    #[default]
    Mean,
    /// First token (the vision encoder's class token) per frame.
    ClsToken,
    /// Last token of each frame's patch sequence.
    LastToken,
}

/// Top-level video model configuration. Holds both the per-frame vision
/// configuration and the temporal stack configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoConfig {
    pub vision: VisionConfig,
    pub temporal: TemporalConfig,
    pub sampling: FrameSamplingStrategy,
    /// Target number of frames produced by the sampler. The decoder is
    /// expected to provide at least this many frames (or `total_frames` if
    /// less). Must be `<= temporal.max_frames`.
    pub target_frames: usize,
    /// Output projection dimension (e.g. size of the LLM's hidden dim). The
    /// VideoEncoder projects temporal hidden states into this space so video
    /// tokens can be dropped into an existing language model's embedding
    /// table. When 0 the temporal hidden size is reused.
    pub llm_hidden_size: usize,
    pub pool: TemporalPool,
    /// Prompt token IDs reserved for `<video>` / `</video>` wrappers. When
    /// non-empty the encoder wraps the projected video embeddings in these
    /// tokens' embeddings to mark the video region in the prompt.
    pub video_start_token_id: u32,
    pub video_end_token_id: u32,
}

impl Default for VideoConfig {
    fn default() -> Self {
        let temporal = TemporalConfig {
            hidden_size: VisionConfig::clip_large().projection_dim,
            ..Default::default()
        };
        Self {
            vision: VisionConfig::clip_large(),
            temporal,
            sampling: FrameSamplingStrategy::Uniform,
            target_frames: 8,
            llm_hidden_size: 0,
            pool: TemporalPool::Mean,
            video_start_token_id: 0,
            video_end_token_id: 0,
        }
    }
}

impl VideoConfig {
    /// Reasonable defaults for a small / fast CPU video model.
    pub fn cpu_small() -> Self {
        let vision = VisionConfig::clip_base();
        let temporal = TemporalConfig {
            hidden_size: vision.projection_dim,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 2048,
            max_frames: 16,
            ..Default::default()
        };
        Self {
            vision,
            temporal,
            sampling: FrameSamplingStrategy::Uniform,
            target_frames: 8,
            llm_hidden_size: 0,
            pool: TemporalPool::Mean,
            video_start_token_id: 0,
            video_end_token_id: 0,
        }
    }

    pub fn validate(&self) -> Result<(), VideoError> {
        self.vision
            .validate()
            .map_err(|e| VideoError::InvalidConfig(format!("vision: {e}")))?;
        if self.vision.projection_dim != self.temporal.hidden_size {
            return invalid_config(&format!(
                "temporal.hidden_size ({}) must equal vision.projection_dim ({})",
                self.temporal.hidden_size, self.vision.projection_dim
            ));
        }
        self.temporal.validate()?;
        if self.target_frames == 0 {
            return invalid_config("target_frames must be non-zero");
        }
        if self.target_frames > self.temporal.max_frames {
            return invalid_config(&format!(
                "target_frames ({}) exceeds temporal.max_frames ({})",
                self.target_frames, self.temporal.max_frames
            ));
        }
        Ok(())
    }
}

fn invalid_config<T>(message: &str) -> Result<T, VideoError> {
    Err(VideoError::InvalidConfig(message.into()))
}
