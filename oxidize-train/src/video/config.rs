use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::VideoError;

/// What the classifier learns to predict from a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelTask {
    /// Predict which creator/account a clip came from.
    Creator,
    /// Predict a view-count bucket (quantile) — "how viral".
    Virality,
    /// Predict a like/view engagement-ratio bucket.
    Engagement,
}

impl LabelTask {
    pub fn as_str(self) -> &'static str {
        match self {
            LabelTask::Creator => "creator",
            LabelTask::Virality => "virality",
            LabelTask::Engagement => "engagement",
        }
    }
}

/// How each clip is turned into a fixed-size tensor of patch tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameConfig {
    /// Number of frames sampled evenly across the clip.
    pub num_frames: usize,
    /// Square edge length each frame is resized/cropped to (pixels).
    pub frame_size: usize,
    /// Square edge length of a patch (pixels). Must divide `frame_size`.
    pub patch_size: usize,
}

impl FrameConfig {
    pub fn patches_per_side(self) -> usize {
        self.frame_size / self.patch_size
    }

    pub fn num_patches(self) -> usize {
        let side = self.patches_per_side();
        side * side
    }

    /// Floats per patch token: `patch_size^2 * 3` (RGB).
    pub fn patch_dim(self) -> usize {
        self.patch_size * self.patch_size * 3
    }

    /// Patch tokens per clip: `num_frames * num_patches`.
    pub fn tokens_per_clip(self) -> usize {
        self.num_frames * self.num_patches()
    }

    pub fn validate(self) -> Result<(), VideoError> {
        if self.num_frames == 0 {
            return Err(VideoError::Config("num_frames must be > 0".into()));
        }
        if self.frame_size == 0 || self.patch_size == 0 {
            return Err(VideoError::Config(
                "frame_size and patch_size must be > 0".into(),
            ));
        }
        if !self.frame_size.is_multiple_of(self.patch_size) {
            return Err(VideoError::Config(format!(
                "frame_size ({}) must be divisible by patch_size ({})",
                self.frame_size, self.patch_size
            )));
        }
        Ok(())
    }
}

impl Default for FrameConfig {
    fn default() -> Self {
        Self {
            num_frames: 8,
            frame_size: 64,
            patch_size: 16,
        }
    }
}

/// Full configuration for a video-classifier training run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoTrainingConfig {
    /// Root that holds `videos/` + `*_metadata.json` (the tt-downloader layout).
    pub data_root: PathBuf,
    /// Where extracted frames are cached (defaults to `<data_root>/.oxidize-frames`).
    pub cache_dir: PathBuf,
    pub task: LabelTask,
    pub frames: FrameConfig,
    /// Width of the learned patch embedding.
    pub embed_dim: usize,
    /// Hidden width of the classifier head.
    pub hidden_size: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub seed: u64,
    /// Fraction of clips held out for validation (0.0..1.0).
    pub val_split: f32,
    /// Number of quantile buckets for Virality/Engagement tasks.
    pub buckets: usize,
    /// Optional cap on number of clips (handy for quick smoke runs).
    pub max_videos: Option<usize>,
    /// Downsample every class to the size of the smallest one before training.
    pub balance: bool,
    /// Creator usernames to drop before training (case-insensitive).
    pub exclude: Vec<String>,
    /// Merge duplicate accounts (`sanymaaa` → `sanymaa`).
    pub merge_aliases: bool,
}

impl VideoTrainingConfig {
    pub fn validate(&self) -> Result<(), VideoError> {
        self.frames.validate()?;
        if self.embed_dim == 0 || self.hidden_size == 0 {
            return Err(VideoError::Config(
                "embed_dim and hidden_size must be > 0".into(),
            ));
        }
        if self.batch_size == 0 {
            return Err(VideoError::Config("batch_size must be > 0".into()));
        }
        if !(0.0..1.0).contains(&self.val_split) {
            return Err(VideoError::Config(
                "val_split must be in [0.0, 1.0)".into(),
            ));
        }
        if matches!(self.task, LabelTask::Virality | LabelTask::Engagement) && self.buckets < 2 {
            return Err(VideoError::Config(
                "buckets must be >= 2 for virality/engagement tasks".into(),
            ));
        }
        Ok(())
    }
}

impl Default for VideoTrainingConfig {
    fn default() -> Self {
        Self {
            data_root: PathBuf::from("."),
            cache_dir: PathBuf::new(),
            task: LabelTask::Virality,
            frames: FrameConfig::default(),
            embed_dim: 64,
            hidden_size: 128,
            epochs: 40,
            batch_size: 16,
            learning_rate: 5e-4,
            weight_decay: 0.05,
            seed: 42,
            val_split: 0.15,
            buckets: 3,
            max_videos: None,
            balance: false,
            exclude: vec!["cellow111".into()],
            merge_aliases: true,
        }
    }
}
