use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::VideoError;

/// What the classifier learns to predict from a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelTask {
    Creator,
    Virality,
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
    pub num_frames: usize,
    /// Frame width in pixels (9 for 9:16 portrait).
    #[serde(default)]
    pub frame_width: usize,
    /// Frame height in pixels (16 for 9:16 portrait).
    #[serde(default)]
    pub frame_height: usize,
    /// Legacy square edge length; used when width/height are unset.
    #[serde(default)]
    pub frame_size: usize,
    pub patch_size: usize,
}

impl FrameConfig {
    /// Resolve width/height from legacy `frame_size` when needed.
    pub fn resolve_dimensions(mut self) -> Self {
        if self.frame_width > 0 && self.frame_height > 0 {
            return self;
        }
        if self.frame_size > 0 {
            self.frame_width = self.frame_size;
            self.frame_height = self.frame_size;
            return self;
        }
        if self.frame_width > 0 {
            self.frame_height = self.frame_width;
        } else if self.frame_height > 0 {
            self.frame_width = self.frame_height;
        } else {
            self.frame_width = 64;
            self.frame_height = 64;
        }
        self
    }

    /// 9:16 portrait TikTok layout (`width` × `width*16/9`).
    pub fn portrait_9_16(width: usize, patch_size: usize, num_frames: usize) -> Self {
        Self {
            num_frames,
            frame_width: width,
            frame_height: width * 16 / 9,
            frame_size: 0,
            patch_size,
        }
        .resolve_dimensions()
    }

    pub fn patches_x(self) -> usize {
        self.frame_width / self.patch_size
    }

    pub fn patches_y(self) -> usize {
        self.frame_height / self.patch_size
    }

    pub fn num_patches(self) -> usize {
        self.patches_x() * self.patches_y()
    }

    pub fn patch_dim(self) -> usize {
        self.patch_size * self.patch_size * 3
    }

    pub fn tokens_per_clip(self) -> usize {
        self.num_frames * self.num_patches()
    }

    pub fn aspect_label(self) -> String {
        format!("{}x{}", self.frame_width, self.frame_height)
    }

    pub fn validate(self) -> Result<Self, VideoError> {
        let cfg = self.resolve_dimensions();
        if cfg.num_frames == 0 {
            return Err(VideoError::Config("num_frames must be > 0".into()));
        }
        if cfg.frame_width == 0 || cfg.frame_height == 0 || cfg.patch_size == 0 {
            return Err(VideoError::Config(
                "frame_width, frame_height, patch_size must be > 0".into(),
            ));
        }
        if !cfg.frame_width.is_multiple_of(cfg.patch_size)
            || !cfg.frame_height.is_multiple_of(cfg.patch_size)
        {
            return Err(VideoError::Config(format!(
                "frame {}x{} must be divisible by patch_size ({})",
                cfg.frame_width, cfg.frame_height, cfg.patch_size
            )));
        }
        Ok(cfg)
    }
}

impl Default for FrameConfig {
    fn default() -> Self {
        Self::portrait_9_16(72, 8, 8)
    }
}

/// Full configuration for a video-classifier training run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoTrainingConfig {
    pub data_root: PathBuf,
    pub cache_dir: PathBuf,
    pub task: LabelTask,
    pub frames: FrameConfig,
    pub embed_dim: usize,
    pub hidden_size: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub seed: u64,
    pub val_split: f32,
    pub buckets: usize,
    pub max_videos: Option<usize>,
    pub balance: bool,
    pub exclude: Vec<String>,
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
