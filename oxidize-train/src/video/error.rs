use std::path::PathBuf;

use crate::TrainingError;

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("i/o error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse metadata {path}: {source}")]
    Metadata {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("no usable video samples were found under {0}")]
    NoSamples(PathBuf),

    #[error("dataset has only {classes} label class(es); need at least 2 to train")]
    DegenerateLabels { classes: usize },

    #[error("ffmpeg is required for frame extraction but was not found on PATH")]
    FfmpegMissing,

    #[error("ffmpeg failed for {path}: {message}")]
    Ffmpeg { path: PathBuf, message: String },

    #[error("failed to decode frame {path}: {source}")]
    Decode {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },

    #[error("frame extraction produced no frames for {0}")]
    NoFrames(PathBuf),

    #[error(transparent)]
    Training(#[from] TrainingError),

    #[error("invalid configuration: {0}")]
    Config(String),
}
