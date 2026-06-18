//! CPU video-classifier training for the `oxidize-train` crate.
//!
//! Turns a directory of short clips + JSON metadata (the `tt-downloader`
//! layout) into a trained model. The pipeline is:
//! `manifest` (discover clips + labels) → `frames` (ffmpeg extract + decode)
//! → `dataset` (materialize patch tensors) → `model` (patch-embed + temporal
//! pool + MLP head) → `train` (AdamW loop + eval + checkpoint).

mod config;
mod dataset;
mod error;
mod frames;
mod manifest;
mod model;
mod prototype;
mod train;

pub use config::{FrameConfig, LabelTask, VideoTrainingConfig};
pub use dataset::{default_model_path, load_dataset, split_indices, VideoDataset};
pub use error::VideoError;
pub use frames::ffmpeg_available;
pub use manifest::{build_manifest, VideoSample};
pub use model::VideoClassifier;
pub use prototype::{build_prototype, PrototypeReport};
pub use train::{save_model, train_video_classifier, SavedModel, VideoTrainingReport};
