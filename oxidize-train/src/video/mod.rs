//! CPU video training and generation for the `oxidize-train` crate.
//!
//! Pipeline:
//! - `manifest` + `frames` — discover clips, extract frames via ffmpeg
//! - `generator` — autoregressive next-frame model (train + sample)
//! - `model` + `train` — optional clip classifier (virality/engagement)

mod config;
mod dataset;
mod error;
mod frames;
mod generator;
mod manifest;
mod model;
mod train;

pub use config::{FrameConfig, LabelTask, VideoTrainingConfig};
pub use dataset::{default_model_path, filter_samples, load_dataset, split_indices, VideoDataset};
pub use error::VideoError;
pub use frames::ffmpeg_available;
pub use generator::{
    default_generator_path, generate_video, load_gen_dataset, load_generator, save_generator,
    train_generator, GenDataset, GenTrainingConfig, GenTrainingReport, GenerateReport,
    SavedGenerator, VideoGenerator,
};
pub use manifest::{build_manifest, VideoSample};
pub use model::VideoClassifier;
pub use train::{save_model, train_video_classifier, SavedModel, VideoTrainingReport};
