//! Merge two HuggingFace SafeTensors checkpoints with linear or SLERP blending.

#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

pub mod blend;
pub mod index;
pub mod merge;
pub mod recipe;
pub mod writer;

pub use merge::{MergeMethod, MergeOptions, MergeReport, MissingTensorPolicy, merge_models};
pub use recipe::MergeRecipe;
