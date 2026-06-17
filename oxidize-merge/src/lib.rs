//! Merge two HuggingFace SafeTensors checkpoints with linear or SLERP blending.

pub mod blend;
pub mod index;
pub mod merge;
pub mod recipe;
pub mod writer;

pub use merge::{MergeMethod, MergeOptions, MergeReport, MissingTensorPolicy, merge_models};
pub use recipe::MergeRecipe;
