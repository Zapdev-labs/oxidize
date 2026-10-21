//! Vision preprocessing and embedding helpers for multimodal inference.
//!
//! RGB/RGBA pixels are resized, normalized, patchified, embedded, and projected
//! into the language model hidden space.

mod config;
mod encoder;
mod error;
mod preprocess;
mod prompt;

pub use config::{VisionConfig, count_image_tokens};
pub use encoder::VisionEncoder;
pub use error::VisionError;
pub use preprocess::{ImagePatches, ImagePreprocessor};
pub use prompt::MultimodalPrompt;

pub fn decode_image_bytes(_data: &[u8]) -> Result<(Vec<u8>, usize, usize), VisionError> {
    Err(VisionError::UnsupportedFormat(
        "encoded image decoding is not implemented; pass decoded RGB or RGBA pixels".into(),
    ))
}
