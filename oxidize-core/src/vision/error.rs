use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisionError {
    InvalidConfig(String),
    InvalidImageDimensions {
        expected: usize,
        actual: usize,
    },
    PatchShapeMismatch {
        expected: usize,
        actual: usize,
    },
    WeightShapeMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidPrompt(String),
    InferenceFailed(String),
    UnsupportedFormat(String),
}

impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid vision config: {message}"),
            Self::InvalidImageDimensions { expected, actual } => write!(
                f,
                "invalid image dimensions: expected {expected} bytes, got {actual}"
            ),
            Self::PatchShapeMismatch { expected, actual } => {
                write!(f, "patch shape mismatch: expected {expected}, got {actual}")
            }
            Self::WeightShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "weight shape mismatch for {name}: expected {expected}, got {actual}"
            ),
            Self::InvalidPrompt(message) => write!(f, "invalid multimodal prompt: {message}"),
            Self::InferenceFailed(message) => write!(f, "vision inference failed: {message}"),
            Self::UnsupportedFormat(format) => write!(f, "unsupported image format: {format}"),
        }
    }
}

impl std::error::Error for VisionError {}
