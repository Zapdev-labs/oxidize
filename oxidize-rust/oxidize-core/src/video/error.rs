use std::fmt;

/// Errors that can occur when constructing, configuring, or running a video
/// understanding model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoError {
    /// Configuration is invalid (zero sizes, mismatched dims, etc.).
    InvalidConfig(String),
    /// Provided frame dimensions do not match the buffer length.
    InvalidFrame {
        expected_bytes: usize,
        actual_bytes: usize,
        width: usize,
        height: usize,
    },
    /// Number of frames in the input is outside the supported range.
    FrameCountOutOfRange {
        requested: usize,
        min: usize,
        max: usize,
    },
    /// Frame sampling produced an empty or otherwise invalid index list.
    EmptySample,
    /// Decoder could not extract any frames from the source.
    DecoderFailed(String),
    /// Temporal encoder weight shape mismatch.
    WeightShapeMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A generic forward / inference failure.
    InferenceFailed(String),
    /// Wrapped vision error.
    Vision(crate::vision::VisionError),
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid video config: {message}"),
            Self::InvalidFrame {
                expected_bytes,
                actual_bytes,
                width,
                height,
            } => write!(
                f,
                "invalid video frame at {width}x{height}: expected {expected_bytes} bytes, got {actual_bytes}"
            ),
            Self::FrameCountOutOfRange {
                requested,
                min,
                max,
            } => write!(f, "frame count {requested} out of range [{min}, {max}]"),
            Self::EmptySample => write!(f, "frame sampler produced no frames"),
            Self::DecoderFailed(message) => write!(f, "video decoder failed: {message}"),
            Self::WeightShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "weight shape mismatch for {name}: expected {expected}, got {actual}"
            ),
            Self::InferenceFailed(message) => write!(f, "video inference failed: {message}"),
            Self::Vision(err) => write!(f, "vision error: {err}"),
        }
    }
}

impl std::error::Error for VideoError {}

impl From<crate::vision::VisionError> for VideoError {
    fn from(value: crate::vision::VisionError) -> Self {
        Self::Vision(value)
    }
}
