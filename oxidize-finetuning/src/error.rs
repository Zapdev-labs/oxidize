use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum FinetuneError {
    EmptyDataset,
    InvalidSequence { len: usize, max: usize },
    Adapter(String),
    Model(String),
}

impl fmt::Display for FinetuneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => write!(f, "SFT dataset is empty"),
            Self::InvalidSequence { len, max } => {
                write!(f, "sequence length {len} exceeds max_seq_len {max}")
            }
            Self::Adapter(msg) => write!(f, "adapter error: {msg}"),
            Self::Model(msg) => write!(f, "model error: {msg}"),
        }
    }
}

impl Error for FinetuneError {}

pub type Result<T> = std::result::Result<T, FinetuneError>;
