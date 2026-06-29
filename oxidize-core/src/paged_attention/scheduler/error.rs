//! Error handling: the scheduler error enum and conversion traits from the
//! block pool error type.

use super::*;

/// Error type for scheduler operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    BlockPool(BlockPoolError),
    SequenceNotFound { seq_id: SeqId },
    OutOfMemory,
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerError::BlockPool(e) => write!(f, "block pool error: {e}"),
            SchedulerError::SequenceNotFound { seq_id } => {
                write!(f, "sequence {seq_id} not found")
            }
            SchedulerError::OutOfMemory => write!(f, "KV cache exhausted"),
        }
    }
}

impl std::error::Error for SchedulerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SchedulerError::BlockPool(e) => Some(e),
            _ => None,
        }
    }
}

impl From<BlockPoolError> for SchedulerError {
    fn from(value: BlockPoolError) -> Self {
        Self::BlockPool(value)
    }
}
