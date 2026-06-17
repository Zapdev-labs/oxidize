//! PagedAttention engine for oxidize.
//!
//! Provides block-based KV cache management with on-demand allocation,
//! reference counting for shared blocks, and copy-on-write semantics.
#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod block_pool;
pub mod scheduler;

pub use block_pool::{
    BlockHash, BlockId, BlockPool, BlockPoolConfig, BlockTable, PhysicalBlock, compute_block_hash,
};
pub use scheduler::{
    InputBatch, Scheduler, SchedulerConfig, SchedulerError, SchedulerStepResult, SeqId, Sequence,
    SequenceStatus,
};
