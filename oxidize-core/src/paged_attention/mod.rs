//! PagedAttention engine for oxidize.
//!
//! Provides block-based KV cache management with on-demand allocation,
//! reference counting for shared blocks, and copy-on-write semantics.

pub mod block_pool;
pub mod scheduler;

pub use block_pool::{BlockId, BlockPool, BlockPoolConfig, BlockTable, PhysicalBlock};
pub use scheduler::{
    InputBatch, Scheduler, SchedulerConfig, SchedulerError, SchedulerStepResult, SeqId,
    Sequence, SequenceStatus,
};
