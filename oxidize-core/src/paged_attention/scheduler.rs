//! PagedAttention scheduler: sequence state machine, token budget tracking,
//! and the scheduling loop that moves sequences between queues.
//!
//! This module is split into focused submodules:
//! - [`sequence`]: sequence state, accessors, and status transitions.
//! - [`config`]: scheduler config, step result, and flattened input batch.
//! - [`error`]: scheduler error enum and conversions.
//! - [`core`]: the `Scheduler` struct, scheduling loop, and batch building.
//! - [`lifecycle`]: finishing, removal, draining, and cache invalidation.
//! - [`prefix_cache`]: prefix caching, cache-aware prefill, preemption, COW.

use super::{BlockId, BlockPool, BlockTable};
use crate::model::Token;
use crate::paged_attention::block_pool::{BlockPoolError, compute_block_hash};
use crate::sampling::SamplingConfig;
use std::collections::{HashMap, VecDeque};

mod config;
mod core;
mod error;
mod lifecycle;
mod prefix_cache;
mod sequence;

#[cfg(test)]
mod tests;

pub use config::{InputBatch, SchedulerConfig, SchedulerStepResult};
pub use core::Scheduler;
pub use error::SchedulerError;
pub use sequence::{SeqId, Sequence, SequenceStatus};
