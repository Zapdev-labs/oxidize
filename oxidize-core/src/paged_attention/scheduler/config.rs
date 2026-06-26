//! Configuration and result types: scheduler config, step result, and the
//! flattened input batch used across the scheduler.

use super::*;

/// Configuration for the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Maximum number of tokens that can be processed in a single batch step.
    pub max_num_batched_tokens: usize,
    /// Default number of tokens per prefill chunk (aligned with block size).
    pub prefill_chunk_size: usize,
    /// Maximum number of sequences that can be in the running state simultaneously.
    pub max_num_running_seqs: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_batched_tokens: 512,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        }
    }
}

/// Result of a single scheduler step: the set of sequences to run.
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerStepResult {
    /// Sequence IDs scheduled for this step.
    pub scheduled_seq_ids: Vec<SeqId>,
    /// Number of prefill tokens scheduled.
    pub prefill_tokens: usize,
    /// Number of decode tokens scheduled.
    pub decode_tokens: usize,
    /// Per-sequence prefill tokens scheduled in this step.
    pub seq_prefill_tokens: HashMap<SeqId, usize>,
    /// Per-sequence decode tokens scheduled in this step.
    pub seq_decode_tokens: HashMap<SeqId, usize>,
}

/// Flattened input batch for a single batched forward pass.
///
/// Multiple sequences (prefill chunks and/or decode tokens) are flattened
/// into a single structure that a batched model forward can consume.
#[derive(Debug, Clone, PartialEq)]
pub struct InputBatch {
    /// Number of sequences in the batch (batch_size).
    pub batch_size: usize,
    /// Sequence IDs in the batch, in scheduling order.
    pub seq_ids: Vec<SeqId>,
    /// Tokens to forward for each sequence this step.
    /// - Prefill: the chunk of prompt tokens processed this step.
    /// - Decode: the single last-generated token from the previous step.
    pub token_ids: Vec<Vec<Token>>,
    /// Position indices within each sequence for the corresponding tokens.
    pub positions: Vec<Vec<usize>>,
    /// Per-sequence physical block tables (logical → physical mapping).
    pub block_tables: Vec<Vec<BlockId>>,
    /// Number of tokens scheduled for each sequence this step.
    pub num_tokens: Vec<usize>,
    /// Total tokens across all sequences (must be ≤ max_num_batched_tokens).
    pub total_tokens: usize,
    /// Whether each sequence is doing prefill (`true`) or decode (`false`).
    pub is_prefill: Vec<bool>,
    /// Per-sequence context length (number of valid tokens up to this step).
    /// Used to mask out unused slots in partially-filled blocks during
    /// paged attention.
    pub context_lens: Vec<usize>,
}
