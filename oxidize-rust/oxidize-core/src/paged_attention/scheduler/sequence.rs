//! Sequence state management: definition, accessors, token tracking, and
//! status transitions for individual generation requests.

use super::*;

/// Unique identifier for a sequence (request).
pub type SeqId = u64;

/// Status of a sequence in the scheduler state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// The sequence is in the waiting queue, not yet started.
    Waiting,
    /// The sequence is actively being processed in a batch.
    Running,
    /// The sequence has completed and its blocks should be reclaimed.
    Finished,
}

/// A single generation request managed by the scheduler.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    pub(super) seq_id: SeqId,
    pub(super) status: SequenceStatus,
    /// Prompt tokens (original input).
    pub(super) prompt_tokens: Vec<Token>,
    /// Tokens generated so far.
    pub(super) generated_tokens: Vec<Token>,
    /// Logical → physical block mapping for this sequence.
    pub(super) block_table: BlockTable,
    /// Arrival order for FCFS scheduling.
    pub(super) arrival_order: usize,
    /// Maximum new tokens to generate.
    pub(super) max_new_tokens: usize,
    /// Stop token (e.g. EOS).
    pub(super) stop_token: Option<Token>,
    /// Sampling configuration.
    pub(super) sampling: SamplingConfig,
    /// Number of prompt tokens already prefilled into the KV cache.
    pub(super) num_prefilled_tokens: usize,
}

impl Sequence {
    /// Create a new sequence in the `Waiting` state.
    ///
    /// The `arrival_order` is assigned automatically by the scheduler, so any
    /// value passed here is overwritten when [`Scheduler::add_sequence`] is called.
    pub fn new(
        seq_id: SeqId,
        prompt_tokens: Vec<Token>,
        block_size: usize,
        max_new_tokens: usize,
        stop_token: Option<Token>,
        sampling: SamplingConfig,
    ) -> Self {
        Self {
            seq_id,
            status: SequenceStatus::Waiting,
            prompt_tokens,
            generated_tokens: Vec::new(),
            block_table: BlockTable::new(block_size),
            arrival_order: 0,
            max_new_tokens,
            stop_token,
            sampling,
            num_prefilled_tokens: 0,
        }
    }

    /// The sequence id.
    pub fn seq_id(&self) -> SeqId {
        self.seq_id
    }

    /// Current status.
    pub fn status(&self) -> SequenceStatus {
        self.status
    }

    /// Set the status (used by the scheduler during state transitions).
    pub(crate) fn set_status(&mut self, status: SequenceStatus) {
        self.status = status;
    }

    /// Total tokens in the sequence (prompt + generated).
    pub fn num_tokens(&self) -> usize {
        self.prompt_tokens
            .len()
            .saturating_add(self.generated_tokens.len())
    }

    /// Number of prompt tokens.
    pub fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    /// Number of generated tokens.
    pub fn generated_len(&self) -> usize {
        self.generated_tokens.len()
    }

    /// Reference to the generated tokens.
    pub fn generated_tokens(&self) -> &[Token] {
        &self.generated_tokens
    }

    /// Reference to the prompt tokens.
    pub fn prompt_tokens(&self) -> &[Token] {
        &self.prompt_tokens
    }

    /// Reference to the block table.
    pub fn block_table(&self) -> &BlockTable {
        &self.block_table
    }

    /// Mutable reference to the block table.
    pub fn block_table_mut(&mut self) -> &mut BlockTable {
        &mut self.block_table
    }

    /// Arrival order for FCFS fairness.
    pub fn arrival_order(&self) -> usize {
        self.arrival_order
    }

    /// Append a generated token.
    pub fn append_token(&mut self, token: Token) {
        self.generated_tokens.push(token);
    }

    /// Check whether the sequence has reached a stop condition.
    pub fn is_finished(&self) -> bool {
        if self.generated_len() >= self.max_new_tokens {
            return true;
        }
        if let Some(stop) = self.stop_token
            && self.generated_tokens.last().copied() == Some(stop)
        {
            return true;
        }
        false
    }

    /// Number of tokens that still need to be prefilled (prompt tokens not yet in KV cache).
    pub fn remaining_prefill_tokens(&self) -> usize {
        self.prompt_tokens
            .len()
            .saturating_sub(self.num_prefilled_tokens)
    }

    /// Increment the count of prefilled tokens.
    pub fn record_prefilled_tokens(&mut self, count: usize) {
        self.num_prefilled_tokens = self
            .num_prefilled_tokens
            .saturating_add(count)
            .min(self.prompt_tokens.len());
    }

    /// Number of decode tokens this sequence needs in the current step (always 1 for autoregressive decode).
    pub fn decode_tokens(&self) -> usize {
        if self.status == SequenceStatus::Running && !self.is_finished() {
            1
        } else {
            0
        }
    }
}
