//! PagedAttention scheduler: sequence state machine, token budget tracking,
//! and the scheduling loop that moves sequences between queues.

use super::{BlockId, BlockPool, BlockTable};
use crate::model::Token;
use crate::paged_attention::block_pool::BlockPoolError;
use crate::sampling::SamplingConfig;
use std::collections::{HashMap, VecDeque};

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
    seq_id: SeqId,
    status: SequenceStatus,
    /// Prompt tokens (original input).
    prompt_tokens: Vec<Token>,
    /// Tokens generated so far.
    generated_tokens: Vec<Token>,
    /// Logical → physical block mapping for this sequence.
    block_table: BlockTable,
    /// Arrival order for FCFS scheduling.
    arrival_order: usize,
    /// Maximum new tokens to generate.
    max_new_tokens: usize,
    /// Stop token (e.g. EOS).
    stop_token: Option<Token>,
    /// Sampling configuration.
    sampling: SamplingConfig,
    /// Number of prompt tokens already prefilled into the KV cache.
    num_prefilled_tokens: usize,
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
        self.prompt_tokens.len().saturating_add(self.generated_tokens.len())
    }

    /// Number of prompt tokens.
    pub fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    /// Number of generated tokens.
    pub fn generated_len(&self) -> usize {
        self.generated_tokens.len()
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
}

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

/// The PagedAttention scheduler manages the waiting and running queues,
/// enforces the token budget per step, and transitions sequences through
/// the WAITING → RUNNING → FINISHED lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub struct Scheduler {
    config: SchedulerConfig,
    block_pool: BlockPool,
    /// All sequences keyed by id.
    sequences: HashMap<SeqId, Sequence>,
    /// Waiting queue: sequences that have not yet started prefill.
    waiting: VecDeque<SeqId>,
    /// Running queue: sequences currently in a batch.
    running: Vec<SeqId>,
    /// Monotonically increasing arrival counter for FCFS ordering.
    next_arrival_order: usize,
}

impl Scheduler {
    /// Create a new scheduler with the given configuration and block pool.
    pub fn new(config: SchedulerConfig, block_pool: BlockPool) -> Self {
        Self {
            config,
            block_pool,
            sequences: HashMap::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            next_arrival_order: 0,
        }
    }

    /// Return a reference to the block pool.
    pub fn block_pool(&self) -> &BlockPool {
        &self.block_pool
    }

    /// Return a mutable reference to the block pool.
    pub fn block_pool_mut(&mut self) -> &mut BlockPool {
        &mut self.block_pool
    }

    /// Return the scheduler configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Add a new sequence to the scheduler (enters the waiting queue).
    pub fn add_sequence(&mut self, mut sequence: Sequence) -> Result<(), SchedulerError> {
        let seq_id = sequence.seq_id();
        sequence.arrival_order = self.next_arrival_order;
        self.next_arrival_order += 1;
        self.sequences.insert(seq_id, sequence);
        self.waiting.push_back(seq_id);
        Ok(())
    }

    /// Return the number of sequences in the waiting queue.
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Return the number of sequences in the running queue.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Return the total number of sequences managed.
    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    /// Get a reference to a sequence by id.
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.sequences.get(&seq_id)
    }

    /// Get a mutable reference to a sequence by id.
    pub fn get_sequence_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        self.sequences.get_mut(&seq_id)
    }

    /// Allocate blocks and advance token count for a prefill chunk of `chunk_size`
    /// on the given sequence.  Does not borrow `self` so it can be called while
    /// `self.sequences` is already mutably borrowed.
    fn apply_prefill_chunk(
        block_pool: &mut BlockPool,
        seq: &mut Sequence,
        chunk_size: usize,
    ) -> Result<(), SchedulerError> {
        let blocks_needed = seq.block_table.blocks_needed_for_tokens(chunk_size);
        if blocks_needed > 0 {
            let physical_blocks = block_pool.allocate_blocks(blocks_needed)?;
            for block_id in physical_blocks {
                seq.block_table.append_block(block_id);
            }
        }
        for _ in 0..chunk_size {
            let _ = seq.block_table.append_token();
        }
        seq.record_prefilled_tokens(chunk_size);
        Ok(())
    }

    /// Perform one scheduler step: select sequences to run, allocate blocks,
    /// enforce token budget, and transition states.
    ///
    /// After the caller runs the forward pass and samples tokens, it should
    /// call [`Self::postprocess_step`] to append tokens, detect finished
    /// sequences, and reclaim blocks.
    pub fn step(&mut self) -> Result<SchedulerStepResult, SchedulerError> {
        let mut budget = self.config.max_num_batched_tokens;
        let mut scheduled = Vec::new();
        let mut prefill_tokens = 0usize;
        let mut decode_tokens = 0usize;
        let mut seq_prefill_tokens: HashMap<SeqId, usize> = HashMap::new();
        let mut seq_decode_tokens: HashMap<SeqId, usize> = HashMap::new();

        let running_ids: Vec<SeqId> = self.running.clone();

        // --- Phase 1: Decode for fully-prefilled running sequences ---
        // Decode gets highest priority. Each needs exactly 1 token.
        for &seq_id in &running_ids {
            let seq = self
                .sequences
                .get_mut(&seq_id)
                .ok_or(SchedulerError::SequenceNotFound { seq_id })?;

            if seq.is_finished() || seq.remaining_prefill_tokens() > 0 {
                continue;
            }

            if budget == 0 {
                break;
            }

            // Decode needs 1 token and may need a new block.
            let needs_block = seq.block_table.append_token();
            if needs_block {
                let physical_block = self.block_pool.allocate_block()?;
                seq.block_table.append_block(physical_block);
            }

            scheduled.push(seq_id);
            budget -= 1;
            decode_tokens += 1;
            seq_decode_tokens.insert(seq_id, 1);
        }

        // --- Phase 2: Continue prefill for partially-prefilled running sequences ---
        for &seq_id in &running_ids {
            let seq = self
                .sequences
                .get_mut(&seq_id)
                .ok_or(SchedulerError::SequenceNotFound { seq_id })?;

            if seq.is_finished() || seq.remaining_prefill_tokens() == 0 {
                continue;
            }

            // Chunk size capped by both prefill_chunk_size and remaining budget.
            let chunk_size = seq
                .remaining_prefill_tokens()
                .min(self.config.prefill_chunk_size)
                .min(budget);
            if chunk_size == 0 {
                continue;
            }

            Self::apply_prefill_chunk(&mut self.block_pool, seq, chunk_size)?;

            scheduled.push(seq_id);
            budget -= chunk_size;
            prefill_tokens += chunk_size;
            *seq_prefill_tokens.entry(seq_id).or_insert(0) += chunk_size;
        }

        // --- Phase 3: Schedule prefill chunks from waiting queue ---
        // FCFS order. Each prefill request gets at most prefill_chunk_size tokens.
        let mut still_waiting = VecDeque::new();
        let mut running_count = scheduled.len();
        while let Some(seq_id) = self.waiting.pop_front() {
            let seq = self
                .sequences
                .get_mut(&seq_id)
                .ok_or(SchedulerError::SequenceNotFound { seq_id })?;

            if seq.status != SequenceStatus::Waiting {
                still_waiting.push_back(seq_id);
                continue;
            }

            if running_count >= self.config.max_num_running_seqs {
                still_waiting.push_back(seq_id);
                continue;
            }

            let remaining_prefill = seq.remaining_prefill_tokens();
            if remaining_prefill == 0 {
                seq.set_status(SequenceStatus::Running);
                scheduled.push(seq_id);
                running_count += 1;
                continue;
            }

            // Chunk size capped by both prefill_chunk_size and remaining budget.
            let chunk_size = remaining_prefill
                .min(self.config.prefill_chunk_size)
                .min(budget);
            if chunk_size == 0 {
                still_waiting.push_back(seq_id);
                continue;
            }

            Self::apply_prefill_chunk(&mut self.block_pool, seq, chunk_size)?;
            seq.set_status(SequenceStatus::Running);

            scheduled.push(seq_id);
            running_count += 1;
            budget -= chunk_size;
            prefill_tokens += chunk_size;
            *seq_prefill_tokens.entry(seq_id).or_insert(0) += chunk_size;
        }

        self.waiting = still_waiting;

        // Every sequence in `scheduled` is Running by construction.
        self.running = scheduled.clone();

        Ok(SchedulerStepResult {
            scheduled_seq_ids: scheduled,
            prefill_tokens,
            decode_tokens,
            seq_prefill_tokens,
            seq_decode_tokens,
        })
    }

    /// Build an [`InputBatch`] from the result of the most recent [`Self::step`].
    ///
    /// This flattens all scheduled sequences into a single structure with
    /// `batch_size > 1` when multiple sequences are in the batch.
    pub fn build_input_batch(&self, step_result: &SchedulerStepResult) -> InputBatch {
        let batch_size = step_result.scheduled_seq_ids.len();
        let mut seq_ids = Vec::with_capacity(batch_size);
        let mut token_ids: Vec<Vec<Token>> = Vec::with_capacity(batch_size);
        let mut positions: Vec<Vec<usize>> = Vec::with_capacity(batch_size);
        let mut block_tables: Vec<Vec<BlockId>> = Vec::with_capacity(batch_size);
        let mut num_tokens = Vec::with_capacity(batch_size);
        let mut is_prefill = Vec::with_capacity(batch_size);
        let mut total_tokens = 0usize;

        for &seq_id in &step_result.scheduled_seq_ids {
            let seq = match self.sequences.get(&seq_id) {
                Some(s) => s,
                None => continue,
            };

            // Determine whether this sequence received prefill or decode tokens.
            let prefill_count = step_result.seq_prefill_tokens.get(&seq_id).copied().unwrap_or(0);
            let decode_count = step_result.seq_decode_tokens.get(&seq_id).copied().unwrap_or(0);

            if prefill_count > 0 {
                // Prefill sequence: tokens are from the prompt.
                let start = seq.num_prefilled_tokens.saturating_sub(prefill_count);
                let end = start + prefill_count;
                let chunk_tokens: Vec<Token> = seq.prompt_tokens[start..end].to_vec();
                let pos: Vec<usize> = (start..end).collect();
                let physical: Vec<BlockId> = seq.block_table.physical_blocks().to_vec();

                seq_ids.push(seq_id);
                token_ids.push(chunk_tokens);
                positions.push(pos);
                block_tables.push(physical);
                num_tokens.push(prefill_count);
                is_prefill.push(true);
                total_tokens += prefill_count;
            } else if decode_count > 0 && !seq.is_finished() {
                // Decode sequence: single token.
                let decode_pos = seq.num_tokens().saturating_sub(1);
                let decode_token = seq.generated_tokens.last().copied();
                let physical: Vec<BlockId> = seq.block_table.physical_blocks().to_vec();

                seq_ids.push(seq_id);
                token_ids.push(decode_token.into_iter().collect());
                positions.push(vec![decode_pos]);
                block_tables.push(physical);
                num_tokens.push(1);
                is_prefill.push(false);
                total_tokens += 1;
            }
        }

        InputBatch {
            batch_size: seq_ids.len(),
            seq_ids,
            token_ids,
            positions,
            block_tables,
            num_tokens,
            total_tokens,
            is_prefill,
        }
    }

    /// Post-process a scheduler step: append sampled tokens, detect finished
    /// sequences, and reclaim blocks for finished sequences.
    ///
    /// `sampled_tokens` maps `seq_id -> next_token` for every sequence that
    /// was scheduled in the most recent step.
    pub fn postprocess_step(
        &mut self,
        sampled_tokens: &HashMap<SeqId, Token>,
    ) -> Result<(), SchedulerError> {
        // Append tokens and check finish conditions.
        for (&seq_id, &token) in sampled_tokens {
            let Some(seq) = self.sequences.get_mut(&seq_id) else {
                continue;
            };
            if seq.status != SequenceStatus::Running {
                continue;
            }
            seq.append_token(token);
        }

        // Reclaim blocks for finished sequences.
        let mut finished_ids = Vec::new();
        for &seq_id in &self.running {
            let Some(seq) = self.sequences.get(&seq_id) else {
                continue;
            };
            if seq.is_finished() {
                finished_ids.push(seq_id);
            }
        }

        for seq_id in finished_ids {
            self.finish_sequence(seq_id)?;
        }

        Ok(())
    }

    /// Mark a sequence as finished and reclaim its blocks.
    fn finish_sequence(&mut self, seq_id: SeqId) -> Result<(), SchedulerError> {
        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(SchedulerError::SequenceNotFound { seq_id })?;
        seq.set_status(SequenceStatus::Finished);
        let physical_blocks: Vec<BlockId> = seq.block_table.physical_blocks().to_vec();
        for block_id in physical_blocks {
            self.block_pool.dec_ref(block_id)?;
        }

        // Remove from running queue.
        self.running.retain(|&id| id != seq_id);
        Ok(())
    }

    /// Remove a finished sequence from the scheduler entirely.
    pub fn remove_sequence(&mut self, seq_id: SeqId) -> Result<(), SchedulerError> {
        let seq = self
            .sequences
            .remove(&seq_id)
            .ok_or(SchedulerError::SequenceNotFound { seq_id })?;

        // If the sequence was still running (not yet finished), free its blocks.
        if seq.status == SequenceStatus::Running {
            for &block_id in seq.block_table.physical_blocks() {
                self.block_pool.dec_ref(block_id)?;
            }
            self.running.retain(|&id| id != seq_id);
        }

        self.waiting.retain(|&id| id != seq_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::BlockPoolConfig;

    fn default_pool(num_blocks: usize) -> BlockPool {
        BlockPool::new(BlockPoolConfig {
            block_size: 16,
            num_blocks,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            dtype: crate::tensor::DType::F32,
        })
    }

    fn make_seq(seq_id: SeqId, prompt_len: usize, max_new: usize) -> Sequence {
        Sequence::new(
            seq_id,
            vec![1u32; prompt_len],
            16,
            max_new,
            Some(0u32), // EOS token = 0
            SamplingConfig::default(),
        )
    }

    #[test]
    fn scheduler_maintains_waiting_and_running_queues() {
        let pool = default_pool(10);
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config, pool);

        let seq = make_seq(1, 4, 10);
        scheduler.add_sequence(seq).unwrap();

        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.sequence_count(), 1);

        let result = scheduler.step().unwrap();
        assert!(result.scheduled_seq_ids.contains(&1));
        assert_eq!(result.prefill_tokens, 4);
        assert_eq!(result.decode_tokens, 0);

        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 1);
    }

    #[test]
    fn sequences_transition_waiting_to_running_to_finished() {
        let pool = default_pool(10);
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config, pool);

        let seq = make_seq(1, 2, 3);
        scheduler.add_sequence(seq).unwrap();

        // Step 1: prefill
        let step1 = scheduler.step().unwrap();
        assert_eq!(step1.prefill_tokens, 2);
        assert_eq!(step1.decode_tokens, 0);
        assert_eq!(scheduler.get_sequence(1).unwrap().status(), SequenceStatus::Running);

        // Step 2: decode
        let mut sampled = HashMap::new();
        sampled.insert(1, 42u32);
        scheduler.postprocess_step(&sampled).unwrap();

        let step2 = scheduler.step().unwrap();
        assert_eq!(step2.decode_tokens, 1);
        assert_eq!(step2.prefill_tokens, 0);

        // Step 3: decode
        let mut sampled = HashMap::new();
        sampled.insert(1, 43u32);
        scheduler.postprocess_step(&sampled).unwrap();

        let step3 = scheduler.step().unwrap();
        assert_eq!(step3.decode_tokens, 1);

        // Step 4: finish (max_new_tokens = 3 reached)
        let mut sampled = HashMap::new();
        sampled.insert(1, 44u32);
        scheduler.postprocess_step(&sampled).unwrap();

        assert_eq!(scheduler.get_sequence(1).unwrap().status(), SequenceStatus::Finished);
        assert_eq!(scheduler.running_count(), 0);
    }

    #[test]
    fn token_budget_enforced_per_scheduler_step() {
        let pool = default_pool(10);
        let config = SchedulerConfig {
            max_num_batched_tokens: 8,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        // Two waiting requests with 6-token prompts each.
        scheduler.add_sequence(make_seq(1, 6, 5)).unwrap();
        scheduler.add_sequence(make_seq(2, 6, 5)).unwrap();

        let result = scheduler.step().unwrap();
        // Budget = 8. First request takes 6 prefill tokens. Second request gets
        // the remaining 2 tokens (chunk split to fit budget). Total = 8.
        assert_eq!(result.prefill_tokens, 8);
        assert_eq!(result.scheduled_seq_ids, vec![1, 2]);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.get_sequence(2).unwrap().num_prefilled_tokens, 2);
    }

    #[test]
    fn decode_requests_scheduled_before_prefill_chunks() {
        let pool = default_pool(10);
        let config = SchedulerConfig {
            max_num_batched_tokens: 10,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        // Add first sequence and prefill it.
        scheduler.add_sequence(make_seq(1, 2, 5)).unwrap();
        let step1 = scheduler.step().unwrap();
        assert_eq!(step1.prefill_tokens, 2);
        assert!(step1.scheduled_seq_ids.contains(&1));

        // Add second sequence while seq 1 is already running.
        scheduler.add_sequence(make_seq(2, 8, 5)).unwrap();

        // Simulate decode token for seq 1.
        let mut sampled = HashMap::new();
        sampled.insert(1, 42u32);
        scheduler.postprocess_step(&sampled).unwrap();

        // Second step: seq 1 needs decode (1 token), seq 2 needs prefill (8 tokens).
        // Budget = 10. Decode gets priority: 1 decode + up to 9 prefill.
        // Seq 2 prefill is 8 tokens, so both fit.
        let step2 = scheduler.step().unwrap();
        assert_eq!(step2.decode_tokens, 1);
        assert_eq!(step2.prefill_tokens, 8);
        assert!(step2.scheduled_seq_ids.contains(&1));
        assert!(step2.scheduled_seq_ids.contains(&2));
    }

    #[test]
    fn decode_never_starved_by_prefill() {
        let pool = default_pool(10);
        let config = SchedulerConfig {
            max_num_batched_tokens: 4,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        scheduler.add_sequence(make_seq(1, 2, 5)).unwrap();
        scheduler.add_sequence(make_seq(2, 10, 5)).unwrap();

        // Step 1: budget=4. Seq 1 gets 2 prefill tokens, seq 2 gets remaining 2.
        let step1 = scheduler.step().unwrap();
        assert_eq!(step1.prefill_tokens, 4);
        assert!(step1.scheduled_seq_ids.contains(&1));
        assert!(step1.scheduled_seq_ids.contains(&2));
        assert_eq!(scheduler.get_sequence(2).unwrap().num_prefilled_tokens, 2);

        // Decode seq 1.
        let mut sampled = HashMap::new();
        sampled.insert(1, 42u32);
        scheduler.postprocess_step(&sampled).unwrap();

        // Now seq 1 is running decode (needs 1 token). Seq 2 is running but still
        // needs 8 more prefill tokens.
        // Budget = 4. Decode gets 1 token, leaving 3 for prefill.
        // With chunked prefill, seq 2 gets a 3-token chunk (split to fit budget).
        let step2 = scheduler.step().unwrap();
        assert_eq!(step2.decode_tokens, 1);
        assert!(step2.scheduled_seq_ids.contains(&1));
        assert_eq!(step2.prefill_tokens, 3); // seq 2 gets remaining budget
        assert!(step2.scheduled_seq_ids.contains(&2));
        assert_eq!(scheduler.get_sequence(2).unwrap().num_prefilled_tokens, 5);
    }

    #[test]
    fn fcfs_order_respected_for_waiting_requests() {
        let pool = default_pool(10);
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config, pool);

        scheduler.add_sequence(make_seq(1, 2, 5)).unwrap();
        scheduler.add_sequence(make_seq(2, 2, 5)).unwrap();
        scheduler.add_sequence(make_seq(3, 2, 5)).unwrap();

        let result = scheduler.step().unwrap();
        // All three fit in default budget (512), scheduled in FCFS order.
        assert_eq!(result.scheduled_seq_ids, vec![1, 2, 3]);

        let orders: Vec<usize> = result
            .scheduled_seq_ids
            .iter()
            .map(|id| scheduler.get_sequence(*id).unwrap().arrival_order())
            .collect();
        assert_eq!(orders, vec![0, 1, 2]);
    }

    #[test]
    fn finished_request_blocks_returned_to_free_pool() {
        let pool = default_pool(10);
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config, pool);

        // 20 tokens needs 2 blocks (block_size=16).
        let seq = make_seq(1, 20, 1);
        scheduler.add_sequence(seq).unwrap();

        let free_before = scheduler.block_pool().free_block_count();

        // Step 1: prefill chunk (default chunk_size=16)
        let step1 = scheduler.step().unwrap();
        assert_eq!(step1.prefill_tokens, 16);

        // Step 2: remaining 4 tokens prefilled.
        let step2 = scheduler.step().unwrap();
        assert_eq!(step2.prefill_tokens, 4);

        // Step 3: decode (max_new=1 → finished after this)
        let mut sampled = HashMap::new();
        sampled.insert(1, 99u32);
        scheduler.postprocess_step(&sampled).unwrap();

        assert_eq!(scheduler.get_sequence(1).unwrap().status(), SequenceStatus::Finished);
        let free_after = scheduler.block_pool().free_block_count();
        assert_eq!(free_after, free_before); // blocks reclaimed
    }

    #[test]
    fn remove_sequence_frees_blocks_and_queues() {
        let pool = default_pool(10);
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config, pool);

        scheduler.add_sequence(make_seq(1, 4, 5)).unwrap();
        scheduler.step().unwrap();

        let free_before = scheduler.block_pool().free_block_count();
        scheduler.remove_sequence(1).unwrap();

        assert_eq!(scheduler.sequence_count(), 0);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.block_pool().free_block_count(), free_before + 1);
    }

    #[test]
    fn multiple_sequences_flatten_into_single_batch() {
        let pool = default_pool(20);
        let config = SchedulerConfig {
            max_num_batched_tokens: 64,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        for i in 1..=4 {
            scheduler.add_sequence(make_seq(i, 4, 3)).unwrap();
        }

        let result = scheduler.step().unwrap();
        assert_eq!(result.scheduled_seq_ids.len(), 4);
        assert_eq!(result.prefill_tokens, 16);
        assert_eq!(result.decode_tokens, 0);
    }

    #[test]
    fn max_running_seqs_limits_prefill() {
        let pool = default_pool(20);
        let config = SchedulerConfig {
            max_num_batched_tokens: 64,
            prefill_chunk_size: 16,
            max_num_running_seqs: 2,
        };
        let mut scheduler = Scheduler::new(config, pool);

        for i in 1..=4 {
            scheduler.add_sequence(make_seq(i, 4, 3)).unwrap();
        }

        let result = scheduler.step().unwrap();
        // Only 2 sequences can enter running state.
        assert_eq!(result.scheduled_seq_ids.len(), 2);
        assert_eq!(scheduler.waiting_count(), 2);
    }

    #[test]
    fn eos_token_finishes_sequence() {
        let pool = default_pool(10);
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config, pool);

        let seq = make_seq(1, 2, 10); // stop_token = Some(0)
        scheduler.add_sequence(seq).unwrap();
        scheduler.step().unwrap();

        // Generate EOS token (0)
        let mut sampled = HashMap::new();
        sampled.insert(1, 0u32);
        scheduler.postprocess_step(&sampled).unwrap();

        assert_eq!(scheduler.get_sequence(1).unwrap().status(), SequenceStatus::Finished);
    }

    #[test]
    fn empty_waiting_queue_step_returns_empty() {
        let pool = default_pool(10);
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config, pool);

        let result = scheduler.step().unwrap();
        assert!(result.scheduled_seq_ids.is_empty());
        assert_eq!(result.prefill_tokens, 0);
        assert_eq!(result.decode_tokens, 0);
    }

    #[test]
    fn block_allocation_failure_propagates() {
        let pool = default_pool(1); // only 1 block
        let config = SchedulerConfig {
            max_num_batched_tokens: 64,
            prefill_chunk_size: 32, // larger than block_size so 20-token prompt needs 2 blocks
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        // 20 tokens needs 2 blocks (block_size=16), but pool only has 1.
        scheduler.add_sequence(make_seq(1, 20, 5)).unwrap();

        let err = scheduler.step().expect_err("should fail out of blocks");
        assert!(matches!(err, SchedulerError::BlockPool(BlockPoolError::OutOfBlocks)));
    }

    #[test]
    fn running_queue_updated_across_steps() {
        let pool = default_pool(10);
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config, pool);

        scheduler.add_sequence(make_seq(1, 2, 1)).unwrap();
        scheduler.step().unwrap();
        assert_eq!(scheduler.running_count(), 1);

        // Finish sequence.
        let mut sampled = HashMap::new();
        sampled.insert(1, 42u32);
        scheduler.postprocess_step(&sampled).unwrap();

        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.get_sequence(1).unwrap().status(), SequenceStatus::Finished);
    }

    #[test]
    fn prefill_chunk_size_splits_long_prompts() {
        let pool = default_pool(10);
        let config = SchedulerConfig {
            max_num_batched_tokens: 64,
            prefill_chunk_size: 8,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        // 20-token prompt with chunk_size = 8.
        scheduler.add_sequence(make_seq(1, 20, 3)).unwrap();

        // Step 1: first 8 tokens prefilled.
        let step1 = scheduler.step().unwrap();
        assert_eq!(step1.prefill_tokens, 8);
        assert_eq!(step1.decode_tokens, 0);
        assert_eq!(scheduler.get_sequence(1).unwrap().status(), SequenceStatus::Running);

        // Step 2: next 8 tokens prefilled (still running, not finished).
        let step2 = scheduler.step().unwrap();
        assert_eq!(step2.prefill_tokens, 8);
        assert_eq!(step2.decode_tokens, 0);

        // Step 3: remaining 4 tokens prefilled.
        let step3 = scheduler.step().unwrap();
        assert_eq!(step3.prefill_tokens, 4);
        assert_eq!(step3.decode_tokens, 0);

        // Step 4: now decode starts.
        let mut sampled = HashMap::new();
        sampled.insert(1, 42u32);
        scheduler.postprocess_step(&sampled).unwrap();

        let step4 = scheduler.step().unwrap();
        assert_eq!(step4.decode_tokens, 1);
        assert_eq!(step4.prefill_tokens, 0);
    }

    // === Validation-contract assertions ===

    /// VAL-PAGED-001: Concurrent requests flatten into InputBatch with batch_size > 1.
    #[test]
    fn concurrent_requests_flatten_into_input_batch_with_batch_size_gt_1() {
        let pool = default_pool(20);
        let config = SchedulerConfig {
            max_num_batched_tokens: 64,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        for i in 1..=4 {
            scheduler.add_sequence(make_seq(i, 4, 3)).unwrap();
        }

        let result = scheduler.step().unwrap();
        assert_eq!(result.scheduled_seq_ids.len(), 4, "all 4 sequences scheduled");

        let batch = scheduler.build_input_batch(&result);
        assert_eq!(batch.batch_size, 4, "InputBatch batch_size > 1");
        assert_eq!(batch.seq_ids.len(), 4);
        assert_eq!(batch.total_tokens, 16);
        assert!(batch.is_prefill.iter().all(|&v| v), "all are prefill");
    }

    /// VAL-PAGED-002: Decode tokens allocated before prefill chunks.
    #[test]
    fn decode_tokens_allocated_before_prefill_chunks() {
        let pool = default_pool(20);
        let config = SchedulerConfig {
            max_num_batched_tokens: 8,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        // Seq 1 is a short prefill that will quickly enter decode.
        scheduler.add_sequence(make_seq(1, 2, 5)).unwrap();
        // Seq 2 is a long waiting prefill.
        scheduler.add_sequence(make_seq(2, 10, 5)).unwrap();

        // Step 1: prefill both (budget=8, 2+6=8).
        let step1 = scheduler.step().unwrap();
        assert_eq!(step1.prefill_tokens, 8);
        assert!(step1.scheduled_seq_ids.contains(&1));
        assert!(step1.scheduled_seq_ids.contains(&2));

        // Decode seq 1.
        let mut sampled = HashMap::new();
        sampled.insert(1, 42u32);
        scheduler.postprocess_step(&sampled).unwrap();

        // Step 2: seq 1 needs decode (1 token). seq 2 still needs 4 more prefill.
        // Budget = 8. Decode gets priority, leaving 7 for prefill.
        let step2 = scheduler.step().unwrap();
        assert_eq!(step2.decode_tokens, 1, "decode allocated first");
        assert!(step2.decode_tokens > 0);
        assert!(step2.prefill_tokens > 0, "prefill follows decode");
        assert!(step2.scheduled_seq_ids.contains(&1));
        assert!(step2.scheduled_seq_ids.contains(&2));
    }

    /// VAL-PAGED-003: Chunked prefill interleaving — decode steps continue between prefill chunks.
    #[test]
    fn chunked_prefill_interleaving_continues_decode_between_chunks() {
        let pool = default_pool(20);
        let config = SchedulerConfig {
            max_num_batched_tokens: 6,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        // Seq 1: short prompt, enters decode quickly.
        scheduler.add_sequence(make_seq(1, 2, 3)).unwrap();
        // Seq 2: long prompt, needs chunked prefill.
        scheduler.add_sequence(make_seq(2, 12, 3)).unwrap();

        // Step 1: prefill seq 1 (2 tokens) + chunk of seq 2 (4 tokens) = 6 budget.
        let step1 = scheduler.step().unwrap();
        assert_eq!(step1.prefill_tokens, 6);
        assert_eq!(step1.decode_tokens, 0);
        assert_eq!(scheduler.get_sequence(2).unwrap().num_prefilled_tokens, 4);

        // Decode seq 1.
        let mut sampled = HashMap::new();
        sampled.insert(1, 42u32);
        scheduler.postprocess_step(&sampled).unwrap();

        // Step 2: seq 1 decode (1 token) + next chunk of seq 2 (5 tokens) = 6 budget.
        let step2 = scheduler.step().unwrap();
        assert_eq!(step2.decode_tokens, 1);
        assert_eq!(step2.prefill_tokens, 5);
        assert!(step2.scheduled_seq_ids.contains(&1));
        assert!(step2.scheduled_seq_ids.contains(&2));

        // Decode seq 1 again.
        let mut sampled = HashMap::new();
        sampled.insert(1, 43u32);
        scheduler.postprocess_step(&sampled).unwrap();

        // Step 3: seq 1 decode (1 token) + remaining seq 2 prefill (3 tokens) = 4.
        let step3 = scheduler.step().unwrap();
        assert_eq!(step3.decode_tokens, 1);
        assert_eq!(step3.prefill_tokens, 3);
        assert_eq!(scheduler.get_sequence(2).unwrap().num_prefilled_tokens, 12);
    }

    /// VAL-PAGED-007: FCFS order respected for waiting requests when sizes identical.
    #[test]
    fn fcfs_order_respected_when_waiting_requests_have_identical_requirements() {
        let pool = default_pool(20);
        let config = SchedulerConfig {
            max_num_batched_tokens: 10,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        scheduler.add_sequence(make_seq(1, 6, 5)).unwrap();
        scheduler.add_sequence(make_seq(2, 6, 5)).unwrap();
        scheduler.add_sequence(make_seq(3, 6, 5)).unwrap();

        // Budget = 10. First two fit (6 + 4 split for third?), but with budget
        // splitting: seq1 gets 6, seq2 gets 4 (split). FCFS means 1 before 2 before 3.
        let result = scheduler.step().unwrap();
        let orders: Vec<usize> = result
            .scheduled_seq_ids
            .iter()
            .map(|id| scheduler.get_sequence(*id).unwrap().arrival_order())
            .collect();

        // Verify the scheduled ids are in ascending arrival_order.
        for window in orders.windows(2) {
            assert!(
                window[0] < window[1],
                "FCFS violated: {:?} should be strictly increasing",
                orders
            );
        }
    }

    /// VAL-PAGED-012: Total tokens per batch never exceed max_num_batched_tokens.
    #[test]
    fn total_tokens_per_batch_never_exceeds_max_num_batched_tokens() {
        let pool = default_pool(20);
        let config = SchedulerConfig {
            max_num_batched_tokens: 8,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        // Mix of running and waiting requests.
        scheduler.add_sequence(make_seq(1, 4, 5)).unwrap();
        scheduler.add_sequence(make_seq(2, 10, 5)).unwrap();
        scheduler.add_sequence(make_seq(3, 6, 5)).unwrap();

        // Simulate multiple steps.
        for _ in 0..10 {
            let result = scheduler.step().unwrap();
            let total = result.prefill_tokens + result.decode_tokens;
            assert!(
                total <= config.max_num_batched_tokens,
                "batch total {} exceeded budget {}",
                total,
                config.max_num_batched_tokens
            );

            let batch = scheduler.build_input_batch(&result);
            assert!(
                batch.total_tokens <= config.max_num_batched_tokens,
                "InputBatch total {} exceeded budget {}",
                batch.total_tokens,
                config.max_num_batched_tokens
            );

            // Postprocess with dummy tokens for all scheduled sequences.
            let mut sampled = HashMap::new();
            for &seq_id in &result.scheduled_seq_ids {
                if !scheduler.get_sequence(seq_id).unwrap().is_finished() {
                    sampled.insert(seq_id, 42u32);
                }
            }
            scheduler.postprocess_step(&sampled).unwrap();

            if scheduler.waiting_count() == 0 && scheduler.running_count() == 0 {
                break;
            }
        }
    }

    /// ITL (inter-token latency) test: a decode request mixed with a long prefill
    /// should not see its ITL spike beyond 2x the baseline single-decode ITL.
    #[test]
    fn decode_itl_remains_within_2x_baseline_when_long_prefill_is_in_progress() {
        let pool = default_pool(30);
        let config = SchedulerConfig {
            max_num_batched_tokens: 6,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        // Baseline: single decode request, measure steps per decode token.
        scheduler.add_sequence(make_seq(100, 2, 5)).unwrap();
        let mut baseline_decode_steps = 0usize;
        for _ in 0..10 {
            let result = scheduler.step().unwrap();
            if result.decode_tokens > 0 {
                baseline_decode_steps += 1;
            }
            let mut sampled = HashMap::new();
            for &seq_id in &result.scheduled_seq_ids {
                if !scheduler.get_sequence(seq_id).unwrap().is_finished() {
                    sampled.insert(seq_id, 42u32);
                }
            }
            scheduler.postprocess_step(&sampled).unwrap();
            if scheduler.get_sequence(100).unwrap().is_finished() {
                break;
            }
        }
        // Remove baseline sequence.
        scheduler.remove_sequence(100).unwrap();

        // Now mix a short decode request with a long prefill request.
        scheduler.add_sequence(make_seq(1, 2, 3)).unwrap(); // short → decode quickly
        scheduler.add_sequence(make_seq(2, 20, 3)).unwrap(); // long prefill

        let mut mixed_decode_steps = 0usize;
        let mut total_decode_tokens = 0usize;
        for _ in 0..20 {
            let result = scheduler.step().unwrap();
            if result.decode_tokens > 0 {
                mixed_decode_steps += 1;
                total_decode_tokens += result.decode_tokens;
            }
            let mut sampled = HashMap::new();
            for &seq_id in &result.scheduled_seq_ids {
                if !scheduler.get_sequence(seq_id).unwrap().is_finished() {
                    sampled.insert(seq_id, 42u32);
                }
            }
            scheduler.postprocess_step(&sampled).unwrap();
            if scheduler.waiting_count() == 0 && scheduler.running_count() == 0 {
                break;
            }
        }

        // ITL ratio = mixed steps per decode / baseline steps per decode.
        // We compare total steps taken to produce the same number of decode tokens.
        // With chunked prefill interleaving, the mixed scenario should not need more
        // than 2x the steps per decode token.
        let baseline_per_token = baseline_decode_steps as f32 / 3.0; // 3 decode tokens
        let mixed_per_token = mixed_decode_steps as f32 / total_decode_tokens as f32;
        assert!(
            mixed_per_token <= 2.0 * baseline_per_token,
            "mixed ITL per token {} exceeded 2x baseline {} (steps={} tokens={})",
            mixed_per_token,
            2.0 * baseline_per_token,
            mixed_decode_steps,
            total_decode_tokens
        );
    }

    /// Verify InputBatch contains the correct block tables and positions for
    /// multiple interleaved prefill+decode sequences.
    #[test]
    fn input_batch_contains_correct_block_tables_for_mixed_prefill_decode() {
        let pool = default_pool(30);
        let config = SchedulerConfig {
            max_num_batched_tokens: 6,
            prefill_chunk_size: 16,
            max_num_running_seqs: 8,
        };
        let mut scheduler = Scheduler::new(config, pool);

        scheduler.add_sequence(make_seq(1, 2, 3)).unwrap();
        scheduler.add_sequence(make_seq(2, 10, 3)).unwrap();

        // Step 1: both prefill.
        let step1 = scheduler.step().unwrap();
        let batch1 = scheduler.build_input_batch(&step1);
        assert_eq!(batch1.batch_size, 2);
        assert_eq!(batch1.is_prefill, vec![true, true]);
        assert_eq!(batch1.num_tokens, vec![2, 4]); // seq1 fully prefilled, seq2 gets 4
        assert!(
            !batch1.block_tables[0].is_empty(),
            "seq 1 has physical blocks assigned"
        );
        assert!(
            !batch1.block_tables[1].is_empty(),
            "seq 2 has physical blocks assigned"
        );

        // Postprocess decode token for seq 1.
        let mut sampled = HashMap::new();
        sampled.insert(1, 42u32);
        scheduler.postprocess_step(&sampled).unwrap();

        // Step 2: seq 1 decode, seq 2 prefill chunk.
        let step2 = scheduler.step().unwrap();
        let batch2 = scheduler.build_input_batch(&step2);
        assert_eq!(batch2.batch_size, 2);
        assert_eq!(batch2.is_prefill, vec![false, true]);
        assert_eq!(batch2.num_tokens, vec![1, 5]);
        assert_eq!(batch2.positions[0], vec![2]); // decode position = prefill_len
    }
}
