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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerStepResult {
    /// Sequence IDs scheduled for this step.
    pub scheduled_seq_ids: Vec<SeqId>,
    /// Number of prefill tokens scheduled.
    pub prefill_tokens: usize,
    /// Number of decode tokens scheduled.
    pub decode_tokens: usize,
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

            let chunk_size = seq
                .remaining_prefill_tokens()
                .min(self.config.prefill_chunk_size);
            if chunk_size > budget {
                continue;
            }

            Self::apply_prefill_chunk(&mut self.block_pool, seq, chunk_size)?;

            scheduled.push(seq_id);
            budget -= chunk_size;
            prefill_tokens += chunk_size;
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

            let chunk_size = remaining_prefill.min(self.config.prefill_chunk_size);
            if chunk_size > budget {
                still_waiting.push_back(seq_id);
                continue;
            }

            Self::apply_prefill_chunk(&mut self.block_pool, seq, chunk_size)?;
            seq.set_status(SequenceStatus::Running);

            scheduled.push(seq_id);
            running_count += 1;
            budget -= chunk_size;
            prefill_tokens += chunk_size;
        }

        self.waiting = still_waiting;

        // Every sequence in `scheduled` is Running by construction.
        self.running = scheduled.clone();

        Ok(SchedulerStepResult {
            scheduled_seq_ids: scheduled,
            prefill_tokens,
            decode_tokens,
        })
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
        // Budget = 8. First request takes 6 prefill tokens. Second request would
        // need 6 more → total 12 > 8, so second request stays waiting.
        assert_eq!(result.prefill_tokens, 6);
        assert_eq!(result.scheduled_seq_ids, vec![1]);
        assert_eq!(scheduler.waiting_count(), 1);
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

        // Prefill seq 1.
        let step1 = scheduler.step().unwrap();
        assert_eq!(step1.prefill_tokens, 2);

        // Decode seq 1.
        let mut sampled = HashMap::new();
        sampled.insert(1, 42u32);
        scheduler.postprocess_step(&sampled).unwrap();

        // Now seq 1 is running decode (needs 1 token). Seq 2 is waiting (needs 10).
        // Budget = 4. Decode gets 1 token, leaving 3 for prefill.
        // Seq 2's prefill chunk is min(10, 16) = 10, but only 3 fit.
        // However our scheduler doesn't split prefill chunks in this core version,
        // so seq 2 stays waiting.
        let step2 = scheduler.step().unwrap();
        assert_eq!(step2.decode_tokens, 1);
        assert!(step2.scheduled_seq_ids.contains(&1));
        assert_eq!(step2.prefill_tokens, 0); // seq 2 deferred
        assert_eq!(scheduler.waiting_count(), 1); // seq 2 still waiting
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
}
