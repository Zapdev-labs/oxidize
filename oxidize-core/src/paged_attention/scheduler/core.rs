//! Core scheduling loop and batch building: the `Scheduler` struct, queue
//! management, the `step()` scheduling loop, and `build_input_batch()`.

use super::*;

/// The PagedAttention scheduler manages the waiting and running queues,
/// enforces the token budget per step, and transitions sequences through
/// the WAITING → RUNNING → FINISHED lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub struct Scheduler {
    pub(super) config: SchedulerConfig,
    pub(super) block_pool: BlockPool,
    /// All sequences keyed by id.
    pub(super) sequences: HashMap<SeqId, Sequence>,
    /// Waiting queue: sequences that have not yet started prefill.
    pub(super) waiting: VecDeque<SeqId>,
    /// Running queue: sequences currently in a batch.
    pub(super) running: Vec<SeqId>,
    /// Monotonically increasing arrival counter for FCFS ordering.
    pub(super) next_arrival_order: usize,
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
            } else {
                // Writing to the existing last block — trigger COW if shared.
                self.cow_decode_block(seq_id)?;
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
        let mut context_lens = Vec::with_capacity(batch_size);
        let mut total_tokens = 0usize;

        for &seq_id in &step_result.scheduled_seq_ids {
            let seq = match self.sequences.get(&seq_id) {
                Some(s) => s,
                None => continue,
            };

            // Determine whether this sequence received prefill or decode tokens.
            let prefill_count = step_result
                .seq_prefill_tokens
                .get(&seq_id)
                .copied()
                .unwrap_or(0);
            let decode_count = step_result
                .seq_decode_tokens
                .get(&seq_id)
                .copied()
                .unwrap_or(0);

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
                context_lens.push(seq.num_prefilled_tokens);
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
                context_lens.push(seq.num_tokens());
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
            context_lens,
        }
    }
}
