//! Prefix caching and copy-on-write: prefix cache hit detection, cache-aware
//! prefill, preemption, and decode-time copy-on-write.

use super::*;

impl Scheduler {
    /// For a given sequence, determine how many prefix tokens can be served
    /// from the prefix cache.
    ///
    /// Walks the prompt tokens block-by-block, computing the cumulative hash
    /// for each block and checking the global prefix cache. Returns the
    /// number of tokens that are already cached (i.e. do not need recomputation).
    pub fn find_prefix_cache_hits(&mut self, seq_id: SeqId) -> Result<usize, SchedulerError> {
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(SchedulerError::SequenceNotFound { seq_id })?;
        let prompt = &seq.prompt_tokens;
        let block_size = seq.block_table.block_size();
        if prompt.is_empty() {
            return Ok(0);
        }

        let mut cached_tokens = 0usize;
        let num_blocks = prompt.len().div_ceil(block_size);
        for block_idx in 0..num_blocks {
            let block_end = ((block_idx + 1) * block_size).min(prompt.len());
            let hash = compute_block_hash(&prompt[..block_end]);
            if let Some(_block_id) = self.block_pool.lookup_prefix_cache(hash) {
                cached_tokens = block_end;
                // Ensure the block is tracked in the sequence's block table.
                if seq.block_table.num_blocks() <= block_idx {
                    // This will be fixed by apply_prefill_chunk_with_prefix_cache.
                }
            } else {
                break;
            }
        }
        Ok(cached_tokens)
    }

    /// Allocate blocks for a prefill chunk, using the prefix cache where possible.
    ///
    /// For each logical block that is fully within the already-cached prefix,
    /// the physical block is shared (ref_count incremented). For blocks beyond
    /// the cached prefix, new physical blocks are allocated.
    ///
    /// Returns the number of **newly computed** tokens (i.e. tokens that were
    /// NOT covered by the prefix cache).  Cached tokens are skipped — their
    /// physical blocks are shared and no KV computation is needed for them.
    pub fn apply_prefill_chunk_with_prefix_cache(
        &mut self,
        seq_id: SeqId,
        chunk_size: usize,
    ) -> Result<usize, SchedulerError> {
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(SchedulerError::SequenceNotFound { seq_id })?;
        let prompt = seq.prompt_tokens.clone();
        let block_size = seq.block_table.block_size();
        let already_prefilled = seq.num_prefilled_tokens;
        let this_chunk = seq.remaining_prefill_tokens().min(chunk_size);
        if this_chunk == 0 {
            return Ok(0);
        }

        // --- Compute how many tokens are cached overall for this prompt. ---
        let mut cached_tokens_total = 0usize;
        if !prompt.is_empty() {
            let num_blocks = prompt.len().div_ceil(block_size);
            for block_idx in 0..num_blocks {
                let block_end = ((block_idx + 1) * block_size).min(prompt.len());
                let hash = compute_block_hash(&prompt[..block_end]);
                if self.block_pool.lookup_prefix_cache(hash).is_some() {
                    cached_tokens_total = block_end;
                } else {
                    break;
                }
            }
        }

        // --- How many tokens in *this chunk* are already cached? ---
        let chunk_end = already_prefilled + this_chunk;
        let cached_in_chunk = if cached_tokens_total > already_prefilled {
            cached_tokens_total
                .min(chunk_end)
                .saturating_sub(already_prefilled)
        } else {
            0
        };
        let new_tokens_in_chunk = this_chunk - cached_in_chunk;

        // --- Ensure block table has physical blocks for all tokens up to chunk_end. ---
        let target_blocks = chunk_end.div_ceil(block_size);
        let current_blocks = self
            .sequences
            .get(&seq_id)
            .ok_or(SchedulerError::SequenceNotFound { seq_id })?
            .block_table
            .num_blocks();

        for block_idx in current_blocks..target_blocks {
            let block_end = ((block_idx + 1) * block_size).min(prompt.len());
            let hash = compute_block_hash(&prompt[..block_end]);

            // Resolve the physical block first (this borrows `self.block_pool`),
            // then do a single `sequences` lookup to append it. Keeping the two
            // borrows disjoint lets us fetch the sequence once per iteration
            // instead of once per branch.
            let block_id = if block_end <= cached_tokens_total {
                // Fully cached block — share it if the cache entry still exists,
                // otherwise allocate fresh (it was evicted since we computed
                // `cached_tokens_total`).
                if let Some(block_id) = self.block_pool.lookup_prefix_cache(hash) {
                    self.block_pool.inc_ref(block_id)?;
                    block_id
                } else {
                    self.block_pool.allocate_block()?
                }
            } else {
                // New or partially-cached block — allocate fresh.
                self.block_pool.allocate_block()?
            };
            let seq = self
                .sequences
                .get_mut(&seq_id)
                .ok_or(SchedulerError::SequenceNotFound { seq_id })?;
            seq.block_table.append_block(block_id);
        }

        // --- Advance token counters. ---
        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(SchedulerError::SequenceNotFound { seq_id })?;
        for _ in 0..this_chunk {
            let _ = seq.block_table.append_token();
        }
        seq.record_prefilled_tokens(this_chunk);

        // --- Insert newly-computed blocks into the prefix cache. ---
        let seq = self
            .sequences
            .get(&seq_id)
            .ok_or(SchedulerError::SequenceNotFound { seq_id })?;
        for block_idx in 0..target_blocks {
            let block_end = ((block_idx + 1) * block_size).min(prompt.len());
            // Only cache blocks that were not fully cached before this call.
            if block_end > cached_tokens_total {
                let hash = compute_block_hash(&prompt[..block_end]);
                if let Some(physical_id) = seq.block_table.get_physical_block(block_idx) {
                    self.block_pool.insert_prefix_cache(hash, physical_id);
                }
            }
        }

        Ok(new_tokens_in_chunk)
    }

    /// Preempt a sequence: free its physical blocks (decrement ref counts)
    /// and move it back to the waiting queue so it can resume later.
    ///
    /// Works regardless of formal status — any sequence that holds physical
    /// blocks will have them freed and its state reset.
    ///
    /// The prefix cache entries for the freed blocks are NOT removed — they
    /// remain so that the sequence can resume from the cached prefix.
    pub fn preempt_sequence(&mut self, seq_id: SeqId) -> Result<(), SchedulerError> {
        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(SchedulerError::SequenceNotFound { seq_id })?;

        // Free all physical blocks held by this sequence.
        let physical_blocks: Vec<BlockId> = seq.block_table.physical_blocks().to_vec();
        for block_id in physical_blocks {
            self.block_pool.dec_ref(block_id)?;
        }

        // Reset block table and prefilled state, but keep prompt tokens.
        seq.block_table = BlockTable::new(seq.block_table.block_size());
        seq.num_prefilled_tokens = 0;
        seq.set_status(SequenceStatus::Waiting);

        // Remove from running, add back to waiting queue (front, so it gets
        // priority when rescheduled).
        self.running.retain(|&id| id != seq_id);
        if !self.waiting.contains(&seq_id) {
            self.waiting.push_front(seq_id);
        }
        Ok(())
    }

    /// Copy-on-write during decode: if the sequence's last block is shared
    /// (ref_count > 1), allocate a new block, update the block table, and
    /// decrement the original block's ref_count.
    ///
    /// Returns `true` if COW was triggered.
    pub fn cow_decode_block(&mut self, seq_id: SeqId) -> Result<bool, SchedulerError> {
        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(SchedulerError::SequenceNotFound { seq_id })?;
        let last_logical = seq.block_table.num_blocks().saturating_sub(1);
        let Some(original_id) = seq.block_table.get_physical_block(last_logical) else {
            return Ok(false);
        };
        let cow_result = self.block_pool.copy_on_write(original_id)?;
        if let Some(new_id) = cow_result {
            seq.block_table.set_physical_block(last_logical, new_id);
            return Ok(true);
        }
        Ok(false)
    }
}
