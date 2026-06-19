//! Sequence lifecycle management: finishing sequences, removal, draining, and
//! prefix-cache invalidation.

use super::*;

impl Scheduler {
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

    /// Clear the prefix cache and reset all sequences.
    ///
    /// Called when switching models so that stale KV blocks are not reused.
    pub fn invalidate_prefix_cache(&mut self) {
        self.block_pool.clear_prefix_cache();
    }

    /// Drain all sequences, free their blocks, and reinitialize the scheduler
    /// state so it can accept a new backend or model.
    ///
    /// This is called when the compute backend is switched (e.g. CPU → MLX)
    /// so that no stale KV block data leaks across backends.
    pub fn drain_and_reinitialize(&mut self) -> Result<(), SchedulerError> {
        // Free blocks for every sequence still in the scheduler.
        for seq in self.sequences.values() {
            for &block_id in seq.block_table.physical_blocks() {
                // Ignore errors — blocks may already be on the free list
                // if the sequence was finished.
                let _ = self.block_pool.dec_ref(block_id);
            }
        }
        // Clear prefix cache and reset queues.
        self.block_pool.clear_prefix_cache();
        self.sequences.clear();
        self.waiting.clear();
        self.running.clear();
        self.next_arrival_order = 0;
        Ok(())
    }
}
