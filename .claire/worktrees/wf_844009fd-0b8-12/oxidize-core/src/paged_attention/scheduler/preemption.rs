//! Memory-pressure and block reclamation methods for the scheduler.

use super::policy::SchedulerError;
use super::request::{SeqId, SequenceStatus};
use super::super::BlockTable;
use super::Scheduler;

impl Scheduler {
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
        let physical_blocks: Vec<_> = seq.block_table.physical_blocks().to_vec();
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
