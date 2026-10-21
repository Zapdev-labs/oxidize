use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct SequenceState {
    positions: Vec<usize>,
    last_active_step: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContinuousBatchKvCache {
    kv_cache: KvCache,
    max_sequences: usize,
    current_step: usize,
    next_position: usize,
    sequences: HashMap<u64, SequenceState>,
    #[serde(skip)]
    pooled_positions: Vec<Vec<usize>>,
}

impl ContinuousBatchKvCache {
    pub fn new(kv_cache: KvCache, max_sequences: usize) -> Self {
        Self {
            kv_cache,
            max_sequences,
            current_step: 0,
            next_position: 0,
            sequences: HashMap::new(),
            pooled_positions: Vec::new(),
        }
    }

    pub fn begin_step(&mut self) {
        self.current_step = self.current_step.saturating_add(1);
    }

    pub fn add_sequence(&mut self, sequence_id: u64) -> Result<(), ContinuousBatchError> {
        if self.sequences.contains_key(&sequence_id) {
            return Err(ContinuousBatchError::SequenceAlreadyExists { sequence_id });
        }
        if self.sequences.len() >= self.max_sequences {
            return Err(ContinuousBatchError::SequenceCapacityExceeded {
                max_sequences: self.max_sequences,
            });
        }
        let positions = self.pooled_positions.pop().unwrap_or_default();
        self.sequences.insert(
            sequence_id,
            SequenceState {
                positions,
                last_active_step: self.current_step,
            },
        );
        Ok(())
    }

    pub fn remove_sequence(&mut self, sequence_id: u64) -> Result<(), ContinuousBatchError> {
        let state = self
            .sequences
            .remove(&sequence_id)
            .ok_or(ContinuousBatchError::SequenceNotFound { sequence_id })?;
        self.recycle_positions(state.positions);
        Ok(())
    }

    pub fn evict_inactive_sequences(&mut self, max_idle_steps: usize) {
        let eviction_step = self.current_step.saturating_sub(max_idle_steps);
        let evicted_ids = self
            .sequences
            .iter()
            .filter_map(|(sequence_id, state)| {
                (state.last_active_step < eviction_step).then_some(*sequence_id)
            })
            .collect::<Vec<_>>();
        for sequence_id in evicted_ids {
            if let Some(state) = self.sequences.remove(&sequence_id) {
                self.recycle_positions(state.positions);
            }
        }
    }

    pub fn append_token(
        &mut self,
        sequence_id: u64,
        layer: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<usize, ContinuousBatchError> {
        let state = self
            .sequences
            .get_mut(&sequence_id)
            .ok_or(ContinuousBatchError::SequenceNotFound { sequence_id })?;
        let position = self.next_position;
        self.kv_cache.set(layer, position, key, value)?;
        state.positions.push(position);
        state.last_active_step = self.current_step;
        self.next_position = self.next_position.saturating_add(1);
        Ok(position)
    }

    pub fn get_sequence_key(
        &self,
        sequence_id: u64,
        layer: usize,
        token_index: usize,
        out: &mut [f32],
    ) -> Result<(), ContinuousBatchError> {
        let position = self.position_for(sequence_id, token_index)?;
        self.kv_cache.get_key(layer, position, out)?;
        Ok(())
    }

    pub fn get_sequence_value(
        &self,
        sequence_id: u64,
        layer: usize,
        token_index: usize,
        out: &mut [f32],
    ) -> Result<(), ContinuousBatchError> {
        let position = self.position_for(sequence_id, token_index)?;
        self.kv_cache.get_value(layer, position, out)?;
        Ok(())
    }

    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    pub fn cache(&self) -> &KvCache {
        &self.kv_cache
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), KvCachePersistenceError> {
        let payload = serde_json::to_vec(self)?;
        std::fs::write(path, payload)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, KvCachePersistenceError> {
        let payload = std::fs::read(path)?;
        let mut cache: Self = serde_json::from_slice(&payload)?;
        cache.kv_cache.migrate_legacy_storage_layout();
        Ok(cache)
    }

    fn position_for(
        &self,
        sequence_id: u64,
        token_index: usize,
    ) -> Result<usize, ContinuousBatchError> {
        let state = self
            .sequences
            .get(&sequence_id)
            .ok_or(ContinuousBatchError::SequenceNotFound { sequence_id })?;
        state.positions.get(token_index).copied().ok_or(
            ContinuousBatchError::TokenIndexOutOfBounds {
                sequence_id,
                token_index,
                token_count: state.positions.len(),
            },
        )
    }

    fn recycle_positions(&mut self, mut positions: Vec<usize>) {
        if self.pooled_positions.len() >= self.max_sequences {
            return;
        }
        positions.clear();
        self.pooled_positions.push(positions);
    }

    #[cfg(test)]
    pub(super) fn pooled_position_buffer_count(&self) -> usize {
        self.pooled_positions.len()
    }
}
