use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct GgufTensorRef {
    pub(super) qtype: GgufQuantizationType,
    pub(super) mmap_index: usize,
    pub(super) offset: usize,
    pub(super) size: usize,
    pub(super) value_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct LayerCache {
    pub(super) capacity: usize,
    pub(super) entries: Vec<Option<LayerWeights>>,
    pub(super) access_count: Vec<u64>,
    pub(super) generation: u64,
}

impl LayerCache {
    pub(super) fn new(capacity: usize, layer_count: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: vec![None; layer_count],
            access_count: vec![0; layer_count],
            generation: 0,
        }
    }
    pub(super) fn get(&mut self, layer_idx: usize) -> Option<LayerWeights> {
        self.generation += 1;
        self.access_count[layer_idx] = self.generation;
        self.entries[layer_idx].take()
    }
    pub(super) fn put(&mut self, layer_idx: usize, weights: LayerWeights) {
        if self.entries[layer_idx].is_some() {
            self.entries[layer_idx] = Some(weights);
            return;
        }
        let occupied = self.entries.iter().filter(|e| e.is_some()).count();
        if occupied < self.capacity {
            self.entries[layer_idx] = Some(weights);
            return;
        }
        let mut min_gen = u64::MAX;
        let mut evict_idx = 0;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_some() && self.access_count[i] < min_gen {
                min_gen = self.access_count[i];
                evict_idx = i;
            }
        }
        self.entries[evict_idx] = None;
        self.entries[layer_idx] = Some(weights);
    }
}
