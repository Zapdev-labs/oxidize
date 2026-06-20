use std::ops::Range;

const MIN_SPLIT_K_SEQUENCE_LENGTH: usize = 1024;
const TOKENS_PER_SPLIT: usize = 256;
const MAX_SPLITS: usize = 32;
const TARGET_BLOCKS_PER_SM: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitKPlan {
    pub split_count: usize,
    pub block_count: usize,
}

impl SplitKPlan {
    #[must_use]
    pub const fn select(sm_count: usize, query_heads: usize, seq_len: usize) -> Option<Self> {
        if sm_count == 0 || query_heads == 0 || seq_len < MIN_SPLIT_K_SEQUENCE_LENGTH {
            return None;
        }

        let context_splits = const_min(seq_len.div_ceil(TOKENS_PER_SPLIT), MAX_SPLITS);
        let target_blocks = match sm_count.checked_mul(TARGET_BLOCKS_PER_SM) {
            Some(value) => value,
            None => usize::MAX,
        };
        let occupancy_splits = const_min(target_blocks.div_ceil(query_heads), MAX_SPLITS);
        let split_count = const_min(context_splits, occupancy_splits);
        let block_count = match query_heads.checked_mul(split_count) {
            Some(value) => value,
            None => return None,
        };

        Some(Self {
            split_count,
            block_count,
        })
    }
}

#[must_use]
pub const fn split_k_range(seq_len: usize, split_count: usize, split_idx: usize) -> Range<usize> {
    if split_count == 0 || split_idx >= split_count {
        return 0..0;
    }

    let base_len = seq_len / split_count;
    let remainder = seq_len % split_count;
    let start = split_idx * base_len + const_min(split_idx, remainder);
    let len = base_len + if split_idx < remainder { 1 } else { 0 };
    start..start + len
}

const fn const_min(left: usize, right: usize) -> usize {
    if left < right { left } else { right }
}

#[cfg(test)]
pub(crate) fn merge_softmax_partials_for_test(partials: &[(f32, f32, Vec<f32>)]) -> Vec<f32> {
    let output_len = partials.first().map_or(0, |partial| partial.2.len());
    let global_max = partials
        .iter()
        .map(|partial| partial.0)
        .fold(f32::NEG_INFINITY, f32::max);
    let denominator = partials
        .iter()
        .map(|partial| (partial.0 - global_max).exp() * partial.1)
        .sum::<f32>();
    let mut output = vec![0.0_f32; output_len];

    for (local_max, _, numerator) in partials {
        let scale = (*local_max - global_max).exp() / denominator;
        for (component, partial_component) in output.iter_mut().zip(numerator) {
            *component += scale * partial_component;
        }
    }

    output
}
