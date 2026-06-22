use std::ops::Range;

const MIN_SPLIT_K_SEQUENCE_LENGTH: usize = 1024;
const TOKENS_PER_SPLIT: usize = 256;
pub(crate) const MAX_SPLITS: usize = 32;
const TARGET_BLOCKS_PER_SM: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitKPlan {
    pub split_count: usize,
    pub block_count: usize,
}

impl SplitKPlan {
    #[must_use]
    pub const fn select(sm_count: usize, query_heads: usize, seq_len: usize) -> Option<Self> {
        if sm_count == 0 || query_heads == 0 || seq_len == 0 {
            return None;
        }

        let target_blocks = match sm_count.checked_mul(TARGET_BLOCKS_PER_SM) {
            Some(value) => value,
            None => usize::MAX,
        };
        let occupancy_splits = const_min(target_blocks.div_ceil(query_heads), MAX_SPLITS);
        let context_splits = if seq_len >= MIN_SPLIT_K_SEQUENCE_LENGTH {
            const_min(seq_len.div_ceil(TOKENS_PER_SPLIT), MAX_SPLITS)
        } else {
            1
        };
        // Long contexts: cap splits by both occupancy and KV length. Short contexts
        // (typical decode benches): still use occupancy splits so H100-class GPUs
        // launch enough flash-attn blocks to fill SMs (OX_FLASH_DECODE_SPLITS=8
        // was +14% on Mistral-7B; auto-select now picks the same on sm_90).
        let split_count = if context_splits >= 2 {
            const_min(context_splits, occupancy_splits)
        } else if occupancy_splits >= 2 {
            occupancy_splits
        } else {
            return None;
        };
        let split_count = const_min(split_count, seq_len);
        if split_count < 2 {
            return None;
        }
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
pub const fn flash_decode_scratch_lengths(
    query_heads: usize,
    split_count: usize,
    head_dim: usize,
) -> Option<(usize, usize)> {
    if query_heads == 0 || split_count == 0 || head_dim == 0 {
        return None;
    }
    let slots = match query_heads.checked_mul(split_count) {
        Some(value) => value,
        None => return None,
    };
    let accumulator_len = match slots.checked_mul(head_dim) {
        Some(value) => value,
        None => return None,
    };
    Some((slots, accumulator_len))
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn validate_flash_decode_launch(
    kv_layers: usize,
    kv_layer_idx: usize,
    kv_context: usize,
    kv_len: usize,
    sequence_length: usize,
    base_row: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    split_count: usize,
) -> Result<(), String> {
    if kv_layers == 0
        || kv_context == 0
        || kv_len == 0
        || sequence_length == 0
        || query_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || split_count == 0
    {
        return Err("flash decode launch dimensions must be nonzero".to_owned());
    }
    if kv_layer_idx >= kv_layers {
        return Err(format!(
            "flash decode layer {kv_layer_idx} is outside {kv_layers} KV layers"
        ));
    }
    if !query_heads.is_multiple_of(kv_heads) {
        return Err("flash decode query heads must be divisible by KV heads".to_owned());
    }
    if !head_dim.is_power_of_two() {
        return Err("flash decode head dimension must be a power of two".to_owned());
    }
    if head_dim > 256 {
        return Err("flash decode head dimension must not exceed 256".to_owned());
    }
    if split_count > sequence_length {
        return Err("flash decode split count must not exceed sequence length".to_owned());
    }
    let expected_kv_len = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "flash decode KV width overflow".to_owned())?;
    if expected_kv_len != kv_len {
        return Err(format!(
            "flash decode KV width mismatch: cache={kv_len}, expected={expected_kv_len}"
        ));
    }
    let window_end = base_row
        .checked_add(sequence_length)
        .ok_or_else(|| "flash decode cache row overflow".to_owned())?;
    if window_end > kv_context {
        return Err(format!(
            "flash decode window ends at {window_end}, beyond cache context {kv_context}"
        ));
    }
    flash_decode_scratch_lengths(query_heads, split_count, head_dim)
        .ok_or_else(|| "flash decode scratch shape overflow".to_owned())?;
    Ok(())
}

#[must_use]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn select_split_count(
    sm_count: usize,
    query_heads: usize,
    sequence_length: usize,
    force_legacy: bool,
    forced_text: Option<&str>,
) -> usize {
    if force_legacy {
        return 1;
    }
    if let Some(forced) = forced_text.and_then(|text| text.parse::<usize>().ok()) {
        return forced.clamp(1, const_min(sequence_length.max(1), MAX_SPLITS));
    }
    SplitKPlan::select(sm_count, query_heads, sequence_length).map_or(1, |plan| plan.split_count)
}

#[cfg(test)]
pub(crate) fn select_split_count_for_test(
    sm_count: usize,
    query_heads: usize,
    seq_len: usize,
    force_legacy: bool,
    forced_text: Option<&str>,
) -> usize {
    select_split_count(sm_count, query_heads, seq_len, force_legacy, forced_text)
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
