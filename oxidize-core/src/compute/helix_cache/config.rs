#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelixCacheConfig {
    pub page_size: usize,
    pub head_dim: usize,
    pub key_radius_bits: u8,
    pub key_phase_bits: u8,
    pub value_bits: u8,
    pub inactive_threshold: f32,
    pub promotion_epsilon: f32,
    pub promotion_budget: u32,
}

impl Default for HelixCacheConfig {
    fn default() -> Self {
        Self {
            page_size: 64,
            head_dim: 128,
            key_radius_bits: 4,
            key_phase_bits: 4,
            value_bits: 3,
            inactive_threshold: 0.0,
            promotion_epsilon: 0.1,
            promotion_budget: 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HelixCacheStats {
    pub cold_pages: usize,
    pub hot_pages: usize,
    pub token_count: usize,
    pub key_bytes: usize,
    pub value_bytes: usize,
    pub hot_bytes: usize,
    pub metadata_bytes: usize,
    pub key_metadata_bytes: usize,
    pub value_metadata_bytes: usize,
    pub page_metadata_bytes: usize,
    pub f32_baseline_bytes: usize,
    pub key_bits_per_coord: f32,
    pub value_bits_per_coord: f32,
    pub total_bits_per_coord: f32,
}

impl HelixCacheStats {
    pub(crate) fn empty() -> Self {
        Self {
            cold_pages: 0,
            hot_pages: 0,
            token_count: 0,
            key_bytes: 0,
            value_bytes: 0,
            hot_bytes: 0,
            metadata_bytes: 0,
            key_metadata_bytes: 0,
            value_metadata_bytes: 0,
            page_metadata_bytes: 0,
            f32_baseline_bytes: 0,
            key_bits_per_coord: 0.0,
            value_bits_per_coord: 0.0,
            total_bits_per_coord: 0.0,
        }
    }

    pub fn compression_ratio_vs_f32(&self) -> f32 {
        let current_bytes = self
            .key_bytes
            .saturating_add(self.value_bytes)
            .saturating_add(self.hot_bytes)
            .saturating_add(self.metadata_bytes);
        if current_bytes == 0 {
            1.0
        } else {
            self.f32_baseline_bytes as f32 / current_bytes as f32
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum HelixCacheError {
    #[error("invalid HelixCache config: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid HelixCache page shape: {0}")]
    Shape(&'static str),
    #[error("HelixCache page not found")]
    PageNotFound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HelixPageTier {
    Cold,
    Hot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PageKey {
    pub layer: usize,
    pub head: usize,
    pub page: usize,
}
