use super::config::{HelixCacheConfig, HelixCacheError, HelixPageTier, PageKey};
use super::pack::{packed_bits_bytes, packed_nibble_bytes};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ColdKeyTile {
    pub(super) mu_phi: Vec<f32>,
    pub(super) log_rho_min: Vec<f32>,
    pub(super) log_rho_step: Vec<f32>,
    pub(super) active_mask: Vec<u8>,
    pub(super) rho_codes: Vec<u8>,
    pub(super) phi_codes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ColdValueTile {
    pub(super) scales: Vec<f32>,
    pub(super) codes: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct PromotionState {
    pub(super) uncertainty_counter: u32,
    pub(super) recent_max_overlap: f32,
    pub(super) access_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Page {
    pub(super) key: PageKey,
    pub(super) tier: HelixPageTier,
    pub(super) positions: Vec<usize>,
    pub(super) cold_key: ColdKeyTile,
    pub(super) cold_value: ColdValueTile,
    pub(super) hot_keys: Vec<f32>,
    pub(super) hot_values: Vec<f32>,
    pub(super) promotion: PromotionState,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HelixCache {
    pub(super) config: HelixCacheConfig,
    pub(super) pages: Vec<Page>,
}

impl HelixCache {
    pub fn new(config: HelixCacheConfig) -> Result<Self, HelixCacheError> {
        if config.page_size == 0 {
            return Err(HelixCacheError::InvalidConfig("page_size must be non-zero"));
        }
        if config.head_dim == 0 || config.head_dim % 8 != 0 {
            return Err(HelixCacheError::InvalidConfig(
                "head_dim must be a non-zero multiple of 8",
            ));
        }
        if config.key_radius_bits != 4 || config.key_phase_bits != 4 || config.value_bits != 3 {
            return Err(HelixCacheError::InvalidConfig(
                "only 4-bit keys and 3-bit values are supported",
            ));
        }
        Ok(Self {
            config,
            pages: Vec::new(),
        })
    }

    pub fn store_hot_page(
        &mut self,
        layer: usize,
        kv_head: usize,
        page_id: usize,
        pre_rope_keys: &[f32],
        values: &[f32],
        positions: &[usize],
    ) -> Result<(), HelixCacheError> {
        self.validate_page_shape(pre_rope_keys, values, positions, "hot")?;
        self.pages.push(Page {
            key: PageKey {
                layer,
                head: kv_head,
                page: page_id,
            },
            tier: HelixPageTier::Hot,
            positions: positions.to_vec(),
            cold_key: empty_key(),
            cold_value: empty_value(),
            hot_keys: pre_rope_keys.to_vec(),
            hot_values: values.to_vec(),
            promotion: PromotionState::default(),
        });
        Ok(())
    }

    pub fn store_cold_page(
        &mut self,
        layer: usize,
        kv_head: usize,
        page_id: usize,
        pre_rope_keys: &[f32],
        values: &[f32],
        positions: &[usize],
    ) -> Result<(), HelixCacheError> {
        self.validate_page_shape(pre_rope_keys, values, positions, "cold")?;
        let pairs = self.config.head_dim / 2;
        let codes = positions.len() * pairs;
        let mut page = Page {
            key: PageKey {
                layer,
                head: kv_head,
                page: page_id,
            },
            tier: HelixPageTier::Cold,
            positions: positions.to_vec(),
            cold_key: ColdKeyTile {
                mu_phi: vec![0.0; pairs],
                log_rho_min: vec![0.0; pairs],
                log_rho_step: vec![0.0; pairs],
                active_mask: vec![0; packed_bits_bytes(codes)],
                rho_codes: vec![0; packed_nibble_bytes(codes)],
                phi_codes: vec![0; packed_nibble_bytes(codes)],
            },
            cold_value: ColdValueTile {
                scales: vec![0.0; self.config.head_dim / 8],
                codes: vec![0; packed_bits_bytes(positions.len() * self.config.head_dim * 3)],
            },
            hot_keys: Vec::new(),
            hot_values: Vec::new(),
            promotion: PromotionState::default(),
        };
        self.encode_keys(&mut page, pre_rope_keys);
        self.encode_values(&mut page, values);
        self.pages.push(page);
        Ok(())
    }

    fn validate_page_shape(
        &self,
        pre_rope_keys: &[f32],
        values: &[f32],
        positions: &[usize],
        tier: &'static str,
    ) -> Result<(), HelixCacheError> {
        let tokens = positions.len();
        if tokens == 0 || tokens > self.config.page_size {
            return Err(HelixCacheError::Shape(tier));
        }
        let expected = tokens * self.config.head_dim;
        if pre_rope_keys.len() != expected || values.len() != expected {
            return Err(HelixCacheError::Shape(tier));
        }
        Ok(())
    }

    pub(super) fn matching_pages(
        &self,
        layer: usize,
        kv_head: usize,
    ) -> impl Iterator<Item = &Page> {
        self.pages
            .iter()
            .filter(move |page| page.key.layer == layer && page.key.head == kv_head)
    }
}

fn empty_key() -> ColdKeyTile {
    ColdKeyTile {
        mu_phi: Vec::new(),
        log_rho_min: Vec::new(),
        log_rho_step: Vec::new(),
        active_mask: Vec::new(),
        rho_codes: Vec::new(),
        phi_codes: Vec::new(),
    }
}

fn empty_value() -> ColdValueTile {
    ColdValueTile {
        scales: Vec::new(),
        codes: Vec::new(),
    }
}

pub(super) fn same_key(key: PageKey, layer: usize, head: usize, page: usize) -> bool {
    key.layer == layer && key.head == head && key.page == page
}
