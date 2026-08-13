use super::codec::{HelixCache, Page, same_key};
use super::config::{HelixCacheError, HelixCacheStats, HelixPageTier};
use super::pack::{PI, get_bit, get_int3, get_nibble, hadamard8, rope_frequency, wrap_angle};
use std::mem::size_of;

impl HelixCache {
    pub fn logits(
        &self,
        layer: usize,
        kv_head: usize,
        query_pre_rope: &[f32],
        query_position: usize,
        rope_theta: f32,
    ) -> Result<Vec<f32>, HelixCacheError> {
        if query_pre_rope.len() != self.config.head_dim {
            return Err(HelixCacheError::Shape("query shape mismatch"));
        }
        let pairs = self.config.head_dim / 2;
        let mut q_mag = vec![0.0; pairs];
        let mut q_phi = vec![0.0; pairs];
        for pair in 0..pairs {
            let x = query_pre_rope[2 * pair];
            let y = query_pre_rope[2 * pair + 1];
            q_mag[pair] = (x * x + y * y).sqrt();
            q_phi[pair] = y.atan2(x);
        }
        let mut logits = Vec::new();
        for page in self.matching_pages(layer, kv_head) {
            self.page_logits(
                page,
                &q_mag,
                &q_phi,
                query_position,
                rope_theta,
                &mut logits,
            );
        }
        Ok(logits)
    }

    pub fn attention(
        &mut self,
        layer: usize,
        kv_head: usize,
        query_pre_rope: &[f32],
        query_position: usize,
        rope_theta: f32,
    ) -> Result<Vec<f32>, HelixCacheError> {
        let mut scores = self.logits(layer, kv_head, query_pre_rope, query_position, rope_theta)?;
        if scores.is_empty() {
            return Ok(vec![0.0; self.config.head_dim]);
        }
        normalize_scores(&mut scores, self.config.head_dim);
        let mut out = vec![0.0; self.config.head_dim];
        self.accumulate_values(layer, kv_head, &scores, &mut out);
        Ok(out)
    }

    pub fn bump_uncertainty(
        &mut self,
        layer: usize,
        kv_head: usize,
        page_id: usize,
        interval_overlap: f32,
    ) -> Result<(), HelixCacheError> {
        let page = self
            .pages
            .iter_mut()
            .find(|page| same_key(page.key, layer, kv_head, page_id))
            .ok_or(HelixCacheError::PageNotFound)?;
        page.promotion.recent_max_overlap = page.promotion.recent_max_overlap.max(interval_overlap);
        if interval_overlap >= self.config.promotion_epsilon {
            page.promotion.uncertainty_counter += 1;
        }
        Ok(())
    }

    pub fn should_promote(
        &self,
        layer: usize,
        kv_head: usize,
        page_id: usize,
    ) -> Result<bool, HelixCacheError> {
        let page = self
            .pages
            .iter()
            .find(|page| same_key(page.key, layer, kv_head, page_id))
            .ok_or(HelixCacheError::PageNotFound)?;
        Ok(page.tier == HelixPageTier::Cold
            && page.promotion.uncertainty_counter >= self.config.promotion_budget)
    }

    pub fn stats(&self) -> HelixCacheStats {
        let mut stats = HelixCacheStats::empty();
        for page in &self.pages {
            add_page_stats(&mut stats, page);
        }
        stats.metadata_bytes =
            stats.key_metadata_bytes + stats.value_metadata_bytes + stats.page_metadata_bytes;
        stats.f32_baseline_bytes = stats.token_count * self.config.head_dim * 2 * size_of::<f32>();
        let coords = (stats.token_count * self.config.head_dim) as f32;
        if coords > 0.0 {
            stats.key_bits_per_coord =
                ((stats.key_bytes + stats.key_metadata_bytes) * 8) as f32 / coords;
            stats.value_bits_per_coord =
                ((stats.value_bytes + stats.value_metadata_bytes) * 8) as f32 / coords;
            stats.total_bits_per_coord = ((stats.key_bytes
                + stats.value_bytes
                + stats.hot_bytes
                + stats.metadata_bytes)
                * 8) as f32
                / coords;
        }
        stats
    }

    pub(super) fn page_logits(
        &self,
        page: &Page,
        q_mag: &[f32],
        q_phi: &[f32],
        query_position: usize,
        rope_theta: f32,
        logits: &mut Vec<f32>,
    ) {
        let pairs = self.config.head_dim / 2;
        for token in 0..page.positions.len() {
            let mut logit = 0.0;
            for pair in 0..pairs {
                if let Some((rho, phi)) = self.page_pair(page, token, pair) {
                    let relative = page.positions[token] as f32 - query_position as f32;
                    let delta = rope_frequency(pair, self.config.head_dim, rope_theta) * relative;
                    logit += q_mag[pair] * rho * (q_phi[pair] - phi + delta).cos();
                }
            }
            logits.push(logit);
        }
    }

    fn page_pair(&self, page: &Page, token: usize, pair: usize) -> Option<(f32, f32)> {
        if page.tier == HelixPageTier::Hot {
            let x = page.hot_keys[token * self.config.head_dim + 2 * pair];
            let y = page.hot_keys[token * self.config.head_dim + 2 * pair + 1];
            return Some(((x * x + y * y).sqrt(), y.atan2(x)));
        }
        let pairs = self.config.head_dim / 2;
        let index = token * pairs + pair;
        if !get_bit(&page.cold_key.active_mask, index) {
            return None;
        }
        let rho_code = get_nibble(&page.cold_key.rho_codes, index);
        let phi_code = get_nibble(&page.cold_key.phi_codes, index);
        let log_rho =
            page.cold_key.log_rho_min[pair] + page.cold_key.log_rho_step[pair] * rho_code as f32;
        let phi = wrap_angle(
            page.cold_key.mu_phi[pair] + (i32::from(phi_code) - 8) as f32 * (2.0 * PI / 16.0),
        );
        Some((log_rho.exp(), phi))
    }

    fn accumulate_values(&mut self, layer: usize, kv_head: usize, scores: &[f32], out: &mut [f32]) {
        let mut score_index = 0;
        let head_dim = self.config.head_dim;
        for page in self
            .pages
            .iter_mut()
            .filter(|page| page.key.layer == layer && page.key.head == kv_head)
        {
            page.promotion.access_count += 1;
            for token in 0..page.positions.len() {
                let weight = scores[score_index];
                score_index += 1;
                if page.tier == HelixPageTier::Hot {
                    let values = &page.hot_values[token * head_dim..(token + 1) * head_dim];
                    for (acc, value) in out.iter_mut().zip(values) {
                        *acc += weight * value;
                    }
                } else {
                    accumulate_cold_value(page, token, head_dim, weight, out);
                }
            }
        }
    }
}

fn normalize_scores(scores: &mut [f32], head_dim: usize) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut max_score = f32::NEG_INFINITY;
    for score in scores.iter_mut() {
        *score *= scale;
        max_score = max_score.max(*score);
    }
    let mut sum = 0.0;
    for score in scores.iter_mut() {
        *score = (*score - max_score).exp();
        sum += *score;
    }
    for score in scores {
        *score /= sum;
    }
}

fn accumulate_cold_value(page: &Page, token: usize, head_dim: usize, weight: f32, out: &mut [f32]) {
    for group in 0..head_dim / 8 {
        let mut decoded = [0.0; 8];
        let mut inverse = [0.0; 8];
        for (coeff, slot) in decoded.iter_mut().enumerate() {
            let code = get_int3(&page.cold_value.codes, token * head_dim + group * 8 + coeff);
            *slot = (i32::from(code) - 3) as f32 * page.cold_value.scales[group];
        }
        hadamard8(&decoded, &mut inverse);
        for coeff in 0..8 {
            out[group * 8 + coeff] += weight * inverse[coeff] * 0.125;
        }
    }
}

fn add_page_stats(stats: &mut HelixCacheStats, page: &Page) {
    stats.token_count += page.positions.len();
    stats.page_metadata_bytes += page.positions.len() * size_of::<usize>();
    if page.tier == HelixPageTier::Hot {
        stats.hot_pages += 1;
        stats.hot_bytes += (page.hot_keys.len() + page.hot_values.len()) * size_of::<f32>();
        return;
    }
    stats.cold_pages += 1;
    stats.key_metadata_bytes += page.cold_key.mu_phi.len() * size_of::<f32>();
    stats.key_metadata_bytes += page.cold_key.log_rho_min.len() * size_of::<f32>();
    stats.key_metadata_bytes += page.cold_key.log_rho_step.len() * size_of::<f32>();
    stats.value_metadata_bytes += page.cold_value.scales.len() * size_of::<f32>();
    stats.key_bytes += page.cold_key.active_mask.len();
    stats.key_bytes += page.cold_key.rho_codes.len();
    stats.key_bytes += page.cold_key.phi_codes.len();
    stats.value_bytes += page.cold_value.codes.len();
}
