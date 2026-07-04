use super::codec::{HelixCache, Page};
use super::pack::{PI, hadamard8, set_bit, set_int3, set_nibble, wrap_angle};

impl HelixCache {
    pub(super) fn encode_keys(&self, page: &mut Page, pre_rope_keys: &[f32]) {
        let pairs = self.config.head_dim / 2;
        let tokens = page.positions.len();
        for pair in 0..pairs {
            let mut sin_sum = 0.0;
            let mut cos_sum = 0.0;
            let mut min_log = f32::INFINITY;
            let mut max_log = f32::NEG_INFINITY;
            for token in 0..tokens {
                let (rho, phi) = key_pair(pre_rope_keys, self.config.head_dim, token, pair);
                if rho >= self.config.inactive_threshold {
                    sin_sum += phi.sin();
                    cos_sum += phi.cos();
                    let log_rho = (rho + 1.0e-12).ln();
                    min_log = min_log.min(log_rho);
                    max_log = max_log.max(log_rho);
                }
            }
            if !min_log.is_finite() {
                continue;
            }
            let mu = sin_sum.atan2(cos_sum);
            let step = if max_log > min_log {
                (max_log - min_log) / 15.0
            } else {
                0.0
            };
            page.cold_key.mu_phi[pair] = mu;
            page.cold_key.log_rho_min[pair] = min_log;
            page.cold_key.log_rho_step[pair] = step;
            for token in 0..tokens {
                let (rho, phi) = key_pair(pre_rope_keys, self.config.head_dim, token, pair);
                if rho < self.config.inactive_threshold {
                    continue;
                }
                let index = token * pairs + pair;
                set_bit(&mut page.cold_key.active_mask, index, true);
                let log_rho = (rho + 1.0e-12).ln();
                let rho_code = if step == 0.0 {
                    0
                } else {
                    ((log_rho - min_log) / step).round() as i32
                };
                let phase_step = 2.0 * PI / 16.0;
                let phi_code = (wrap_angle(phi - mu) / phase_step).round() as i32 + 8;
                set_nibble(
                    &mut page.cold_key.rho_codes,
                    index,
                    rho_code.clamp(0, 15) as u8,
                );
                set_nibble(
                    &mut page.cold_key.phi_codes,
                    index,
                    phi_code.clamp(0, 15) as u8,
                );
            }
        }
    }

    pub(super) fn encode_values(&self, page: &mut Page, values: &[f32]) {
        let tokens = page.positions.len();
        let groups = self.config.head_dim / 8;
        for group in 0..groups {
            let mut transformed = vec![0.0; tokens * 8];
            let mut max_abs = 0.0f32;
            for token in 0..tokens {
                let start = token * self.config.head_dim + group * 8;
                let mut encoded = [0.0; 8];
                hadamard8(&values[start..start + 8], &mut encoded);
                for coeff in 0..8 {
                    transformed[token * 8 + coeff] = encoded[coeff];
                    max_abs = max_abs.max(encoded[coeff].abs());
                }
            }
            let scale = if max_abs > 0.0 { max_abs / 3.0 } else { 0.0 };
            page.cold_value.scales[group] = scale;
            for token in 0..tokens {
                for coeff in 0..8 {
                    let value = transformed[token * 8 + coeff];
                    let quantized = if scale == 0.0 {
                        0
                    } else {
                        (value / scale).round() as i32
                    };
                    set_int3(
                        &mut page.cold_value.codes,
                        token * self.config.head_dim + group * 8 + coeff,
                        (quantized.clamp(-3, 3) + 3) as u8,
                    );
                }
            }
        }
    }
}

fn key_pair(values: &[f32], head_dim: usize, token: usize, pair: usize) -> (f32, f32) {
    let x = values[token * head_dim + 2 * pair];
    let y = values[token * head_dim + 2 * pair + 1];
    ((x * x + y * y).sqrt(), y.atan2(x))
}
