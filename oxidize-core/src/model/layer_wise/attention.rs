use super::*;

pub(super) enum AttentionCacheSlice<'a> {
    Borrowed(&'a [f32]),
    Owned(Vec<f32>),
}

impl<'a> AttentionCacheSlice<'a> {
    pub(super) fn as_slice(&'a self) -> &'a [f32] {
        match self {
            Self::Borrowed(data) => data,
            Self::Owned(data) => data,
        }
    }
}

impl LayerWiseModel {
    pub(super) fn run_attention_layer(
        &mut self,
        layer_idx: usize,
        layer: &LayerWeights,
        x: &[f32],
        pos: usize,
        cfg: &InferenceConfig,
    ) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let k = cfg.num_key_value_heads;
        let mut attn_out = vec![0.0_f32; h];

        let mut normed = vec![0.0_f32; h];
        rms_norm_model(x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed, cfg)?;

        let q_len = weight_output_dim(&layer.attn_q, h);
        let kv_len = if !weight_is_empty(&layer.attn_k) {
            weight_output_dim(&layer.attn_k, h)
        } else {
            0
        };
        let attn_output_input_len = if !weight_is_empty(&layer.attn_output) {
            weight_output_dim(&layer.attn_output, h)
        } else {
            0
        };

        // Q/K/V projections run sequentially; each GEMV is internally
        // parallel (spin pool), and a rayon::join here would wake the rayon
        // workers mid-decode, contending with the pinned spin workers.
        let mut q_full = vec![0.0_f32; q_len];
        let mut k_vec = vec![0.0_f32; kv_len];
        let mut v_vec = vec![0.0_f32; kv_len];
        gemv_weight(&layer.attn_q, q_len, h, &normed, &mut q_full)
            .map_err(|e| ModelError::InferenceFailed(format!("attn_q: {:?}", e)))?;
        if !weight_is_empty(&layer.attn_k) {
            gemv_weight(&layer.attn_k, kv_len, h, &normed, &mut k_vec)
                .map_err(|e| ModelError::InferenceFailed(format!("attn_k: {:?}", e)))?;
        }
        if !weight_is_empty(&layer.attn_v) {
            gemv_weight(&layer.attn_v, kv_len, h, &normed, &mut v_vec)
                .map_err(|e| ModelError::InferenceFailed(format!("attn_v: {:?}", e)))?;
        }
        if !layer.attn_q_bias.is_empty() {
            for (i, q) in q_full.iter_mut().enumerate() {
                *q += layer.attn_q_bias[i % layer.attn_q_bias.len()];
            }
        }
        if !layer.attn_k_bias.is_empty() {
            for (i, k) in k_vec.iter_mut().enumerate() {
                *k += layer.attn_k_bias[i % layer.attn_k_bias.len()];
            }
        }
        if !layer.attn_v_bias.is_empty() {
            for (i, v) in v_vec.iter_mut().enumerate() {
                *v += layer.attn_v_bias[i % layer.attn_v_bias.len()];
            }
        }

        let kv_head_dim = if k > 0 && kv_len % k == 0 {
            kv_len / k
        } else if kv_len > 0 {
            kv_len
        } else {
            cfg.kv_head_dim()
        };

        let q_len_used_guess = if attn_output_input_len > 0 && q_len == 2 * attn_output_input_len {
            q_len / 2
        } else if attn_output_input_len > 0 {
            q_len.min(attn_output_input_len)
        } else if q_len > h {
            h
        } else {
            q_len
        };
        let q_heads_guess = if n > 0 && q_len_used_guess.is_multiple_of(n) {
            n
        } else {
            1
        };
        let q_head_dim_guess = if q_heads_guess > 0 {
            q_len_used_guess / q_heads_guess
        } else {
            q_len_used_guess
        };

        let (mut q, attn_gate) = if attn_output_input_len > 0 && q_len == 2 * attn_output_input_len
        {
            let (query, gate) =
                split_gated_query_proj(&q_full, q_head_dim_guess).ok_or_else(|| {
                    ModelError::InferenceFailed("gated q_proj split failed".to_owned())
                })?;
            (query, Some(gate))
        } else {
            (q_full[..q_len_used_guess].to_vec(), None)
        };

        if crate::inference::trace_fwd_enabled() {
            let s = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
            eprintln!(
                "STAGE lw pos={pos} layer={layer_idx} normed={:.6e} q={:.6e} k={:.6e} v={:.6e} x={:.6e} nw_len={} nw={:.6e}",
                s(&normed),
                s(&q),
                s(&k_vec),
                s(&v_vec),
                s(x),
                layer.attn_norm.len(),
                s(&layer.attn_norm)
            );
        }
        let q_len_used = q.len();
        let q_head_dim = if n > 0 && q_len_used.is_multiple_of(n) {
            q_len_used / n
        } else {
            q_len_used
        };
        let q_heads = q_len_used.checked_div(q_head_dim).unwrap_or(1);
        let kv_heads = kv_len.checked_div(kv_head_dim).unwrap_or(1);

        if !layer.attn_q_norm.is_empty() && q_head_dim == layer.attn_q_norm.len() {
            for head in 0..q_heads {
                let start = head * q_head_dim;
                let end = start + q_head_dim;
                if end > q.len() {
                    break;
                }
                let mut normed_head = vec![0.0_f32; q_head_dim];
                rms_norm_model(
                    &q[start..end],
                    &layer.attn_q_norm,
                    cfg.rms_norm_eps,
                    &mut normed_head,
                    cfg,
                )?;
                q[start..end].copy_from_slice(&normed_head);
            }
        }
        if !layer.attn_k_norm.is_empty() && kv_head_dim == layer.attn_k_norm.len() {
            for head in 0..kv_heads {
                let start = head * kv_head_dim;
                let end = start + kv_head_dim;
                if end > k_vec.len() {
                    break;
                }
                let mut normed_head = vec![0.0_f32; kv_head_dim];
                rms_norm_model(
                    &k_vec[start..end],
                    &layer.attn_k_norm,
                    cfg.rms_norm_eps,
                    &mut normed_head,
                    cfg,
                )?;
                k_vec[start..end].copy_from_slice(&normed_head);
            }
        }

        if layer_idx == 3 && pos == 0 && crate::inference::trace_vals_enabled() {
            eprintln!(
                "ATTN L3 h0 pos0: q_prerope[0..6]={:?} q_head_dim={q_head_dim} rope_len={}",
                &q[..6.min(q.len())],
                cfg.effective_rope_dim().min(q_head_dim),
            );
        }
        for head in 0..q_heads {
            let off = head * q_head_dim;
            if off + q_head_dim > q.len() {
                break;
            }
            let q_rope_len = cfg.effective_rope_dim().min(q_head_dim);
            let mut rotated = vec![0.0_f32; q_rope_len];
            apply_rope_f32(
                &q[off..off + q_rope_len],
                pos,
                q_rope_len,
                cfg.rope_theta,
                &mut rotated,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("rope q: {:?}", e)))?;
            q[off..off + q_rope_len].copy_from_slice(&rotated);
        }
        if layer_idx == 3 && pos == 0 && crate::inference::trace_vals_enabled() {
            eprintln!(
                "ATTN L3 h0 pos0: q_postrope[0..6]={:?}",
                &q[..6.min(q.len())]
            );
        }
        for head in 0..kv_heads {
            let off = head * kv_head_dim;
            if off + kv_head_dim > k_vec.len() {
                break;
            }
            let k_rope_len = cfg.effective_rope_dim().min(kv_head_dim);
            let mut rotated = vec![0.0_f32; k_rope_len];
            apply_rope_f32(
                &k_vec[off..off + k_rope_len],
                pos,
                k_rope_len,
                cfg.rope_theta,
                &mut rotated,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("rope k: {:?}", e)))?;
            k_vec[off..off + k_rope_len].copy_from_slice(&rotated);
        }

        self.kv_cache
            .set(layer_idx, pos, &k_vec, &v_vec)
            .map_err(|e| ModelError::InferenceFailed(format!("kv set: {:?}", e)))?;

        let seq_len = pos + 1;
        let borrowed_key_cache = self
            .kv_cache
            .f32_layer_key_prefix(layer_idx, seq_len)
            .map_err(|e| ModelError::InferenceFailed(format!("kv borrow keys: {:?}", e)))?;
        let borrowed_value_cache = self
            .kv_cache
            .f32_layer_value_prefix(layer_idx, seq_len)
            .map_err(|e| ModelError::InferenceFailed(format!("kv borrow values: {:?}", e)))?;
        let (key_cache, value_cache) = match (borrowed_key_cache, borrowed_value_cache) {
            (Some(keys), Some(values)) => (
                AttentionCacheSlice::Borrowed(keys),
                AttentionCacheSlice::Borrowed(values),
            ),
            _ => {
                let mut key_cache = vec![0.0_f32; seq_len * kv_len];
                let mut value_cache = vec![0.0_f32; seq_len * kv_len];
                self.kv_cache
                    .copy_layer_keys(layer_idx, seq_len, &mut key_cache)
                    .map_err(|e| ModelError::InferenceFailed(format!("kv copy keys: {:?}", e)))?;
                self.kv_cache
                    .copy_layer_values(layer_idx, seq_len, &mut value_cache)
                    .map_err(|e| ModelError::InferenceFailed(format!("kv copy values: {:?}", e)))?;
                (
                    AttentionCacheSlice::Owned(key_cache),
                    AttentionCacheSlice::Owned(value_cache),
                )
            }
        };
        let key_cache = key_cache.as_slice();
        let value_cache = value_cache.as_slice();

        let mut attn_result = vec![0.0_f32; q_len_used];
        let actual_kv_group_size = q_heads
            .checked_div(kv_heads)
            .filter(|g| *g > 0)
            .unwrap_or(1);
        // Heads are independent; this loop grows linearly with context and
        // serializes ~tens of ms/token at long sequences, so dispatch it
        // through the spin pool. Per-head output slices are disjoint.
        {
            let attn_failed = std::sync::atomic::AtomicBool::new(false);
            let out_base = attn_result.as_mut_ptr() as usize;
            let attn_len = attn_result.len();
            let q_ref = &q;
            crate::spinpool::run_chunks(q_heads, |head| {
                let kv_head = head / actual_kv_group_size;
                let q_head_start = head * q_head_dim;
                let q_head_end = q_head_start + q_head_dim;
                if q_head_end > q_ref.len() {
                    return;
                }
                let q_head = &q_ref[q_head_start..q_head_end];
                let q_head_for_attn = if q_head_dim > kv_head_dim {
                    &q_head[..kv_head_dim]
                } else {
                    q_head
                };
                let write_start = head * kv_head_dim;
                if write_start + kv_head_dim > attn_len {
                    return;
                }
                // SAFETY: per-head output ranges are disjoint; attn_result outlives dispatch.
                let out_head = unsafe {
                    std::slice::from_raw_parts_mut(
                        (out_base as *mut f32).add(write_start),
                        kv_head_dim,
                    )
                };
                if flash_attention_decode_f32(
                    q_head_for_attn,
                    key_cache,
                    value_cache,
                    seq_len,
                    kv_head_dim,
                    kv_len,
                    kv_head,
                    out_head,
                )
                .is_err()
                {
                    attn_failed.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            });
            if attn_failed.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(ModelError::InferenceFailed(
                    "flash attention failed".to_owned(),
                ));
            }
        }

        let mut attn_input =
            if attn_output_input_len > 0 && attn_result.len() != attn_output_input_len {
                if attn_result.len() >= attn_output_input_len {
                    attn_result[..attn_output_input_len].to_vec()
                } else {
                    let mut padded = vec![0.0_f32; attn_output_input_len];
                    padded[..attn_result.len()].copy_from_slice(&attn_result);
                    padded
                }
            } else {
                attn_result
            };

        if let Some(gate) = attn_gate {
            for (out, g) in attn_input.iter_mut().zip(gate.iter()) {
                *out *= sigmoid(*g);
            }
        }

        if !weight_is_empty(&layer.attn_output) && attn_output_input_len > 0 {
            gemv_weight(
                &layer.attn_output,
                h,
                attn_output_input_len,
                &attn_input,
                &mut attn_out,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("attn_output: {:?}", e)))?;
            if !layer.attn_output_bias.is_empty() {
                for (i, out) in attn_out.iter_mut().enumerate() {
                    *out += layer.attn_output_bias[i % layer.attn_output_bias.len()];
                }
            }
        }

        Ok(attn_out)
    }
}
