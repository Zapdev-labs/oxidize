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
        let theta = cfg.layer_rope_theta(layer_idx);
        if !cfg.architecture.uses_alibi() {
            for head in 0..q_heads {
                let off = head * q_head_dim;
                if off + q_head_dim > q.len() {
                    break;
                }
                let q_rope_len = cfg.effective_rope_dim().min(q_head_dim);
                let mut rotated = vec![0.0_f32; q_rope_len];
                cfg.apply_rope_head(
                    &q[off..off + q_rope_len],
                    pos,
                    q_rope_len,
                    theta,
                    &mut rotated,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("rope q: {:?}", e)))?;
                q[off..off + q_rope_len].copy_from_slice(&rotated);
            }
            for head in 0..kv_heads {
                let off = head * kv_head_dim;
                if off + kv_head_dim > k_vec.len() {
                    break;
                }
                let k_rope_len = cfg.effective_rope_dim().min(kv_head_dim);
                let mut rotated = vec![0.0_f32; k_rope_len];
                cfg.apply_rope_head(
                    &k_vec[off..off + k_rope_len],
                    pos,
                    k_rope_len,
                    theta,
                    &mut rotated,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("rope k: {:?}", e)))?;
                k_vec[off..off + k_rope_len].copy_from_slice(&rotated);
            }
        }
        if layer_idx == 3 && pos == 0 && crate::inference::trace_vals_enabled() {
            eprintln!(
                "ATTN L3 h0 pos0: q_postrope[0..6]={:?}",
                &q[..6.min(q.len())]
            );
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
        let layer_window = cfg.layer_sliding_window(layer_idx);
        let skip_tokens = if layer_window > 0 && seq_len > layer_window {
            seq_len - layer_window
        } else {
            0
        };
        let eff_seq_len = seq_len - skip_tokens;
        let kv_skip = skip_tokens.saturating_mul(kv_len);
        let key_cache = &key_cache[kv_skip.min(key_cache.len())..];
        let value_cache = &value_cache[kv_skip.min(value_cache.len())..];

        let mut attn_result = vec![0.0_f32; q_len_used];
        let actual_kv_group_size = q_heads
            .checked_div(kv_heads)
            .filter(|g| *g > 0)
            .unwrap_or(1);
        if cfg.architecture.uses_alibi() {
            for head in 0..q_heads {
                let kv_head = head / actual_kv_group_size;
                let q_head_start = head * q_head_dim;
                let q_head_end = q_head_start + q_head_dim;
                if q_head_end > q.len() {
                    break;
                }
                let q_head = &q[q_head_start..q_head_end];
                let q_head_for_attn = if q_head_dim > kv_head_dim {
                    &q_head[..kv_head_dim]
                } else {
                    q_head
                };
                let write_start = head * kv_head_dim;
                if write_start + kv_head_dim > attn_result.len() {
                    break;
                }
                alibi_attend(
                    q_head_for_attn,
                    key_cache,
                    value_cache,
                    eff_seq_len,
                    kv_head_dim,
                    kv_len,
                    kv_head,
                    alibi_slope(
                        head,
                        if cfg.alibi_num_heads > 0 {
                            cfg.alibi_num_heads
                        } else {
                            q_heads
                        },
                    ),
                    &mut attn_result[write_start..write_start + kv_head_dim],
                );
            }
        } else {
            // Heads are independent; this loop grows linearly with context and
            // serializes ~tens of ms/token at long sequences, so dispatch it
            // through the spin pool. Per-head output slices are disjoint.
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
                    eff_seq_len,
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

fn alibi_attend(
    query: &[f32],
    key_layer: &[f32],
    value_layer: &[f32],
    seq_len: usize,
    head_dim: usize,
    kv_len: usize,
    kv_head: usize,
    slope: f32,
    output: &mut [f32],
) {
    if head_dim == 0 || seq_len == 0 || kv_len == 0 || query.is_empty() {
        output.fill(0.0);
        return;
    }
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut max_s = f32::NEG_INFINITY;
    let mut scores = vec![0.0_f32; seq_len];
    let qn = head_dim.min(query.len());
    for t in 0..seq_len {
        let row = t * kv_len + kv_head * head_dim;
        if row + head_dim > key_layer.len() {
            scores[t] = f32::NEG_INFINITY;
            continue;
        }
        let mut dot = 0.0_f32;
        let k = &key_layer[row..row + qn];
        for i in 0..qn {
            dot += query[i] * k[i];
        }
        let dist = (seq_len - 1 - t) as f32;
        let score = dot * scale + slope * dist;
        scores[t] = score;
        if score > max_s {
            max_s = score;
        }
    }
    if !max_s.is_finite() {
        output.fill(0.0);
        return;
    }
    let mut sum = 0.0_f32;
    for score in &mut scores {
        if !score.is_finite() {
            *score = 0.0;
            continue;
        }
        *score = (*score - max_s).exp();
        sum += *score;
    }
    let inv = 1.0_f32 / sum.max(1e-12);
    let out_n = output.len().min(head_dim);
    output[..out_n].fill(0.0);
    for t in 0..seq_len {
        let row = t * kv_len + kv_head * head_dim;
        if row + head_dim > value_layer.len() {
            continue;
        }
        let w = scores[t] * inv;
        let v = &value_layer[row..row + out_n];
        for i in 0..out_n {
            output[i] += w * v[i];
        }
    }
}

impl LayerWiseModel {
    /// DeepSeek / GLM-DSA multi-head latent attention.
    /// Returns the attention residual (not yet added to `x`).
    pub(super) fn run_mla_layer(
        &mut self,
        layer_idx: usize,
        layer: &LayerWeights,
        x: &[f32],
        pos: usize,
        cfg: &InferenceConfig,
    ) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads.max(1);
        let kv_lora = layer.mla_kv_a_norm.len();
        let q_lora = layer.mla_q_a.output_dim(h);
        let q_len = layer.mla_q_b.output_dim(q_lora);
        let k_head_dim = cfg.kv_head_dim();
        let kv_out = layer.mla_kv_a_mqa.output_dim(h);
        let kv_pe_dim = kv_out.saturating_sub(kv_lora);
        let k_nope_dim = layer.mla_k_b.output_dim(kv_lora) / n_heads;
        let v_head_dim = layer.mla_v_b.output_dim(kv_lora) / n_heads;
        let q_pe_dim = k_head_dim.saturating_sub(k_nope_dim);
        if kv_lora == 0 || q_lora == 0 || q_len == 0 || k_head_dim == 0 || kv_out < kv_lora {
            return Err(ModelError::InferenceFailed(
                "mla: missing latent projections".to_owned(),
            ));
        }
        if v_head_dim > k_head_dim {
            return Err(ModelError::InferenceFailed(
                "mla: value head wider than the KV cache slot".to_owned(),
            ));
        }

        let mut normed = vec![0.0_f32; h];
        rms_norm_model(x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed, cfg)?;

        let mut c_q = vec![0.0_f32; q_lora];
        gemv_weight(&layer.mla_q_a, q_lora, h, &normed, &mut c_q)
            .map_err(|e| ModelError::InferenceFailed(format!("mla q_a: {e}")))?;
        if !layer.mla_q_a_norm.is_empty() {
            let src = c_q.clone();
            rms_norm_model(&src, &layer.mla_q_a_norm, cfg.rms_norm_eps, &mut c_q, cfg)?;
        }

        let mut q = vec![0.0_f32; q_len];
        gemv_weight(&layer.mla_q_b, q_len, q_lora, &c_q, &mut q)
            .map_err(|e| ModelError::InferenceFailed(format!("mla q_b: {e}")))?;

        let mut kv_pe = vec![0.0_f32; kv_out];
        gemv_weight(&layer.mla_kv_a_mqa, kv_out, h, &normed, &mut kv_pe)
            .map_err(|e| ModelError::InferenceFailed(format!("mla kv_a: {e}")))?;

        let mut c_kv = kv_pe[..kv_lora].to_vec();
        if kv_lora == layer.mla_kv_a_norm.len() {
            let src = c_kv.clone();
            rms_norm_model(&src, &layer.mla_kv_a_norm, cfg.rms_norm_eps, &mut c_kv, cfg)?;
        }

        let mut k_pe_rope = vec![0.0_f32; kv_pe_dim];
        if kv_pe_dim > 0 {
            cfg.apply_rope_head(
                &kv_pe[kv_lora..kv_lora + kv_pe_dim],
                pos,
                kv_pe_dim,
                cfg.rope_theta,
                &mut k_pe_rope,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("mla k_pe rope: {e:?}")))?;
        }

        let total_k = n_heads * k_head_dim;
        let mut k_store = vec![0.0_f32; total_k];
        let mut v_store = vec![0.0_f32; n_heads * v_head_dim.max(1)];
        for head in 0..n_heads {
            let k_off = head * k_head_dim;
            if k_nope_dim > 0 {
                gemv_weight_head(
                    &layer.mla_k_b,
                    k_nope_dim,
                    kv_lora,
                    head,
                    n_heads,
                    &c_kv,
                    &mut k_store[k_off..k_off + k_nope_dim],
                )
                .map_err(|e| ModelError::InferenceFailed(format!("mla k_b h{head}: {e}")))?;
            }
            let copy = q_pe_dim
                .min(kv_pe_dim)
                .min(k_head_dim.saturating_sub(k_nope_dim));
            if copy > 0 {
                let rope_off = k_off + k_nope_dim;
                k_store[rope_off..rope_off + copy].copy_from_slice(&k_pe_rope[..copy]);
            }
            if v_head_dim > 0 {
                let v_off = head * v_head_dim;
                mla_v_b_head(
                    &layer.mla_v_b,
                    kv_lora,
                    v_head_dim,
                    head,
                    n_heads,
                    &c_kv,
                    &mut v_store[v_off..v_off + v_head_dim],
                )
                .map_err(|e| ModelError::InferenceFailed(format!("mla v_b h{head}: {e}")))?;
            }
            let q_off = head * k_head_dim;
            if q_pe_dim > 0 && q_off + k_head_dim <= q.len() {
                let mut rotated = vec![0.0_f32; q_pe_dim];
                cfg.apply_rope_head(
                    &q[q_off + k_nope_dim..q_off + k_head_dim],
                    pos,
                    q_pe_dim,
                    cfg.rope_theta,
                    &mut rotated,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("mla q_pe: {e:?}")))?;
                let q_pe = &mut q[q_off + k_nope_dim..q_off + k_head_dim];
                q_pe.copy_from_slice(&rotated[..q_pe.len()]);
            }
        }

        let mut v_padded = vec![0.0_f32; total_k];
        for head in 0..n_heads {
            let copy = v_head_dim.min(k_head_dim);
            if copy == 0 {
                continue;
            }
            let v_off = head * v_head_dim;
            let k_off = head * k_head_dim;
            v_padded[k_off..k_off + copy].copy_from_slice(&v_store[v_off..v_off + copy]);
        }
        self.kv_cache
            .set(layer_idx, pos, &k_store, &v_padded)
            .map_err(|e| ModelError::InferenceFailed(format!("mla kv set: {e:?}")))?;

        let seq_len = pos + 1;
        let borrowed_keys = self
            .kv_cache
            .f32_layer_key_prefix(layer_idx, seq_len)
            .map_err(|e| ModelError::InferenceFailed(format!("mla kv keys: {e:?}")))?;
        let borrowed_values = self
            .kv_cache
            .f32_layer_value_prefix(layer_idx, seq_len)
            .map_err(|e| ModelError::InferenceFailed(format!("mla kv values: {e:?}")))?;
        let (key_cache, value_cache) = match (borrowed_keys, borrowed_values) {
            (Some(keys), Some(values)) => (
                AttentionCacheSlice::Borrowed(keys),
                AttentionCacheSlice::Borrowed(values),
            ),
            _ => {
                let mut keys = vec![0.0_f32; seq_len * total_k];
                let mut values = vec![0.0_f32; seq_len * total_k];
                self.kv_cache
                    .copy_layer_keys(layer_idx, seq_len, &mut keys)
                    .map_err(|e| ModelError::InferenceFailed(format!("mla copy keys: {e:?}")))?;
                self.kv_cache
                    .copy_layer_values(layer_idx, seq_len, &mut values)
                    .map_err(|e| ModelError::InferenceFailed(format!("mla copy values: {e:?}")))?;
                (
                    AttentionCacheSlice::Owned(keys),
                    AttentionCacheSlice::Owned(values),
                )
            }
        };
        let key_cache = key_cache.as_slice();
        let value_cache = value_cache.as_slice();

        let mut attn_result = vec![0.0_f32; total_k];
        let scale = 1.0_f32 / (k_head_dim as f32).sqrt();
        for head in 0..n_heads {
            let off = head * k_head_dim;
            if off + k_head_dim > q.len() {
                break;
            }
            let q_h = &q[off..off + k_head_dim];
            let mut scores = vec![0.0_f32; seq_len];
            let mut max_s = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let row = t * total_k + off;
                if row + k_head_dim > key_cache.len() {
                    scores[t] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0_f32;
                let k_t = &key_cache[row..row + k_head_dim];
                for i in 0..k_head_dim {
                    dot += q_h[i] * k_t[i];
                }
                let score = dot * scale;
                scores[t] = score;
                if score > max_s {
                    max_s = score;
                }
            }
            if !max_s.is_finite() {
                continue;
            }
            let mut sum = 0.0_f32;
            for s in &mut scores {
                if !s.is_finite() {
                    *s = 0.0;
                    continue;
                }
                *s = (*s - max_s).exp();
                sum += *s;
            }
            let inv = 1.0_f32 / sum.max(1e-12);
            for i in 0..v_head_dim {
                let mut acc = 0.0_f32;
                for t in 0..seq_len {
                    let row = t * total_k + off + i;
                    if row < value_cache.len() {
                        acc += scores[t] * inv * value_cache[row];
                    }
                }
                attn_result[off + i] = acc;
            }
        }

        let attn_in_len = layer.attn_output.output_dim(h);
        let attn_input = if attn_in_len > 0 && attn_result.len() >= attn_in_len {
            &attn_result[..attn_in_len]
        } else {
            &attn_result[..total_k.min(attn_result.len())]
        };
        let mut attn_out = vec![0.0_f32; h];
        gemv_weight(
            &layer.attn_output,
            h,
            attn_input.len(),
            attn_input,
            &mut attn_out,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mla attn_out: {e}")))?;
        Ok(attn_out)
    }
}

fn gemv_weight_head(
    storage: &WeightStorage,
    rows: usize,
    cols: usize,
    head: usize,
    n_heads: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    if n_heads == 0 {
        return Err("n_heads is zero".to_string());
    }
    match storage {
        WeightStorage::F32(data) => {
            let per_head = data.len() / n_heads;
            let start = head * per_head;
            let end = start + per_head;
            if end > data.len() {
                return Err("head slice out of range".to_string());
            }
            gemv_f32(&data[start..end], rows, cols, input, output).map_err(|e| format!("{e:?}"))
        }
        WeightStorage::Quantized(qtype, data) => {
            let (block_width, block_size) = quant_block_layout(*qtype).unwrap_or((1, 4));
            let blocks_per_row = cols / block_width;
            let per_head = rows * blocks_per_row * block_size;
            let start = head * per_head;
            let end = start + per_head;
            gemv_quantized_f32(*qtype, &data[start..end], rows, cols, input, output)
                .map_err(|e| format!("{e:?}"))
        }
        WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
            let data = &mmap[*offset..*offset + *size];
            let (block_width, block_size) = quant_block_layout(*qtype).unwrap_or((1, 4));
            let blocks_per_row = cols / block_width;
            let per_head = rows * blocks_per_row * block_size;
            let start = head * per_head;
            let end = start + per_head;
            gemv_quantized_f32(*qtype, &data[start..end], rows, cols, input, output)
                .map_err(|e| format!("{e:?}"))
        }
    }
}

fn mla_v_b_head(
    storage: &WeightStorage,
    kv_lora: usize,
    v_dim: usize,
    head: usize,
    n_heads: usize,
    kv_cmpr: &[f32],
    out: &mut [f32],
) -> Result<(), String> {
    out.fill(0.0);
    if let WeightStorage::F32(data) = storage {
        for v in 0..v_dim {
            let mut sum = 0.0_f32;
            for l in 0..kv_lora {
                let idx = l * v_dim * n_heads + v * n_heads + head;
                if idx < data.len() {
                    sum += data[idx] * kv_cmpr[l];
                }
            }
            out[v] = sum;
        }
        return Ok(());
    }
    let per_head_elems = kv_lora * v_dim;
    let mut w_host = vec![0.0_f32; per_head_elems];
    match storage {
        WeightStorage::Quantized(qtype, data) => {
            let per_head_bytes = data.len() / n_heads.max(1);
            let start = head * per_head_bytes;
            let end = (head + 1) * per_head_bytes;
            if end > data.len() {
                return Err(format!("v_b head {head} out of range"));
            }
            dequantize_scalar(*qtype, &data[start..end], &mut w_host)
                .map_err(|e| format!("dequant v_b: {e:?}"))?;
        }
        WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
            let data = &mmap[*offset..*offset + *size];
            let per_head_bytes = data.len() / n_heads.max(1);
            let start = head * per_head_bytes;
            let end = (head + 1) * per_head_bytes;
            if end > data.len() {
                return Err(format!("v_b head {head} out of range"));
            }
            dequantize_scalar(*qtype, &data[start..end], &mut w_host)
                .map_err(|e| format!("dequant v_b: {e:?}"))?;
        }
        WeightStorage::F32(_) => {}
    }
    for v in 0..v_dim {
        let mut sum = 0.0_f32;
        for l in 0..kv_lora {
            sum += w_host[l * v_dim + v] * kv_cmpr[l];
        }
        out[v] = sum;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mla_v_b_head_uses_latent_major_layout() {
        let n_heads = 2;
        let kv_lora = 2;
        let v_dim = 2;
        let mut data = vec![0.0_f32; kv_lora * v_dim * n_heads];
        for l in 0..kv_lora {
            for v in 0..v_dim {
                let idx = l * v_dim * n_heads + v * n_heads + 1;
                data[idx] = (l + 1) as f32;
            }
        }
        let mut out = [0.0_f32; 2];
        mla_v_b_head(
            &WeightStorage::F32(data),
            kv_lora,
            v_dim,
            1,
            n_heads,
            &[1.0, 1.0],
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [3.0, 3.0]);
    }
}
