use super::*;

/// Golden-logits oracle (env `OX_GOLDEN_LOGITS`). When the env var is set, every
/// token's logits are summarised (argmax + top-5 `(idx, logit)`) and appended to
/// the file named by the env value (or stderr when the value is empty). This is a
/// cheap, architecture-independent reference for diffing the CPU path against the
/// on-device attention path (env `OX_GPU_ATTN`); it is OFF by default and adds a
/// single relaxed bool load to the hot path when disabled.
#[derive(Clone)]
enum GoldenSink {
    Disabled,
    Stderr,
    File(std::path::PathBuf),
}

fn golden_sink() -> &'static GoldenSink {
    static SINK: std::sync::OnceLock<GoldenSink> = std::sync::OnceLock::new();
    SINK.get_or_init(|| match std::env::var("OX_GOLDEN_LOGITS") {
        Ok(p) if !p.is_empty() => GoldenSink::File(std::path::PathBuf::from(p)),
        Ok(_) => GoldenSink::Stderr,
        Err(_) => GoldenSink::Disabled,
    })
}

/// Append one `GOLDEN tok=... argmax=... logit=... top5=[...]` line for `logits`.
/// No-op unless `OX_GOLDEN_LOGITS` is set. Kept off the hot path (cold).
#[cold]
fn dump_golden_logits(logits: &[f32]) {
    let sink = golden_sink();
    if matches!(sink, GoldenSink::Disabled) || logits.is_empty() {
        return;
    }
    static TOK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let tok = TOK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // argmax over the full logit vector.
    let mut argmax = 0usize;
    let mut argmax_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > argmax_v {
            argmax_v = v;
            argmax = i;
        }
    }

    // top-5 by partial selection (no full vocab sort).
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = 5.min(idx.len());
    if k > 0 && idx.len() > k {
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let mut top: Vec<usize> = idx.into_iter().take(k).collect();
    top.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut top5 = String::from("[");
    for (j, &i) in top.iter().enumerate() {
        if j > 0 {
            top5.push(',');
        }
        top5.push_str(&format!("({},{:.6e})", i, logits[i]));
    }
    top5.push(']');

    let line = format!(
        "GOLDEN tok={} argmax={} logit={:.6e} top5={}\n",
        tok, argmax, argmax_v, top5
    );
    match sink {
        GoldenSink::File(path) => {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = f.write_all(line.as_bytes());
            }
        }
        GoldenSink::Stderr => {
            eprint!("{line}");
        }
        GoldenSink::Disabled => {}
    }
}

impl InferenceModel {
    pub fn forward_tokens_no_logits(
        &mut self,
        tokens: &[Token],
        session: &mut Session,
    ) -> Result<(), ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::EmptyInput);
        }
        let requested_total = session.consumed_tokens().saturating_add(tokens.len());
        if requested_total > self.config.context_size {
            return Err(ModelError::ContextExceeded {
                context_size: self.config.context_size,
                requested_total_tokens: requested_total,
            });
        }
        let start_pos = session.consumed_tokens();
        // On-device attention (default ON with gpu_native) owns the device F16 KV
        // cache; batched prefill only warms the HOST cache. Force per-token prefill
        // so each position is appended via `launch_kv_append_f16`. Set OX_GPU_ATTN=0
        // to allow batched prefill on the legacy CPU-attention path.
        let use_batched = tokens.len() > 1
            && self.layers_supported_for_batched()
            && !super::layers::ox_gpu_attn_enabled();
        if use_batched {
            self.forward_batched(tokens, start_pos, false)?;
        } else {
            for (i, &token) in tokens.iter().enumerate() {
                self.forward_single(token, start_pos + i, false)?;
            }
        }
        session.record_tokens(tokens.len());
        Ok(())
    }

    /// True when all layers use the standard attention + FFN path that
    /// [`forward_batched`] can handle. Mamba/SSM and MoE layers need per-token
    /// state and aren't supported by the batched path yet.
    pub(super) fn layers_supported_for_batched(&self) -> bool {
        if self.layers.is_empty() {
            return false;
        }
        let mut attention_widths: Option<(usize, usize, usize)> = None;
        for layer in &self.layers {
            let is_mamba = !layer.attn_qkv.is_empty() && layer.attn_q.is_empty();
            if is_mamba {
                return false;
            }
            let is_moe = !layer.ffn_gate_exps.is_empty()
                || !layer.ffn_up_exps.is_empty()
                || !layer.ffn_down_exps.is_empty()
                || !layer.ffn_gate_inp.is_empty();
            if is_moe {
                return false;
            }
            // No standard attention → can't batch the layer (degenerate case).
            if layer.attn_q.is_empty() {
                return false;
            }
            let widths = (
                layer.attn_q.output_dim(self.config.hidden_size),
                layer.attn_k.output_dim(self.config.hidden_size),
                layer.attn_v.output_dim(self.config.hidden_size),
            );
            if let Some(first_widths) = attention_widths {
                if widths != first_widths {
                    return false;
                }
            } else {
                attention_widths = Some(widths);
            }
        }
        true
    }

    /// Batched prefill: process `tokens` in one shot, sharing each weight
    /// matrix across all batch positions via [`gemm_weight`]. Per-token work
    /// (per-head Q/K norms, RoPE, KV cache writes, attention) stays in a
    /// position loop — only the matmuls are batched, but those dominate.
    #[allow(clippy::too_many_lines)]
    pub(super) fn forward_batched(
        &mut self,
        tokens: &[Token],
        start_pos: usize,
        need_logits: bool,
    ) -> Result<Option<Vec<Logits>>, ModelError> {
        let batch = tokens.len();
        debug_assert!(batch >= 1);
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let kvh = cfg.num_key_value_heads;

        // 1. Embedding lookup for every batch position into x_batch[batch, h].
        let mut x_batch = vec![0.0_f32; batch * h];
        for (i, &token) in tokens.iter().enumerate() {
            let token_idx = (token as usize).min(cfg.vocab_size.saturating_sub(1));
            let target = &mut x_batch[i * h..(i + 1) * h];
            match &self.tok_embeddings {
                WeightStorage::F32(data) => {
                    let row = &data[token_idx * h..(token_idx + 1) * h];
                    target.copy_from_slice(row);
                }
                WeightStorage::Quantized(qtype, data) => {
                    lookup_quantized_embedding(h, *qtype, data, token_idx, target);
                }
                WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                    let data = &mmap[*offset..*offset + *size];
                    lookup_quantized_embedding(h, *qtype, data, token_idx, target);
                }
            }
            if cfg.embedding_scale != 1.0 {
                for v in target.iter_mut() {
                    *v *= cfg.embedding_scale;
                }
            }
        }

        // Scratch buffers reused across layers. Allocated once per call (batched
        // prefill is not in the per-token hot path, so this is fine).
        let layer0 = &self.layers[0];
        let q_len = layer0.attn_q.output_dim(h);
        let kv_len = if !layer0.attn_k.is_empty() {
            layer0.attn_k.output_dim(h)
        } else {
            0
        };
        let aoil0 = if !layer0.attn_output.is_empty() {
            layer0.attn_output.output_dim(h)
        } else {
            0
        };
        let q_len_used0 = if aoil0 > 0 {
            q_len.min(aoil0)
        } else if q_len > h {
            h
        } else {
            q_len
        };
        let q_head_dim = if n > 0 && q_len_used0.is_multiple_of(n) {
            q_len_used0 / n
        } else {
            q_len_used0
        };
        let kv_head_dim = if kvh > 0 && kv_len > 0 && kv_len.is_multiple_of(kvh) {
            kv_len / kvh
        } else if kv_len > 0 {
            kv_len
        } else {
            q_head_dim
        };
        let i_size = cfg.intermediate_size;

        let mut normed_batch = vec![0.0_f32; batch * h];
        let mut q_batch = vec![0.0_f32; batch * q_len];
        let mut k_batch = vec![0.0_f32; batch * kv_len.max(1)];
        let mut v_batch = vec![0.0_f32; batch * kv_len.max(1)];
        let mut attn_result_batch = vec![0.0_f32; batch * q_len_used0];
        let mut attn_proj_batch = vec![0.0_f32; batch * h];
        let mut gate_batch = vec![0.0_f32; batch * i_size];
        let mut up_batch = vec![0.0_f32; batch * i_size];
        let mut ffn_out_batch = vec![0.0_f32; batch * h];
        let mut head_scratch = vec![0.0_f32; q_head_dim.max(kv_head_dim)];
        let mut qk_norm_scratch = vec![0.0_f32; q_len.max(kv_len)];

        for layer_idx in 0..cfg.layer_count {
            let layer = &self.layers[layer_idx];
            // Map to the KV cache layer index (only attention layers are stored).
            let kv_layer_idx = self
                .kv_layer_map
                .get(layer_idx)
                .copied()
                .flatten()
                .unwrap_or(layer_idx);

            let ffn_norm_weight: &[f32] = if cfg.sandwich_norm {
                // Gemma: post_attention_norm is a sandwich norm (applied to the
                // attention output), NOT the pre-FFN norm. Use ffn_norm here.
                &layer.ffn_norm
            } else if !layer.post_attention_norm.is_empty() {
                &layer.post_attention_norm
            } else if !layer.ffn_norm.is_empty() {
                &layer.ffn_norm
            } else {
                &[]
            };

            // Per-layer RoPE theta and sliding-window size (Gemma interleaves
            // local SWA and global attention layers with distinct RoPE bases).
            let layer_rope = cfg.layer_rope_theta(layer_idx);
            let layer_window = cfg.layer_sliding_window(layer_idx);

            // 2. Per-token attn RMSNorm into normed_batch.
            for i in 0..batch {
                rms_norm_f32(
                    &x_batch[i * h..(i + 1) * h],
                    &layer.attn_norm,
                    cfg.rms_norm_eps,
                    &mut normed_batch[i * h..(i + 1) * h],
                )
                .map_err(|e| ModelError::InferenceFailed(format!("rms_norm: {:?}", e)))?;
            }

            // 3. Batched Q/K/V via GEMM — the main win over per-token GEMV.
            gemm_weight(&layer.attn_q, q_len, h, &normed_batch, &mut q_batch, batch)
                .map_err(|e| ModelError::InferenceFailed(format!("attn_q: {:?}", e)))?;
            if !layer.attn_q_bias.is_empty() {
                add_repeating_bias(&mut q_batch, &layer.attn_q_bias);
            }
            if kv_len > 0 {
                gemm_weight(
                    &layer.attn_k,
                    kv_len,
                    h,
                    &normed_batch,
                    &mut k_batch[..batch * kv_len],
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("attn_k: {:?}", e)))?;
                if !layer.attn_k_bias.is_empty() {
                    add_repeating_bias(&mut k_batch[..batch * kv_len], &layer.attn_k_bias);
                }
                gemm_weight(
                    &layer.attn_v,
                    kv_len,
                    h,
                    &normed_batch,
                    &mut v_batch[..batch * kv_len],
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("attn_v: {:?}", e)))?;
                if !layer.attn_v_bias.is_empty() {
                    add_repeating_bias(&mut v_batch[..batch * kv_len], &layer.attn_v_bias);
                }
            }

            if trace_fwd_enabled() {
                let s = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
                for t in 0..batch {
                    eprintln!(
                        "STAGE inf pos={} layer={layer_idx} normed={:.6e} q={:.6e} k={:.6e} v={:.6e}",
                        start_pos + t,
                        s(&normed_batch[t * h..(t + 1) * h]),
                        s(&q_batch[t * q_len..(t + 1) * q_len]),
                        s(&k_batch[t * kv_len..(t + 1) * kv_len]),
                        s(&v_batch[t * kv_len..(t + 1) * kv_len])
                    );
                }
            }
            let q_heads = q_len_used0 / q_head_dim.max(1);
            let kv_heads = kv_len.checked_div(kv_head_dim).unwrap_or(0);
            let mut key_copy_buf: Vec<f32> = Vec::new();
            let mut value_copy_buf: Vec<f32> = Vec::new();

            // 4. Per-token: Q/K norm, RoPE, KV cache writes.
            for i in 0..batch {
                let pos = start_pos + i;
                let q = &mut q_batch[i * q_len..i * q_len + q_len_used0];
                let k = &mut k_batch[i * kv_len..(i + 1) * kv_len];
                let v = &v_batch[i * kv_len..(i + 1) * kv_len];

                if !layer.attn_q_norm.is_empty() && q.len() == layer.attn_q_norm.len() {
                    let normed_q = &mut qk_norm_scratch[..q.len()];
                    rms_norm_f32(q, &layer.attn_q_norm, cfg.rms_norm_eps, normed_q)
                        .map_err(|e| ModelError::InferenceFailed(format!("q_norm: {:?}", e)))?;
                    q.copy_from_slice(normed_q);
                } else if !layer.attn_q_norm.is_empty() && q_head_dim == layer.attn_q_norm.len() {
                    for head in 0..q_heads {
                        let start = head * q_head_dim;
                        let end = start + q_head_dim;
                        if end > q.len() {
                            break;
                        }
                        let normed_head = &mut head_scratch[..q_head_dim];
                        normed_head.fill(0.0_f32);
                        rms_norm_f32(
                            &q[start..end],
                            &layer.attn_q_norm,
                            cfg.rms_norm_eps,
                            normed_head,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("q_norm: {:?}", e)))?;
                        q[start..end].copy_from_slice(normed_head);
                    }
                }
                if !layer.attn_k_norm.is_empty() && k.len() == layer.attn_k_norm.len() {
                    let normed_k = &mut qk_norm_scratch[..k.len()];
                    rms_norm_f32(k, &layer.attn_k_norm, cfg.rms_norm_eps, normed_k)
                        .map_err(|e| ModelError::InferenceFailed(format!("k_norm: {:?}", e)))?;
                    k.copy_from_slice(normed_k);
                } else if !layer.attn_k_norm.is_empty() && kv_head_dim == layer.attn_k_norm.len() {
                    for head in 0..kv_heads {
                        let start = head * kv_head_dim;
                        let end = start + kv_head_dim;
                        if end > k.len() {
                            break;
                        }
                        let normed_head = &mut head_scratch[..kv_head_dim];
                        normed_head.fill(0.0_f32);
                        rms_norm_f32(
                            &k[start..end],
                            &layer.attn_k_norm,
                            cfg.rms_norm_eps,
                            normed_head,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("k_norm: {:?}", e)))?;
                        k[start..end].copy_from_slice(normed_head);
                    }
                }

                // RoPE Q — only rotate the first `rope_dim` elements per head (partial RoPE).
                let q_rope_len = cfg.effective_rope_dim().min(q_head_dim);
                for head in 0..q_heads {
                    let off = head * q_head_dim;
                    if off + q_head_dim > q.len() {
                        break;
                    }
                    let rotated = &mut head_scratch[..q_rope_len];
                    rotated.fill(0.0_f32);
                    apply_rope_f32(
                        &q[off..off + q_rope_len],
                        pos,
                        q_rope_len,
                        layer_rope,
                        rotated,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("rope q: {:?}", e)))?;
                    q[off..off + q_rope_len].copy_from_slice(rotated);
                }
                // RoPE K — partial RoPE: same rope_dim slice.
                let k_rope_len = cfg.effective_rope_dim().min(kv_head_dim);
                for head in 0..kv_heads {
                    let off = head * kv_head_dim;
                    if off + kv_head_dim > k.len() {
                        break;
                    }
                    let rotated = &mut head_scratch[..k_rope_len];
                    rotated.fill(0.0_f32);
                    apply_rope_f32(
                        &k[off..off + k_rope_len],
                        pos,
                        k_rope_len,
                        layer_rope,
                        rotated,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("rope k: {:?}", e)))?;
                    k[off..off + k_rope_len].copy_from_slice(rotated);
                }

                let cache_token_size = self.kv_cache.config().token_size();
                if cache_token_size == kv_len {
                    self.kv_cache
                        .set(kv_layer_idx, pos, k, v)
                        .map_err(|e| ModelError::InferenceFailed(format!("kv set: {:?}", e)))?;
                } else {
                    key_copy_buf.resize(cache_token_size, 0.0_f32);
                    value_copy_buf.resize(cache_token_size, 0.0_f32);
                    key_copy_buf[..kv_len].copy_from_slice(k);
                    value_copy_buf[..kv_len].copy_from_slice(v);
                    self.kv_cache
                        .set(kv_layer_idx, pos, &key_copy_buf, &value_copy_buf)
                        .map_err(|e| ModelError::InferenceFailed(format!("kv set: {:?}", e)))?;
                }
            }

            // 5. Per-token: attention. Each position attends to its own causal
            // prefix (positions 0..=pos).
            //
            // For F32 KV caches we try to borrow the prefix directly (zero-copy).
            // For quantized KV caches we copy into temporary buffers. Both paths
            // use the same flash attention kernel.
            for i in 0..batch {
                let pos = start_pos + i;
                let seq_len = pos + 1;
                let q = &q_batch[i * q_len..i * q_len + q_len_used0];
                let attn_out_slice = &mut attn_result_batch[i * q_len_used0..(i + 1) * q_len_used0];
                attn_out_slice.fill(0.0_f32);

                let (key_cache, value_cache): (&[f32], &[f32]) = {
                    let cache_token_size = self.kv_cache.config().token_size();
                    let key_borrow = self
                        .kv_cache
                        .f32_layer_key_prefix(kv_layer_idx, seq_len)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("kv borrow keys: {:?}", e))
                        })?;
                    let value_borrow = self
                        .kv_cache
                        .f32_layer_value_prefix(kv_layer_idx, seq_len)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("kv borrow vals: {:?}", e))
                        })?;

                    if cache_token_size == kv_len
                        && let (Some(keys), Some(values)) = (key_borrow, value_borrow)
                    {
                        (keys, values)
                    } else {
                        let needed = seq_len * kv_len;
                        if key_copy_buf.len() < needed {
                            key_copy_buf.resize(needed, 0.0_f32);
                        }
                        if value_copy_buf.len() < needed {
                            value_copy_buf.resize(needed, 0.0_f32);
                        }
                        if cache_token_size == kv_len {
                            self.kv_cache
                                .copy_layer_keys(kv_layer_idx, seq_len, &mut key_copy_buf[..needed])
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("kv copy keys: {:?}", e))
                                })?;
                            self.kv_cache
                                .copy_layer_values(
                                    kv_layer_idx,
                                    seq_len,
                                    &mut value_copy_buf[..needed],
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("kv copy vals: {:?}", e))
                                })?;
                        } else {
                            self.kv_cache
                                .copy_layer_key_prefix_values(
                                    kv_layer_idx,
                                    seq_len,
                                    kv_len,
                                    &mut key_copy_buf[..needed],
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("kv copy keys: {:?}", e))
                                })?;
                            self.kv_cache
                                .copy_layer_value_prefix_values(
                                    kv_layer_idx,
                                    seq_len,
                                    kv_len,
                                    &mut value_copy_buf[..needed],
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("kv copy vals: {:?}", e))
                                })?;
                        }
                        (&key_copy_buf[..needed], &value_copy_buf[..needed])
                    }
                };

                // Sliding-window attention: a local layer attends only to the most
                // recent `layer_window` positions. Since RoPE encodes absolute
                // positions, slicing off the oldest key/value rows yields exactly
                // the windowed-causal mask with relative positions preserved.
                let (eff_seq_len, key_cache, value_cache) =
                    if layer_window > 0 && seq_len > layer_window {
                        let skip = (seq_len - layer_window) * kv_len;
                        (layer_window, &key_cache[skip..], &value_cache[skip..])
                    } else {
                        (seq_len, key_cache, value_cache)
                    };

                flash_attention_decode_heads_f32(
                    q,
                    key_cache,
                    value_cache,
                    eff_seq_len,
                    kv_head_dim,
                    kv_len,
                    q_heads,
                    kv_heads,
                    attn_out_slice,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("flash attn: {:?}", e)))?;
            }

            // 6. Batched attn_output projection.
            if !layer.attn_output.is_empty() && aoil0 > 0 {
                gemm_weight(
                    &layer.attn_output,
                    h,
                    aoil0,
                    &attn_result_batch,
                    &mut attn_proj_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("attn_output: {:?}", e)))?;
                if !layer.attn_output_bias.is_empty() {
                    add_repeating_bias(&mut attn_proj_batch, &layer.attn_output_bias);
                }
            } else {
                attn_proj_batch.fill(0.0_f32);
            }

            // 6b. Gemma sandwich norm: normalize the attention output before the
            // residual add (post_attention_norm).
            if cfg.sandwich_norm && !layer.post_attention_norm.is_empty() {
                for i in 0..batch {
                    let range = i * h..(i + 1) * h;
                    rms_norm_f32(
                        &attn_proj_batch[range.clone()],
                        &layer.post_attention_norm,
                        cfg.rms_norm_eps,
                        &mut normed_batch[range.clone()],
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("post_attn_norm: {:?}", e)))?;
                    attn_proj_batch[range.clone()].copy_from_slice(&normed_batch[range]);
                }
            }

            // 7. Residual add (attn).
            for i in 0..batch * h {
                x_batch[i] += attn_proj_batch[i];
            }

            // 8. FFN: per-token RMSNorm, batched gate+up, SwiGLU, batched down.
            let has_ffn = !layer.ffn_gate.is_empty()
                && !layer.ffn_up.is_empty()
                && !layer.ffn_down.is_empty()
                && !ffn_norm_weight.is_empty();
            if has_ffn {
                for i in 0..batch {
                    rms_norm_f32(
                        &x_batch[i * h..(i + 1) * h],
                        ffn_norm_weight,
                        cfg.rms_norm_eps,
                        &mut normed_batch[i * h..(i + 1) * h],
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("ffn_norm: {:?}", e)))?;
                }

                gemm_weight(
                    &layer.ffn_gate,
                    i_size,
                    h,
                    &normed_batch,
                    &mut gate_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ffn_gate: {:?}", e)))?;
                gemm_weight(
                    &layer.ffn_up,
                    i_size,
                    h,
                    &normed_batch,
                    &mut up_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ffn_up: {:?}", e)))?;

                if cfg.gelu_ffn {
                    apply_geglu_inplace_f32(&mut gate_batch, &up_batch);
                } else {
                    for (g, u) in gate_batch.iter_mut().zip(up_batch.iter()) {
                        let sigmoid = 1.0_f32 / (1.0 + (-*g).exp());
                        *g = *g * sigmoid * *u;
                    }
                }

                gemm_weight(
                    &layer.ffn_down,
                    h,
                    i_size,
                    &gate_batch,
                    &mut ffn_out_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ffn_down: {:?}", e)))?;
                if !layer.ffn_down_bias.is_empty() {
                    add_repeating_bias(&mut ffn_out_batch, &layer.ffn_down_bias);
                }

                // Gemma sandwich norm: normalize the FFN output before residual.
                if cfg.sandwich_norm && !layer.post_ffn_norm.is_empty() {
                    for i in 0..batch {
                        let range = i * h..(i + 1) * h;
                        rms_norm_f32(
                            &ffn_out_batch[range.clone()],
                            &layer.post_ffn_norm,
                            cfg.rms_norm_eps,
                            &mut normed_batch[range.clone()],
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("post_ffn_norm: {:?}", e))
                        })?;
                        ffn_out_batch[range.clone()].copy_from_slice(&normed_batch[range]);
                    }
                }

                for i in 0..batch * h {
                    x_batch[i] += ffn_out_batch[i];
                }
            }
            if trace_fwd_enabled() {
                for t in 0..batch {
                    let sum: f64 = x_batch[t * h..(t + 1) * h].iter().map(|v| *v as f64).sum();
                    eprintln!(
                        "TRACE inf pos={} layer={layer_idx} sum={sum:.9e}",
                        start_pos + t
                    );
                }
            }
        }

        if !need_logits {
            return Ok(None);
        }

        let mut all_logits = Vec::with_capacity(batch);
        let mut final_normed = vec![0.0_f32; h];
        for row in x_batch.chunks_exact(h) {
            rms_norm_f32(row, &self.norm_weight, cfg.rms_norm_eps, &mut final_normed)
                .map_err(|e| ModelError::InferenceFailed(format!("final_norm: {e:?}")))?;
            let mut logits = vec![0.0_f32; cfg.vocab_size];
            gemv_weight(
                &self.output_weight,
                cfg.vocab_size,
                h,
                &final_normed,
                &mut logits,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("output: {e:?}")))?;
            all_logits.push(logits);
        }
        self.last_output_hidden.clone_from(&final_normed);
        Ok(Some(all_logits))
    }

    /// True continuous-batching decode across N **sequences** (one decode token
    /// each), as opposed to [`forward_batched`] which batches positions of ONE
    /// sequence. `rows[i] = (token, absolute_position)` for sequence `i`, and
    /// `kv[i]` is that sequence's caller-owned KV buffer (see [`SeqKv`]). All
    /// weight matrices are read once and fanned out across the `N = rows.len()`
    /// rows via [`gemm_weight`] — that amortization is the whole point: decode is
    /// memory-bound, so N sequences as one batched GEMM run at ~N× the per-token
    /// throughput until compute-bound.
    ///
    /// Per-sequence attention reads each sequence's own contiguous KV slice (no
    /// gather) and uses `rows[i].1` for RoPE, so each sequence produces logits
    /// IDENTICAL to running it alone. Returns `Vec<Logits>` in `rows` order
    /// (positional identity `rows[i] <-> result[i]`).
    ///
    /// This method NEVER touches `self.kv_cache` (that cache is a single flat
    /// timeline and would alias all N sequences onto one position axis). F32 KV
    /// only in this first cut; a non-F32 model KV dtype is rejected up front.
    #[allow(clippy::too_many_lines)]
    pub fn forward_batch(
        &mut self,
        rows: &[(Token, usize)],
        kv: &mut [SeqKv],
        need_logits: bool,
    ) -> Result<Vec<Logits>, ModelError> {
        if rows.is_empty() {
            return Err(ModelError::EmptyInput);
        }
        if rows.len() != kv.len() {
            return Err(ModelError::InferenceFailed(format!(
                "forward_batch: rows ({}) and kv ({}) length mismatch",
                rows.len(),
                kv.len()
            )));
        }
        if self.kv_cache.config().dtype != crate::tensor::DType::F32 {
            return Err(ModelError::InferenceFailed(
                "forward_batch: only F32 KV cache is supported in this path".to_owned(),
            ));
        }
        if !self.layers_supported_for_batched() {
            return Err(ModelError::InferenceFailed(
                "forward_batch: model layers are not supported by the batched path".to_owned(),
            ));
        }

        let batch = rows.len();
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let kvh = cfg.num_key_value_heads;

        // 1. Embedding lookup for every sequence into x_batch[batch, h].
        let mut x_batch = vec![0.0_f32; batch * h];
        for (i, &(token, _pos)) in rows.iter().enumerate() {
            let token_idx = (token as usize).min(cfg.vocab_size.saturating_sub(1));
            let target = &mut x_batch[i * h..(i + 1) * h];
            match &self.tok_embeddings {
                WeightStorage::F32(data) => {
                    let row = &data[token_idx * h..(token_idx + 1) * h];
                    target.copy_from_slice(row);
                }
                WeightStorage::Quantized(qtype, data) => {
                    lookup_quantized_embedding(h, *qtype, data, token_idx, target);
                }
                WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                    let data = &mmap[*offset..*offset + *size];
                    lookup_quantized_embedding(h, *qtype, data, token_idx, target);
                }
            }
            if cfg.embedding_scale != 1.0 {
                for v in target.iter_mut() {
                    *v *= cfg.embedding_scale;
                }
            }
        }

        // Dimensions — identical derivation to forward_batched (lines 224-254).
        let layer0 = &self.layers[0];
        let q_len = layer0.attn_q.output_dim(h);
        let kv_len = if !layer0.attn_k.is_empty() {
            layer0.attn_k.output_dim(h)
        } else {
            0
        };
        let aoil0 = if !layer0.attn_output.is_empty() {
            layer0.attn_output.output_dim(h)
        } else {
            0
        };
        let q_len_used0 = if aoil0 > 0 {
            q_len.min(aoil0)
        } else if q_len > h {
            h
        } else {
            q_len
        };
        let q_head_dim = if n > 0 && q_len_used0.is_multiple_of(n) {
            q_len_used0 / n
        } else {
            q_len_used0
        };
        let kv_head_dim = if kvh > 0 && kv_len > 0 && kv_len.is_multiple_of(kvh) {
            kv_len / kvh
        } else if kv_len > 0 {
            kv_len
        } else {
            q_head_dim
        };
        let i_size = cfg.intermediate_size;

        // The caller-owned SeqKv layout must agree with the model's KV width.
        for (i, seq) in kv.iter().enumerate() {
            let needed = self.kv_layer_count() * seq.capacity_tokens * kv_len;
            if seq.key.len() < needed || seq.value.len() < needed {
                return Err(ModelError::InferenceFailed(format!(
                    "forward_batch: kv[{i}] buffer too small (have {}/{}, need {needed} for kv_len={kv_len})",
                    seq.key.len(),
                    seq.value.len()
                )));
            }
            if rows[i].1 != seq.len {
                return Err(ModelError::InferenceFailed(format!(
                    "forward_batch: rows[{i}] pos {} != kv[{i}].len {} (KV write slot must equal decode position)",
                    rows[i].1, seq.len
                )));
            }
            if seq.len >= seq.capacity_tokens {
                return Err(ModelError::InferenceFailed(format!(
                    "forward_batch: kv[{i}] is full (len {} >= capacity {})",
                    seq.len, seq.capacity_tokens
                )));
            }
        }

        let mut normed_batch = vec![0.0_f32; batch * h];
        let mut q_batch = vec![0.0_f32; batch * q_len];
        let mut k_batch = vec![0.0_f32; batch * kv_len.max(1)];
        let mut v_batch = vec![0.0_f32; batch * kv_len.max(1)];
        let mut attn_result_batch = vec![0.0_f32; batch * q_len_used0];
        let mut attn_proj_batch = vec![0.0_f32; batch * h];
        let mut gate_batch = vec![0.0_f32; batch * i_size];
        let mut up_batch = vec![0.0_f32; batch * i_size];
        let mut ffn_out_batch = vec![0.0_f32; batch * h];
        let mut head_scratch = vec![0.0_f32; q_head_dim.max(kv_head_dim)];
        let mut qk_norm_scratch = vec![0.0_f32; q_len.max(kv_len)];

        for layer_idx in 0..cfg.layer_count {
            let layer = &self.layers[layer_idx];
            // Only attention layers own a KV slot; non-attention layers (MoE
            // routers, shortconv, etc.) map to `None`. Never fall back to
            // `layer_idx` — that can index past `kv_layer_count` and corrupt /
            // panic on the raw SeqKv slice. `None` ⇒ skip all KV access.
            let kv_layer_idx: Option<usize> = self.kv_layer_map.get(layer_idx).copied().flatten();

            let ffn_norm_weight: &[f32] = if cfg.sandwich_norm {
                &layer.ffn_norm
            } else if !layer.post_attention_norm.is_empty() {
                &layer.post_attention_norm
            } else if !layer.ffn_norm.is_empty() {
                &layer.ffn_norm
            } else {
                &[]
            };

            let layer_rope = cfg.layer_rope_theta(layer_idx);
            let layer_window = cfg.layer_sliding_window(layer_idx);

            // 2. Per-row attn RMSNorm.
            for i in 0..batch {
                rms_norm_f32(
                    &x_batch[i * h..(i + 1) * h],
                    &layer.attn_norm,
                    cfg.rms_norm_eps,
                    &mut normed_batch[i * h..(i + 1) * h],
                )
                .map_err(|e| ModelError::InferenceFailed(format!("rms_norm: {:?}", e)))?;
            }

            // 3. Batched Q/K/V GEMM — amortizes one weight read across all N seqs.
            gemm_weight(&layer.attn_q, q_len, h, &normed_batch, &mut q_batch, batch)
                .map_err(|e| ModelError::InferenceFailed(format!("attn_q: {:?}", e)))?;
            if !layer.attn_q_bias.is_empty() {
                add_repeating_bias(&mut q_batch, &layer.attn_q_bias);
            }
            if kv_len > 0 {
                gemm_weight(
                    &layer.attn_k,
                    kv_len,
                    h,
                    &normed_batch,
                    &mut k_batch[..batch * kv_len],
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("attn_k: {:?}", e)))?;
                if !layer.attn_k_bias.is_empty() {
                    add_repeating_bias(&mut k_batch[..batch * kv_len], &layer.attn_k_bias);
                }
                gemm_weight(
                    &layer.attn_v,
                    kv_len,
                    h,
                    &normed_batch,
                    &mut v_batch[..batch * kv_len],
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("attn_v: {:?}", e)))?;
                if !layer.attn_v_bias.is_empty() {
                    add_repeating_bias(&mut v_batch[..batch * kv_len], &layer.attn_v_bias);
                }
            }

            let q_heads = q_len_used0 / q_head_dim.max(1);
            let kv_heads = kv_len.checked_div(kv_head_dim).unwrap_or(0);

            // 4. Per-row: Q/K norm, RoPE (using the sequence's OWN position), then
            //    KV write into that sequence's own buffer at slot (layer, len).
            for i in 0..batch {
                let pos = rows[i].1;
                let q = &mut q_batch[i * q_len..i * q_len + q_len_used0];
                let k = &mut k_batch[i * kv_len..(i + 1) * kv_len];
                let v = &v_batch[i * kv_len..(i + 1) * kv_len];

                if !layer.attn_q_norm.is_empty() && q.len() == layer.attn_q_norm.len() {
                    let normed_q = &mut qk_norm_scratch[..q.len()];
                    rms_norm_f32(q, &layer.attn_q_norm, cfg.rms_norm_eps, normed_q)
                        .map_err(|e| ModelError::InferenceFailed(format!("q_norm: {:?}", e)))?;
                    q.copy_from_slice(normed_q);
                } else if !layer.attn_q_norm.is_empty() && q_head_dim == layer.attn_q_norm.len() {
                    for head in 0..q_heads {
                        let start = head * q_head_dim;
                        let end = start + q_head_dim;
                        if end > q.len() {
                            break;
                        }
                        let normed_head = &mut head_scratch[..q_head_dim];
                        normed_head.fill(0.0_f32);
                        rms_norm_f32(
                            &q[start..end],
                            &layer.attn_q_norm,
                            cfg.rms_norm_eps,
                            normed_head,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("q_norm: {:?}", e)))?;
                        q[start..end].copy_from_slice(normed_head);
                    }
                }
                if !layer.attn_k_norm.is_empty() && k.len() == layer.attn_k_norm.len() {
                    let normed_k = &mut qk_norm_scratch[..k.len()];
                    rms_norm_f32(k, &layer.attn_k_norm, cfg.rms_norm_eps, normed_k)
                        .map_err(|e| ModelError::InferenceFailed(format!("k_norm: {:?}", e)))?;
                    k.copy_from_slice(normed_k);
                } else if !layer.attn_k_norm.is_empty() && kv_head_dim == layer.attn_k_norm.len() {
                    for head in 0..kv_heads {
                        let start = head * kv_head_dim;
                        let end = start + kv_head_dim;
                        if end > k.len() {
                            break;
                        }
                        let normed_head = &mut head_scratch[..kv_head_dim];
                        normed_head.fill(0.0_f32);
                        rms_norm_f32(
                            &k[start..end],
                            &layer.attn_k_norm,
                            cfg.rms_norm_eps,
                            normed_head,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("k_norm: {:?}", e)))?;
                        k[start..end].copy_from_slice(normed_head);
                    }
                }

                let q_rope_len = cfg.effective_rope_dim().min(q_head_dim);
                for head in 0..q_heads {
                    let off = head * q_head_dim;
                    if off + q_head_dim > q.len() {
                        break;
                    }
                    let rotated = &mut head_scratch[..q_rope_len];
                    rotated.fill(0.0_f32);
                    apply_rope_f32(
                        &q[off..off + q_rope_len],
                        pos,
                        q_rope_len,
                        layer_rope,
                        rotated,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("rope q: {:?}", e)))?;
                    q[off..off + q_rope_len].copy_from_slice(rotated);
                }
                let k_rope_len = cfg.effective_rope_dim().min(kv_head_dim);
                for head in 0..kv_heads {
                    let off = head * kv_head_dim;
                    if off + kv_head_dim > k.len() {
                        break;
                    }
                    let rotated = &mut head_scratch[..k_rope_len];
                    rotated.fill(0.0_f32);
                    apply_rope_f32(
                        &k[off..off + k_rope_len],
                        pos,
                        k_rope_len,
                        layer_rope,
                        rotated,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("rope k: {:?}", e)))?;
                    k[off..off + k_rope_len].copy_from_slice(rotated);
                }

                // 5. KV WRITE into the sequence's own buffer at slot (layer, len).
                //    Only attention layers have a KV slot; skip otherwise.
                if kv_len > 0 {
                    if let Some(kv_idx) = kv_layer_idx {
                        let seq = &mut kv[i];
                        let cap = seq.capacity_tokens;
                        let base = kv_idx * cap * kv_len + seq.len * kv_len;
                        seq.key[base..base + kv_len].copy_from_slice(k);
                        seq.value[base..base + kv_len].copy_from_slice(v);
                    }
                }
            }

            // 6. Per-row attention over each sequence's OWN contiguous KV prefix.
            //    Guarded on `Some(kv_idx)`: non-attention layers wrote no KV and
            //    must not read the (out-of-bounds) raw SeqKv slice.
            if let (true, Some(kv_idx)) = (kv_len > 0, kv_layer_idx) {
                for i in 0..batch {
                    let seq_len = kv[i].len + 1; // current token now written
                    let q = &q_batch[i * q_len..i * q_len + q_len_used0];
                    let attn_out_slice =
                        &mut attn_result_batch[i * q_len_used0..(i + 1) * q_len_used0];
                    attn_out_slice.fill(0.0_f32);

                    let cap = kv[i].capacity_tokens;
                    let layer_base = kv_idx * cap * kv_len;
                    let key_cache = &kv[i].key[layer_base..layer_base + seq_len * kv_len];
                    let value_cache = &kv[i].value[layer_base..layer_base + seq_len * kv_len];

                    // Sliding window: drop oldest rows (RoPE preserves relative pos).
                    let (eff_seq_len, key_cache, value_cache) =
                        if layer_window > 0 && seq_len > layer_window {
                            let skip = (seq_len - layer_window) * kv_len;
                            (layer_window, &key_cache[skip..], &value_cache[skip..])
                        } else {
                            (seq_len, key_cache, value_cache)
                        };

                    flash_attention_decode_heads_f32(
                        q,
                        key_cache,
                        value_cache,
                        eff_seq_len,
                        kv_head_dim,
                        kv_len,
                        q_heads,
                        kv_heads,
                        attn_out_slice,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("flash attn: {:?}", e)))?;
                }
            }

            // 7. Batched attn_output projection + per-row residual.
            if !layer.attn_output.is_empty() && aoil0 > 0 {
                gemm_weight(
                    &layer.attn_output,
                    h,
                    aoil0,
                    &attn_result_batch,
                    &mut attn_proj_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("attn_output: {:?}", e)))?;
                if !layer.attn_output_bias.is_empty() {
                    add_repeating_bias(&mut attn_proj_batch, &layer.attn_output_bias);
                }
            } else {
                attn_proj_batch.fill(0.0_f32);
            }

            if cfg.sandwich_norm && !layer.post_attention_norm.is_empty() {
                for i in 0..batch {
                    let range = i * h..(i + 1) * h;
                    rms_norm_f32(
                        &attn_proj_batch[range.clone()],
                        &layer.post_attention_norm,
                        cfg.rms_norm_eps,
                        &mut normed_batch[range.clone()],
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("post_attn_norm: {:?}", e)))?;
                    attn_proj_batch[range.clone()].copy_from_slice(&normed_batch[range]);
                }
            }

            for i in 0..batch * h {
                x_batch[i] += attn_proj_batch[i];
            }

            // 8. FFN: per-row norm, batched gate+up, SwiGLU, batched down.
            let has_ffn = !layer.ffn_gate.is_empty()
                && !layer.ffn_up.is_empty()
                && !layer.ffn_down.is_empty()
                && !ffn_norm_weight.is_empty();
            if has_ffn {
                for i in 0..batch {
                    rms_norm_f32(
                        &x_batch[i * h..(i + 1) * h],
                        ffn_norm_weight,
                        cfg.rms_norm_eps,
                        &mut normed_batch[i * h..(i + 1) * h],
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("ffn_norm: {:?}", e)))?;
                }

                gemm_weight(
                    &layer.ffn_gate,
                    i_size,
                    h,
                    &normed_batch,
                    &mut gate_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ffn_gate: {:?}", e)))?;
                gemm_weight(
                    &layer.ffn_up,
                    i_size,
                    h,
                    &normed_batch,
                    &mut up_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ffn_up: {:?}", e)))?;

                if cfg.gelu_ffn {
                    apply_geglu_inplace_f32(&mut gate_batch, &up_batch);
                } else {
                    for (g, u) in gate_batch.iter_mut().zip(up_batch.iter()) {
                        let sigmoid = 1.0_f32 / (1.0 + (-*g).exp());
                        *g = *g * sigmoid * *u;
                    }
                }

                gemm_weight(
                    &layer.ffn_down,
                    h,
                    i_size,
                    &gate_batch,
                    &mut ffn_out_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ffn_down: {:?}", e)))?;
                if !layer.ffn_down_bias.is_empty() {
                    add_repeating_bias(&mut ffn_out_batch, &layer.ffn_down_bias);
                }

                if cfg.sandwich_norm && !layer.post_ffn_norm.is_empty() {
                    for i in 0..batch {
                        let range = i * h..(i + 1) * h;
                        rms_norm_f32(
                            &ffn_out_batch[range.clone()],
                            &layer.post_ffn_norm,
                            cfg.rms_norm_eps,
                            &mut normed_batch[range.clone()],
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("post_ffn_norm: {:?}", e))
                        })?;
                        ffn_out_batch[range.clone()].copy_from_slice(&normed_batch[range]);
                    }
                }

                for i in 0..batch * h {
                    x_batch[i] += ffn_out_batch[i];
                }
            }
        }

        // 8b. Bump each sequence's KV length exactly once, after the last layer.
        for seq in kv.iter_mut() {
            seq.len += 1;
        }

        if !need_logits {
            return Ok(Vec::new());
        }

        // 9. LM HEAD per sequence — final norm + lm_head for EVERY row (unlike
        //    forward_batched which only emits the last position).
        let mut out: Vec<Logits> = Vec::with_capacity(batch);
        let mut final_normed = vec![0.0_f32; h];
        for i in 0..batch {
            rms_norm_f32(
                &x_batch[i * h..(i + 1) * h],
                &self.norm_weight,
                cfg.rms_norm_eps,
                &mut final_normed,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("final_norm: {:?}", e)))?;
            let mut logits = vec![0.0_f32; cfg.vocab_size];
            gemv_weight(
                &self.output_weight,
                cfg.vocab_size,
                h,
                &final_normed,
                &mut logits,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))?;
            out.push(logits);
        }
        Ok(out)
    }

    /// True when EVERY layer is representable by the device batched forward
    /// (`gpu_forward_batch_layer`): strict Q4_K for all of `attn_q/k/v/output`
    /// and `ffn_gate/up/down`, no sliding window, no sandwich norm, SwiGLU (not
    /// GeGLU). QK-norm (Qwen3) is allowed (the device path reuses the per-head
    /// `rms_norm_f32_kernel`). The bN GEMV kernel requires 144-byte Q4_K blocks,
    /// so Q6_K tensors (mixed-precision Q4_K_M) are NOT eligible here and force a
    /// fall back to the CPU `forward_batch`.
    #[cfg(feature = "cuda")]
    fn layers_supported_for_batched_gpu(&self) -> bool {
        let cfg = &self.config;
        if cfg.sandwich_norm || cfg.gelu_ffn {
            return false;
        }
        // Hidden / intermediate must be Q8_K block aligned (256) for the device
        // quantize + bN GEMV.
        if !cfg.hidden_size.is_multiple_of(256) || !cfg.intermediate_size.is_multiple_of(256) {
            return false;
        }
        if self.layers.is_empty() {
            return false;
        }
        for (idx, layer) in self.layers.iter().enumerate() {
            if cfg.layer_sliding_window(idx) != 0 {
                return false;
            }
            for ws in [
                &layer.attn_q,
                &layer.attn_k,
                &layer.attn_v,
                &layer.attn_output,
                &layer.ffn_gate,
                &layer.ffn_up,
                &layer.ffn_down,
            ] {
                if Self::q4k_bytes(ws).is_none() {
                    return false;
                }
            }
            // Attention projection widths must be Q4_K-block aligned too.
            let q_len = layer.attn_q.output_dim(cfg.hidden_size);
            let kv_len = layer.attn_k.output_dim(cfg.hidden_size);
            let ao_len = layer.attn_output.output_dim(cfg.hidden_size);
            if !q_len.is_multiple_of(256)
                || !kv_len.is_multiple_of(256)
                || !ao_len.is_multiple_of(256)
            {
                return false;
            }
            // The batched Wo/QKV path quantizes q_len/kv_len-wide activations into
            // the `xq8k` scratch sized for `hidden_size`; if a projection is wider
            // than hidden, that write would overrun the device buffer. Fall back to
            // CPU in that (rare) case rather than corrupt VRAM.
            if q_len > cfg.hidden_size || kv_len > cfg.hidden_size {
                return false;
            }
        }
        true
    }

    /// GPU analogue of [`forward_batch`](Self::forward_batch): `B` decode rows
    /// (one token each, distinct sequences) in one weighted pass per projection
    /// matrix on-device. Returns `Ok(Some(logits))` when the device batched path
    /// ran, `Ok(None)` when ineligible — the caller MUST then fall back to
    /// [`forward_batch`](Self::forward_batch). Never touches `self.kv_cache`; the
    /// batched F16 KV lives device-side (`kv_*_batched`). Host `SeqKv.key/value`
    /// arrays are NOT written here (only `len` is bumped for bookkeeping); this is
    /// the documented divergence from the CPU `forward_batch`.
    ///
    /// Eligibility (any failure → `Ok(None)`): `OX_GPU_BATCHED=1`, `B <= 8`, F32
    /// model KV-cache config, `layers_supported_for_batched`, every layer
    /// `layer_can_use_gpu_native` + `layers_supported_for_batched_gpu`, and an
    /// active CUDA device. Capacity / position violations are hard `Err` (caller
    /// bug, matching CPU `forward_batch`).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_lines)]
    pub fn forward_batch_gpu(
        &mut self,
        rows: &[(Token, usize)],
        kv: &mut [SeqKv],
        need_logits: bool,
    ) -> Result<Option<Vec<Logits>>, ModelError> {
        if !crate::cuda::ox_gpu_batched_enabled() {
            return Ok(None);
        }
        if rows.is_empty() || rows.len() != kv.len() {
            return Ok(None);
        }
        let batch = rows.len();
        if batch > crate::cuda::GPU_BATCHED_MAX_B {
            return Ok(None);
        }
        let dbg = std::env::var("OX_GPU_BATCHED_DEBUG").is_ok();
        if self.kv_cache.config().dtype != crate::tensor::DType::F32 {
            if dbg {
                eprintln!(
                    "forward_batch_gpu ineligible: KV dtype {:?} != F32",
                    self.kv_cache.config().dtype
                );
            }
            return Ok(None);
        }
        if !self.layers_supported_for_batched() {
            if dbg {
                eprintln!("forward_batch_gpu ineligible: layers_supported_for_batched()=false");
            }
            return Ok(None);
        }
        if !self.layers_supported_for_batched_gpu() {
            if dbg {
                eprintln!(
                    "forward_batch_gpu ineligible: layers_supported_for_batched_gpu()=false (a projection is not strict Q4_K, or q_len/kv_len > hidden)"
                );
            }
            return Ok(None);
        }
        let cfg = self.config.clone();
        if !self
            .layers
            .iter()
            .all(|l| Self::layer_can_use_gpu_native(l, &cfg))
        {
            if dbg {
                eprintln!("forward_batch_gpu ineligible: a layer fails layer_can_use_gpu_native");
            }
            return Ok(None);
        }
        if crate::gpu_dispatch::active_gpu().is_none() {
            if dbg {
                eprintln!("forward_batch_gpu ineligible: active_gpu() is None");
            }
            return Ok(None);
        }

        let h = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let n_q = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;

        let layer0 = &self.layers[0];
        let q_len = layer0.attn_q.output_dim(h);
        let kv_len = layer0.attn_k.output_dim(h);
        let head_dim = if n_q > 0 && q_len.is_multiple_of(n_q) {
            q_len / n_q
        } else {
            cfg.head_dim()
        };
        let rope_dim = {
            let eff = cfg.effective_rope_dim();
            if eff == 0 { head_dim } else { eff }
        };

        // The output (lm_head) weight must be strict Q4_K for the batched head;
        // otherwise we cannot run the whole forward on-device → fall back.
        let out_q4k = Self::q4k_bytes(&self.output_weight);
        if need_logits && out_q4k.is_none() {
            if dbg {
                eprintln!(
                    "forward_batch_gpu ineligible: output (lm_head) weight is not strict Q4_K"
                );
            }
            return Ok(None);
        }

        // Per-seq KV / position validation — hard error (caller contract).
        for (i, seq) in kv.iter().enumerate() {
            let needed = self.kv_layer_count() * seq.capacity_tokens * kv_len;
            if seq.key.len() < needed || seq.value.len() < needed {
                return Err(ModelError::InferenceFailed(format!(
                    "forward_batch_gpu: kv[{i}] buffer too small (have {}/{}, need {needed})",
                    seq.key.len(),
                    seq.value.len()
                )));
            }
            if rows[i].1 != seq.len {
                return Err(ModelError::InferenceFailed(format!(
                    "forward_batch_gpu: rows[{i}] pos {} != kv[{i}].len {}",
                    rows[i].1, seq.len
                )));
            }
            if seq.len >= seq.capacity_tokens {
                return Err(ModelError::InferenceFailed(format!(
                    "forward_batch_gpu: kv[{i}] is full (len {} >= capacity {})",
                    seq.len, seq.capacity_tokens
                )));
            }
        }

        let context_size = cfg.context_size;
        let kv_layers = self.kv_layer_count();
        let bn_rows = q_len
            .max(kv_len)
            .max(i_size)
            .max(if need_logits { vocab } else { 0 });

        // Lazy device buffer init (no-op when geometry+B match prior calls).
        crate::cuda::gpu_batched_activation_init(batch, h, i_size, q_len, kv_len, bn_rows)
            .map_err(ModelError::InferenceFailed)?;
        crate::cuda::gpu_kv_batched_init(kv_layers, kv_len, context_size, batch)
            .map_err(ModelError::InferenceFailed)?;
        // New prompt (all rows at position 0) ⇒ clear per-seq KV counters.
        if rows.iter().all(|&(_, p)| p == 0) {
            crate::cuda::gpu_kv_batched_reset().map_err(ModelError::InferenceFailed)?;
        }

        // 1. Host-side embedding lookup into row-major [B, h], then upload.
        let mut x_batch = vec![0.0_f32; batch * h];
        for (i, &(token, _pos)) in rows.iter().enumerate() {
            let token_idx = (token as usize).min(vocab.saturating_sub(1));
            let target = &mut x_batch[i * h..(i + 1) * h];
            match &self.tok_embeddings {
                WeightStorage::F32(data) => {
                    target.copy_from_slice(&data[token_idx * h..(token_idx + 1) * h]);
                }
                WeightStorage::Quantized(qtype, data) => {
                    lookup_quantized_embedding(h, *qtype, data, token_idx, target);
                }
                WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                    let data = &mmap[*offset..*offset + *size];
                    lookup_quantized_embedding(h, *qtype, data, token_idx, target);
                }
            }
            if cfg.embedding_scale != 1.0 {
                for v in target.iter_mut() {
                    *v *= cfg.embedding_scale;
                }
            }
        }
        crate::cuda::gpu_batched_upload_hidden(&x_batch).map_err(ModelError::InferenceFailed)?;

        let pos: Vec<usize> = rows.iter().map(|&(_, p)| p).collect();
        let kv_seq_len_pre: Vec<usize> = kv.iter().map(|s| s.len).collect();

        let geom = crate::cuda::BatchedGeom {
            batch,
            hidden_size: h,
            intermediate_size: i_size,
            q_len,
            kv_len,
            head_dim,
            n_q_heads: n_q,
            n_kv_heads: n_kv,
            rope_dim,
        };

        // 2. Run every transformer layer on-device (one weight pass per proj).
        for layer_idx in 0..cfg.layer_count {
            let layer = &self.layers[layer_idx];
            let kv_layer_idx = self
                .kv_layer_map
                .get(layer_idx)
                .copied()
                .flatten()
                .unwrap_or(layer_idx);
            let ffn_norm_weight: &[f32] = if !layer.post_attention_norm.is_empty() {
                &layer.post_attention_norm
            } else {
                &layer.ffn_norm
            };
            let wq = Self::q4k_bytes(&layer.attn_q).expect("eligibility checked");
            let wk = Self::q4k_bytes(&layer.attn_k).expect("eligibility checked");
            let wv = Self::q4k_bytes(&layer.attn_v).expect("eligibility checked");
            let wo = Self::q4k_bytes(&layer.attn_output).expect("eligibility checked");
            let gate = Self::q4k_bytes(&layer.ffn_gate).expect("eligibility checked");
            let up = Self::q4k_bytes(&layer.ffn_up).expect("eligibility checked");
            let down = Self::q4k_bytes(&layer.ffn_down).expect("eligibility checked");
            let weights = crate::cuda::BatchedLayerWeights {
                attn_norm: &layer.attn_norm,
                ffn_norm: ffn_norm_weight,
                eps: cfg.rms_norm_eps,
                wq,
                wk,
                wv,
                wo,
                gate,
                up,
                down,
                q_norm: &layer.attn_q_norm,
                k_norm: &layer.attn_k_norm,
                kv_layer_idx,
                layer_window: cfg.layer_sliding_window(layer_idx),
                theta: cfg.layer_rope_theta(layer_idx),
            };
            crate::cuda::gpu_forward_batch_layer(&weights, geom, &pos, &kv_seq_len_pre)
                .map_err(ModelError::InferenceFailed)?;
        }

        // 3. Bump each sequence's KV length once after the last layer (host
        //    bookkeeping; device counters were bumped during KV-append).
        for seq in kv.iter_mut() {
            seq.len += 1;
        }

        if !need_logits {
            // Sync so the device residual accumulator is complete before return.
            let mut sink = vec![0.0_f32; batch * h];
            crate::cuda::gpu_batched_download_hidden(&mut sink)
                .map_err(ModelError::InferenceFailed)?;
            return Ok(Some(Vec::new()));
        }

        // 4. Batched final head: per-row final norm + bN lm_head GEMV.
        let out_bytes = out_q4k.expect("checked above");
        let mut flat = vec![0.0_f32; batch * vocab];
        crate::cuda::gpu_batched_final_head(
            &self.norm_weight,
            cfg.rms_norm_eps,
            &out_bytes,
            vocab,
            h,
            batch,
            &mut flat,
        )
        .map_err(ModelError::InferenceFailed)?;

        let mut out: Vec<Logits> = Vec::with_capacity(batch);
        for i in 0..batch {
            out.push(flat[i * vocab..(i + 1) * vocab].to_vec());
        }
        Ok(Some(out))
    }

    /// Non-CUDA stub: the device batched path is never eligible, so the caller
    /// always falls back to the CPU [`forward_batch`](Self::forward_batch).
    #[cfg(not(feature = "cuda"))]
    pub fn forward_batch_gpu(
        &mut self,
        _rows: &[(Token, usize)],
        _kv: &mut [SeqKv],
        _need_logits: bool,
    ) -> Result<Option<Vec<Logits>>, ModelError> {
        Ok(None)
    }

    pub(super) fn forward_single(
        &mut self,
        token: Token,
        pos: usize,
        need_logits: bool,
    ) -> Result<Option<Logits>, ModelError> {
        let token_t0 = crate::tensor::decode_profile_enabled().then(std::time::Instant::now);
        self.embed_token_into_workspace(token);
        let layer_count = self.config.layer_count;
        self.run_layer_range_in_workspace(pos, 0..layer_count)?;
        if !need_logits {
            return Ok(None);
        }
        let logits = self.final_head_from_workspace().map(Some);
        if let Some(t0) = token_t0 {
            crate::tensor::decode_profile_record("token_forward", t0.elapsed().as_nanos() as u64);
        }
        logits
    }

    /// Write `token`'s embedding into `workspace.x[..hidden_size]`. First stage
    /// of pipeline-parallel decode.
    pub fn embed_token_into_workspace(&mut self, token: Token) {
        #[cfg(feature = "cuda")]
        {
            self.pending_embed_token = Some(token);
        }
        let h = self.config.hidden_size;
        let x = &mut self.workspace.x[..h];
        x.fill(0.0_f32);
        let token_idx = (token as usize).min(self.config.vocab_size.saturating_sub(1));
        match &self.tok_embeddings {
            WeightStorage::F32(data) => {
                let row = &data[token_idx * h..(token_idx + 1) * h];
                x.copy_from_slice(row);
            }
            WeightStorage::Quantized(qtype, data) => {
                lookup_quantized_embedding(h, *qtype, data, token_idx, x);
            }
            WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                let data = &mmap[*offset..*offset + *size];
                lookup_quantized_embedding(h, *qtype, data, token_idx, x);
            }
        }
        let scale = self.config.embedding_scale;
        if scale != 1.0 {
            for v in x.iter_mut() {
                *v *= scale;
            }
        }
    }

    /// Read the current hidden state from `workspace.x[..hidden_size]`.
    pub fn hidden_state(&self) -> &[f32] {
        &self.workspace.x[..self.config.hidden_size]
    }

    /// Hidden size from the loaded config (so pipeline drivers can size
    /// activation buffers without re-parsing GGUF metadata).
    pub fn config_hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Overwrite the current hidden state with `hidden`. Used by pipeline-
    /// parallel stages that receive activations over the ring.
    pub fn set_hidden_state(&mut self, hidden: &[f32]) -> Result<(), ModelError> {
        let h = self.config.hidden_size;
        if hidden.len() != h {
            return Err(ModelError::InferenceFailed(format!(
                "set_hidden_state: expected {} floats, got {}",
                h,
                hidden.len()
            )));
        }
        self.workspace.x[..h].copy_from_slice(hidden);
        Ok(())
    }

    /// RMSNorm on `hidden` using the model's final norm weights (for LoRA / training).
    pub fn apply_final_norm(&self, hidden: &[f32], out: &mut [f32]) -> Result<(), ModelError> {
        let h = self.config.hidden_size;
        if hidden.len() != h || out.len() != h {
            return Err(ModelError::InferenceFailed(format!(
                "apply_final_norm: expected {h} floats, got hidden={} out={}",
                hidden.len(),
                out.len()
            )));
        }
        rms_norm_f32(hidden, &self.norm_weight, self.config.rms_norm_eps, out)
            .map_err(|e| ModelError::InferenceFailed(format!("final_norm: {:?}", e)))
    }

    /// Final norm weights (read-only) for external training loops.
    pub fn final_norm_weight(&self) -> &[f32] {
        &self.norm_weight
    }

    /// Whether this GGUF contains a usable native MTP/nextn draft block.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Number of nextn layers advertised by GGUF metadata.
    pub fn nextn_predict_layers(&self) -> usize {
        self.config.nextn_predict_layers
    }

    /// Final output-normalized hidden row for the latest committed target token.
    pub fn last_output_hidden(&self) -> &[f32] {
        &self.last_output_hidden
    }

    /// Configure which target layers are snapshotted for EAGLE3 feature fusion.
    pub fn set_eagle3_capture_layers(&mut self, layers: Vec<usize>) {
        self.eagle3_capture_layers = layers;
        self.eagle3_layer_hiddens = vec![None; self.config.layer_count];
    }

    /// Concatenate the most recent hidden rows from [`set_eagle3_capture_layers`].
    pub fn concat_eagle3_features(&self) -> Result<Vec<f32>, ModelError> {
        let mut out = Vec::new();
        for &layer in &self.eagle3_capture_layers {
            let hidden = self
                .eagle3_layer_hiddens
                .get(layer)
                .and_then(|row| row.as_ref())
                .ok_or_else(|| {
                    ModelError::InferenceFailed(format!(
                        "missing EAGLE3 capture for target layer {layer}"
                    ))
                })?;
            out.extend_from_slice(hidden);
        }
        Ok(out)
    }

    /// Project already-normalized hidden states through the output (lm_head) matrix.
    pub fn lm_head_logits_from_normed(
        &self,
        normed: &[f32],
        logits: &mut [f32],
    ) -> Result<(), ModelError> {
        let h = self.config.hidden_size;
        if normed.len() != h {
            return Err(ModelError::InferenceFailed(format!(
                "lm_head_logits_from_normed: expected {h} floats, got {}",
                normed.len()
            )));
        }
        if logits.len() != self.config.vocab_size {
            return Err(ModelError::InferenceFailed(format!(
                "lm_head_logits_from_normed: logits len {} != vocab {}",
                logits.len(),
                self.config.vocab_size
            )));
        }
        logits.fill(0.0_f32);
        gemv_weight(
            &self.output_weight,
            self.config.vocab_size,
            h,
            normed,
            logits,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))
    }

    /// Apply final RMSNorm + lm_head to the current hidden state in
    /// `workspace.x` and return the logits. Last stage of pipeline-parallel.
    ///
    /// When the CUDA backend is available and the output weight is Q4K/Q6K, the
    /// lm_head GEMV is dispatched to the GPU.  For a 128 k-vocab model this
    /// saves 6–16 ms/token vs the CPU path (HBM bandwidth vs RAM bandwidth).
    pub fn final_head_from_workspace(&mut self) -> Result<Logits, ModelError> {
        let h = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let rms_norm_eps = self.config.rms_norm_eps;

        let ws = &mut self.workspace;

        #[cfg(feature = "cuda")]
        {
            if crate::cuda::gpu_activation_ready() {
                if let Some(w_bytes) = Self::q4k_or_q6k_bytes(&self.output_weight) {
                    let logits = &mut ws.logits[..vocab_size];
                    let last_hidden = &mut ws.hidden_a[..h];
                    if crate::cuda::gpu_final_head_device_resident(
                        &self.norm_weight,
                        rms_norm_eps,
                        w_bytes,
                        vocab_size,
                        h,
                        logits,
                        last_hidden,
                    )
                    .is_ok()
                    {
                        if !matches!(golden_sink(), GoldenSink::Disabled) {
                            dump_golden_logits(&logits[..vocab_size]);
                        }
                        let logits_out = logits.to_vec();
                        self.last_output_hidden = last_hidden.to_vec();
                        return Ok(logits_out);
                    }
                }
            }
        }

        // CPU final norm + lm_head (fallback when gpu_native inactive).
        let normed = &mut ws.hidden_a[..h];
        normed.fill(0.0_f32);
        rms_norm_f32(&ws.x[..h], &self.norm_weight, rms_norm_eps, normed)
            .map_err(|e| ModelError::InferenceFailed(format!("final_norm: {:?}", e)))?;
        let last_hidden = normed.to_vec();

        // lm_head GEMV: try GPU for Q4K/Q6K weights (avoids ~6–16 ms/token CPU bottleneck).
        #[cfg(feature = "cuda")]
        {
            if let Some(w_bytes) = Self::q4k_or_q6k_bytes(&self.output_weight) {
                let logits = &mut ws.logits[..vocab_size];
                if let Ok(()) =
                    crate::cuda::gpu_lm_head_quantized(w_bytes, vocab_size, h, normed, logits)
                {
                    if !matches!(golden_sink(), GoldenSink::Disabled) {
                        dump_golden_logits(&logits[..vocab_size]);
                    }
                    let logits_out = logits.to_vec();
                    self.last_output_hidden = last_hidden;
                    return Ok(logits_out);
                }
                // Fall through to CPU on error (first token before CUDA init, etc.)
            }
        }

        let logits = &mut ws.logits[..vocab_size];
        logits.fill(0.0_f32);
        gemv_weight(&self.output_weight, vocab_size, h, normed, logits)
            .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))?;
        if !matches!(golden_sink(), GoldenSink::Disabled) {
            dump_golden_logits(&logits[..vocab_size]);
        }
        let logits_out = logits.to_vec();
        self.last_output_hidden = last_hidden;
        Ok(logits_out)
    }
}
