use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ConvHistoryRing {
    pub(super) slots: Vec<f32>,
    pub(super) dim: usize,
    pub(super) capacity: usize,
    pub(super) head: usize,
    pub(super) len: usize,
}

impl ConvHistoryRing {
    pub(super) fn checksum(&self) -> f64 {
        self.slots.iter().map(|v| *v as f64).sum::<f64>()
            + self.head as f64 * 1e-3
            + self.len as f64 * 1e-6
    }

    pub(super) fn new(capacity: usize, dim: usize) -> Self {
        Self {
            slots: vec![0.0_f32; capacity.saturating_mul(dim)],
            dim,
            capacity: capacity.max(1),
            head: 0,
            len: 0,
        }
    }

    pub(super) fn push(&mut self, frame: &[f32]) {
        if self.dim == 0 || frame.len() != self.dim {
            return;
        }
        let start = self.head * self.dim;
        self.slots[start..start + self.dim].copy_from_slice(frame);
        self.head = (self.head + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    pub(super) fn past_frame(&self, steps_back: usize) -> Option<&[f32]> {
        if steps_back == 0 || steps_back > self.len {
            return None;
        }
        let idx = (self.head + self.capacity - steps_back) % self.capacity;
        let start = idx * self.dim;
        Some(&self.slots[start..start + self.dim])
    }
}

impl LayerWiseModel {
    pub(super) fn trace_state(&self, label: &str, pos: usize) {
        if crate::inference::trace_fwd_enabled() {
            let s0: f64 = self
                .ssm_states
                .first()
                .map(|s| s.iter().map(|v| *v as f64).sum())
                .unwrap_or(0.0);
            let r0: f64 = self
                .ssm_conv_buffers
                .first()
                .map(|b| b.checksum())
                .unwrap_or(0.0);
            eprintln!(
                "STATE {label} pos={pos} ssm_pos={} s0={s0:.9e} r0={r0:.9e}",
                self.ssm_pos
            );
        }
    }

    /// Record a recurrent-state checkpoint at `pos`, keeping at most the two
    /// most recent distinct positions (one speculative round needs two: the
    /// rollback target and the verify-window entry).
    pub(super) fn push_ssm_checkpoint(&mut self, pos: usize) {
        self.trace_state("push", pos);
        self.ssm_checkpoints.retain(|(p, _, _)| *p != pos);
        self.ssm_checkpoints
            .push((pos, self.ssm_states.clone(), self.ssm_conv_buffers.clone()));
        if self.ssm_checkpoints.len() > 2 {
            self.ssm_checkpoints.remove(0);
        }
    }

    pub(super) fn run_mamba_layer(
        &mut self,
        layer_idx: usize,
        layer: &LayerWeights,
        x: &[f32],
        cfg: &InferenceConfig,
    ) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden_size;
        let mut normed = vec![0.0_f32; h];
        rms_norm_model(x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed, cfg)?;

        let qkv_out_len = weight_output_dim(&layer.attn_qkv, h);
        let value_dim = weight_output_dim(&layer.attn_gate, h);
        if qkv_out_len == 0 || value_dim == 0 {
            return Ok(vec![0.0_f32; h]);
        }
        let key_dim = qkv_out_len.saturating_sub(value_dim) / 2;
        let num_v_heads = layer.ssm_a.len().max(1);
        let head_v_dim = if layer.ssm_norm.len() > 1 {
            layer.ssm_norm.len()
        } else if value_dim.is_multiple_of(num_v_heads) {
            value_dim / num_v_heads
        } else {
            return Err(ModelError::InferenceFailed(format!(
                "layer {layer_idx} GDN: cannot infer head_v_dim (value_dim={value_dim}, num_v_heads={num_v_heads})"
            )));
        };
        if weight_is_empty(&layer.ssm_alpha) {
            return Err(ModelError::InferenceFailed(format!(
                "layer {layer_idx} GDN missing ssm_alpha (in_proj_a) weights"
            )));
        }
        let num_k_heads = if head_v_dim > 0 && key_dim >= head_v_dim {
            key_dim / head_v_dim
        } else {
            1
        };
        let head_k_dim = if num_k_heads > 0 {
            key_dim / num_k_heads
        } else {
            head_v_dim
        };
        let head_repeat = num_v_heads / num_k_heads.max(1);

        let mut mixed_qkv = vec![0.0_f32; qkv_out_len];
        gemv_weight(&layer.attn_qkv, qkv_out_len, h, &normed, &mut mixed_qkv)
            .map_err(|e| ModelError::InferenceFailed(format!("attn_qkv: {:?}", e)))?;

        let conv_kernel = 4_usize;
        let mut conv_out = vec![0.0_f32; qkv_out_len];
        if !layer.ssm_conv1d.is_empty() && layer.ssm_conv1d.len() == conv_kernel * qkv_out_len {
            if self.ssm_conv_buffers[layer_idx].dim != qkv_out_len {
                self.ssm_conv_buffers[layer_idx] = ConvHistoryRing::new(conv_kernel, qkv_out_len);
            }
            let buffer = &self.ssm_conv_buffers[layer_idx];
            for c in 0..qkv_out_len {
                let mut sum = 0.0_f32;
                // Weights are tap-major [kernel, channels]; newest input uses the last tap.
                sum += layer.ssm_conv1d[(conv_kernel - 1) * qkv_out_len + c] * mixed_qkv[c];
                for b in 1..conv_kernel {
                    if let Some(prev) = buffer.past_frame(b) {
                        let weight_idx = (conv_kernel - 1 - b) * qkv_out_len + c;
                        sum += layer.ssm_conv1d[weight_idx] * prev[c];
                    }
                }
                conv_out[c] = sum;
            }
            self.ssm_conv_buffers[layer_idx].push(&mixed_qkv);
        } else {
            conv_out.copy_from_slice(&mixed_qkv);
        }

        for val in conv_out.iter_mut() {
            *val *= 1.0_f32 / (1.0_f32 + (-*val).exp());
        }

        debug_vec(&format!("layer {layer_idx} gdn conv"), &conv_out);

        let mut b = vec![0.0_f32; num_v_heads];
        if !weight_is_empty(&layer.ssm_beta) {
            gemv_weight(&layer.ssm_beta, num_v_heads, h, &normed, &mut b)
                .map_err(|e| ModelError::InferenceFailed(format!("ssm_beta: {:?}", e)))?;
        }
        let mut a = vec![0.0_f32; num_v_heads];
        gemv_weight(&layer.ssm_alpha, num_v_heads, h, &normed, &mut a)
            .map_err(|e| ModelError::InferenceFailed(format!("ssm_alpha: {:?}", e)))?;

        let mut z = vec![0.0_f32; value_dim];
        gemv_weight(&layer.attn_gate, value_dim, h, &normed, &mut z)
            .map_err(|e| ModelError::InferenceFailed(format!("attn_gate: {:?}", e)))?;

        let state_elems = num_v_heads * head_k_dim * head_v_dim;
        if self.ssm_states[layer_idx].len() != state_elems {
            self.ssm_states[layer_idx] = vec![0.0_f32; state_elems];
        }
        let state = &mut self.ssm_states[layer_idx];

        let q_scale = 1.0_f32 / (head_k_dim as f32).sqrt();
        let mut core_out = vec![0.0_f32; value_dim];

        // Heads are independent (disjoint state and output chunks), so run the
        // delta-rule update in parallel across v_heads. Body is numerically
        // identical to the previous sequential loop.
        let head_state = head_k_dim * head_v_dim;
        let conv_out_ref = &conv_out;
        let state_base = state.as_mut_ptr() as usize;
        let out_base = core_out.as_mut_ptr() as usize;
        crate::spinpool::run_chunks(num_v_heads, |v_head| {
            let state_h = unsafe {
                std::slice::from_raw_parts_mut(
                    (state_base as *mut f32).add(v_head * head_state),
                    head_state,
                )
            };
            let out_h = unsafe {
                std::slice::from_raw_parts_mut(
                    (out_base as *mut f32).add(v_head * head_v_dim),
                    head_v_dim,
                )
            };
            {
                let k_head = v_head / head_repeat;
                let q_off = k_head * head_k_dim;
                let k_off = key_dim + k_head * head_k_dim;
                let v_off = key_dim * 2 + v_head * head_v_dim;

                let mut q = conv_out_ref[q_off..q_off + head_k_dim].to_vec();
                let mut k = conv_out_ref[k_off..k_off + head_k_dim].to_vec();
                l2_normalize(&mut q);
                l2_normalize(&mut k);
                for x in q.iter_mut() {
                    *x *= q_scale;
                }

                let v = &conv_out_ref[v_off..v_off + head_v_dim];
                let beta = sigmoid(b[v_head]);
                let a_val = a[v_head];
                let dt = if v_head < layer.ssm_dt_bias.len() {
                    softplus(a_val + layer.ssm_dt_bias[v_head])
                } else {
                    softplus(a_val)
                };
                let a_log = if v_head < layer.ssm_a.len() {
                    layer.ssm_a[v_head]
                } else {
                    0.0_f32
                };
                let g = -(a_log.exp()) * dt;
                let decay = g.exp();

                for s in state_h.iter_mut() {
                    *s *= decay;
                }

                let mut kv_mem = vec![0.0_f32; head_v_dim];
                for j in 0..head_v_dim {
                    let mut sum = 0.0_f32;
                    for i in 0..head_k_dim {
                        sum += state_h[i * head_v_dim + j] * k[i];
                    }
                    kv_mem[j] = sum;
                }

                let mut delta = vec![0.0_f32; head_v_dim];
                for j in 0..head_v_dim {
                    delta[j] = (v[j] - kv_mem[j]) * beta;
                }

                for i in 0..head_k_dim {
                    for j in 0..head_v_dim {
                        state_h[i * head_v_dim + j] += k[i] * delta[j];
                    }
                }

                for j in 0..head_v_dim {
                    let mut sum = 0.0_f32;
                    for i in 0..head_k_dim {
                        sum += state_h[i * head_v_dim + j] * q[i];
                    }
                    out_h[j] = sum;
                }
            }
        });

        debug_vec(&format!("layer {layer_idx} gdn core"), &core_out);

        if !layer.ssm_norm.is_empty() && layer.ssm_norm.len() == head_v_dim {
            for head in 0..num_v_heads {
                let start = head * head_v_dim;
                let end = start + head_v_dim;
                gated_rms_norm(
                    &mut core_out[start..end],
                    &layer.ssm_norm,
                    &z[start..end],
                    cfg.rms_norm_eps,
                );
            }
        }

        debug_vec(&format!("layer {layer_idx} gdn normed"), &core_out);

        let mut residual = vec![0.0_f32; h];
        if !weight_is_empty(&layer.ssm_out) {
            let out_len = weight_output_dim(&layer.ssm_out, value_dim);
            if out_len > 0 {
                let mut projected = vec![0.0_f32; out_len];
                gemv_weight(
                    &layer.ssm_out,
                    out_len,
                    value_dim,
                    &core_out,
                    &mut projected,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ssm_out: {:?}", e)))?;
                let copy_len = h.min(projected.len());
                residual[..copy_len].copy_from_slice(&projected[..copy_len]);
            }
        } else {
            let copy_len = h.min(core_out.len());
            residual[..copy_len].copy_from_slice(&core_out[..copy_len]);
        }
        Ok(residual)
    }

    /// Batched [`Self::run_mamba_layer`]: processes `kk` consecutive tokens
    /// through one GDN layer with the dense projections (qkv / beta / alpha /
    /// gate / out) computed as fused batch GEMMs so each weight block is read
    /// once per window instead of once per token. The conv ring and delta-rule
    /// recurrence stay sequential over tokens (per head), preserving the exact
    /// math of the single-token path.
    pub(super) fn run_mamba_layer_batch(
        &mut self,
        layer_idx: usize,
        layer: &LayerWeights,
        xs: &[f32],
        kk: usize,
        cfg: &InferenceConfig,
    ) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden_size;
        let mut normed_all = vec![0.0_f32; kk * h];
        for t in 0..kk {
            let mut normed = vec![0.0_f32; h];
            rms_norm_model(
                &xs[t * h..(t + 1) * h],
                &layer.attn_norm,
                cfg.rms_norm_eps,
                &mut normed,
                cfg,
            )?;
            normed_all[t * h..(t + 1) * h].copy_from_slice(&normed);
        }

        let qkv_out_len = weight_output_dim(&layer.attn_qkv, h);
        let value_dim = weight_output_dim(&layer.attn_gate, h);
        if qkv_out_len == 0 || value_dim == 0 {
            return Ok(vec![0.0_f32; kk * h]);
        }
        let key_dim = qkv_out_len.saturating_sub(value_dim) / 2;
        let num_v_heads = layer.ssm_a.len().max(1);
        let head_v_dim = if layer.ssm_norm.len() > 1 {
            layer.ssm_norm.len()
        } else if value_dim.is_multiple_of(num_v_heads) {
            value_dim / num_v_heads
        } else {
            return Err(ModelError::InferenceFailed(format!(
                "layer {layer_idx} GDN: cannot infer head_v_dim (value_dim={value_dim}, num_v_heads={num_v_heads})"
            )));
        };
        if weight_is_empty(&layer.ssm_alpha) {
            return Err(ModelError::InferenceFailed(format!(
                "layer {layer_idx} GDN missing ssm_alpha (in_proj_a) weights"
            )));
        }
        let num_k_heads = if head_v_dim > 0 && key_dim >= head_v_dim {
            key_dim / head_v_dim
        } else {
            1
        };
        let head_k_dim = key_dim.checked_div(num_k_heads).unwrap_or(head_v_dim);
        let head_repeat = num_v_heads / num_k_heads.max(1);

        // Batched dense projections — all share `normed_all` as input.
        let mut mixed_all = vec![0.0_f32; kk * qkv_out_len];
        gemm_weight(
            &layer.attn_qkv,
            qkv_out_len,
            h,
            &normed_all,
            &mut mixed_all,
            kk,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("attn_qkv: {:?}", e)))?;
        let mut b_all = vec![0.0_f32; kk * num_v_heads];
        if !weight_is_empty(&layer.ssm_beta) {
            gemm_weight(&layer.ssm_beta, num_v_heads, h, &normed_all, &mut b_all, kk)
                .map_err(|e| ModelError::InferenceFailed(format!("ssm_beta: {:?}", e)))?;
        }
        let mut a_all = vec![0.0_f32; kk * num_v_heads];
        gemm_weight(
            &layer.ssm_alpha,
            num_v_heads,
            h,
            &normed_all,
            &mut a_all,
            kk,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("ssm_alpha: {:?}", e)))?;
        let mut z_all = vec![0.0_f32; kk * value_dim];
        gemm_weight(&layer.attn_gate, value_dim, h, &normed_all, &mut z_all, kk)
            .map_err(|e| ModelError::InferenceFailed(format!("attn_gate: {:?}", e)))?;

        // Causal conv + SiLU: sequential over tokens (ring buffer recurrence).
        let conv_kernel = 4_usize;
        let mut conv_all = vec![0.0_f32; kk * qkv_out_len];
        for t in 0..kk {
            let mixed = &mixed_all[t * qkv_out_len..(t + 1) * qkv_out_len];
            let conv_out = &mut conv_all[t * qkv_out_len..(t + 1) * qkv_out_len];
            if !layer.ssm_conv1d.is_empty() && layer.ssm_conv1d.len() == conv_kernel * qkv_out_len {
                if self.ssm_conv_buffers[layer_idx].dim != qkv_out_len {
                    self.ssm_conv_buffers[layer_idx] =
                        ConvHistoryRing::new(conv_kernel, qkv_out_len);
                }
                let buffer = &self.ssm_conv_buffers[layer_idx];
                // llama.cpp-converted GGUFs store ssm_conv1d as {kernel, channels}
                // (kernel contiguous → offset c*kernel + tap); oxidize's own
                // converter stores {channels, kernel} (tap-major → tap*ch + c).
                let chan_major = std::env::var_os("OXIDIZE_CONV_CHAN_MAJOR").is_some();
                let widx = |tap: usize, c: usize| {
                    if chan_major {
                        c * conv_kernel + tap
                    } else {
                        tap * qkv_out_len + c
                    }
                };
                for c in 0..qkv_out_len {
                    let mut sum = layer.ssm_conv1d[widx(conv_kernel - 1, c)] * mixed[c];
                    for b in 1..conv_kernel {
                        if let Some(prev) = buffer.past_frame(b) {
                            sum += layer.ssm_conv1d[widx(conv_kernel - 1 - b, c)] * prev[c];
                        }
                    }
                    conv_out[c] = sum;
                }
                self.ssm_conv_buffers[layer_idx].push(mixed);
            } else {
                conv_out.copy_from_slice(mixed);
            }
            for val in conv_out.iter_mut() {
                *val *= 1.0_f32 / (1.0_f32 + (-*val).exp());
            }
        }

        let state_elems = num_v_heads * head_k_dim * head_v_dim;
        if self.ssm_states[layer_idx].len() != state_elems {
            self.ssm_states[layer_idx] = vec![0.0_f32; state_elems];
        }
        let state = &mut self.ssm_states[layer_idx];

        let q_scale = 1.0_f32 / (head_k_dim as f32).sqrt();

        // Delta-rule recurrence: parallel over heads, sequential over tokens
        // within each head. Output is head-major scratch, transposed below.
        let head_state = head_k_dim * head_v_dim;
        let conv_ref = &conv_all;
        let b_ref = &b_all;
        let a_ref = &a_all;
        let mut core_head_major = vec![0.0_f32; num_v_heads * kk * head_v_dim];
        state
            .par_chunks_mut(head_state)
            .zip(core_head_major.par_chunks_mut(kk * head_v_dim))
            .enumerate()
            .for_each(|(v_head, (state_h, out_h))| {
                let k_head = v_head / head_repeat;
                let q_off = k_head * head_k_dim;
                let k_off = key_dim + k_head * head_k_dim;
                let v_off = key_dim * 2 + v_head * head_v_dim;
                let a_log = if v_head < layer.ssm_a.len() {
                    layer.ssm_a[v_head]
                } else {
                    0.0_f32
                };

                for t in 0..kk {
                    let conv_out = &conv_ref[t * qkv_out_len..(t + 1) * qkv_out_len];
                    let mut q = conv_out[q_off..q_off + head_k_dim].to_vec();
                    let mut k = conv_out[k_off..k_off + head_k_dim].to_vec();
                    l2_normalize(&mut q);
                    l2_normalize(&mut k);
                    // llama.cpp's GATED_DELTA_NET L2-norms q,k with NO 1/sqrt(d)
                    // scale. Applying q_scale shrinks the core into the
                    // eps-dominated regime of the per-head gated RMS norm,
                    // breaking normalization. OXIDIZE_NO_QSCALE=1 disables it.
                    if std::env::var_os("OXIDIZE_NO_QSCALE").is_none() {
                        for x in q.iter_mut() {
                            *x *= q_scale;
                        }
                    }

                    let v = &conv_out[v_off..v_off + head_v_dim];
                    let beta = sigmoid(b_ref[t * num_v_heads + v_head]);
                    let a_val = a_ref[t * num_v_heads + v_head];
                    let dt = if v_head < layer.ssm_dt_bias.len() {
                        softplus(a_val + layer.ssm_dt_bias[v_head])
                    } else {
                        softplus(a_val)
                    };
                    // Raw A_log (oxidize converter): A = -exp(A_log). Baked A
                    // (llama.cpp converter): ssm_a already stores A (negative),
                    // use directly. OXIDIZE_SSM_A_DIRECT=1 selects baked mode.
                    let g = if std::env::var_os("OXIDIZE_SSM_A_DIRECT").is_some() {
                        a_log * dt
                    } else {
                        -(a_log.exp()) * dt
                    };
                    let decay = g.exp();

                    for s in state_h.iter_mut() {
                        *s *= decay;
                    }

                    let mut kv_mem = vec![0.0_f32; head_v_dim];
                    for j in 0..head_v_dim {
                        let mut sum = 0.0_f32;
                        for i in 0..head_k_dim {
                            sum += state_h[i * head_v_dim + j] * k[i];
                        }
                        kv_mem[j] = sum;
                    }

                    let mut delta = vec![0.0_f32; head_v_dim];
                    for j in 0..head_v_dim {
                        delta[j] = (v[j] - kv_mem[j]) * beta;
                    }

                    for i in 0..head_k_dim {
                        for j in 0..head_v_dim {
                            state_h[i * head_v_dim + j] += k[i] * delta[j];
                        }
                    }

                    let out_t = &mut out_h[t * head_v_dim..(t + 1) * head_v_dim];
                    for j in 0..head_v_dim {
                        let mut sum = 0.0_f32;
                        for i in 0..head_k_dim {
                            sum += state_h[i * head_v_dim + j] * q[i];
                        }
                        out_t[j] = sum;
                    }
                }
            });

        // Transpose head-major scratch into token-major core output.
        let mut core_all = vec![0.0_f32; kk * value_dim];
        for v_head in 0..num_v_heads {
            for t in 0..kk {
                let src = v_head * kk * head_v_dim + t * head_v_dim;
                let dst = t * value_dim + v_head * head_v_dim;
                core_all[dst..dst + head_v_dim]
                    .copy_from_slice(&core_head_major[src..src + head_v_dim]);
            }
        }

        if layer_idx == 0 && crate::inference::trace_vals_enabled() {
            let mabs = |v: &[f32]| v.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
            // Locate the outlier element of token-0 core and dump its factors.
            let (mut bi, mut bv) = (0usize, 0.0_f32);
            for (i, &x) in core_all[..value_dim.min(core_all.len())].iter().enumerate() {
                if x.abs() > bv {
                    bv = x.abs();
                    bi = i;
                }
            }
            let v_head = bi / head_v_dim;
            let j = bi % head_v_dim;
            let k_head = v_head / head_repeat.max(1);
            // Recompute q,k (post conv+silu, l2norm, q_scale) for this head, t=0.
            let conv0 = &conv_all[..qkv_out_len];
            let q_off = k_head * head_k_dim;
            let k_off = key_dim + k_head * head_k_dim;
            let v_off = key_dim * 2 + v_head * head_v_dim;
            let mut q = conv0[q_off..q_off + head_k_dim].to_vec();
            let mut k = conv0[k_off..k_off + head_k_dim].to_vec();
            l2_normalize(&mut q);
            l2_normalize(&mut k);
            for x in q.iter_mut() {
                *x *= 1.0_f32 / (head_k_dim as f32).sqrt();
            }
            let kq: f32 = k.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            let vval = conv0[v_off + j];
            let beta = sigmoid(b_all[v_head]);
            let ssum = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
            // head0 t0 raw conv slices for direct comparison to llama:
            //   llama v head0=[-0.0004,0.0526,0.0150]  q(l2)=[-0.0139,0.0896,-0.0231]
            let mut q0 = conv0[..head_k_dim].to_vec();
            let mut k0 = conv0[key_dim..key_dim + head_k_dim].to_vec();
            l2_normalize(&mut q0);
            l2_normalize(&mut k0);
            eprintln!(
                "GDN L0 head0 t0: v_raw={:?} q_l2={:?} k_l2={:?}",
                &conv0[key_dim * 2..key_dim * 2 + 4],
                &q0[..4],
                &k0[..4],
            );
            eprintln!(
                "GDN L0 head0 t0: core_pre(=attn_output)[0..6]={:?} (llama [-0.0000,0.0001,0.0000,..])",
                &core_all[..6.min(core_all.len())],
            );
            // head46 factors: v, k·q, beta — diagnose higher-head collapse
            for &vh in &[1usize, 46usize] {
                let kh = vh / head_repeat.max(1);
                let qo = kh * head_k_dim;
                let ko = key_dim + kh * head_k_dim;
                let vo = key_dim * 2 + vh * head_v_dim;
                let mut qh = conv0[qo..qo + head_k_dim].to_vec();
                let mut kh2 = conv0[ko..ko + head_k_dim].to_vec();
                l2_normalize(&mut qh);
                l2_normalize(&mut kh2);
                for x in qh.iter_mut() {
                    *x *= 1.0_f32 / (head_k_dim as f32).sqrt();
                }
                let kqv: f32 = kh2.iter().zip(qh.iter()).map(|(a, b)| a * b).sum();
                // q,k post-l2norm (pre q_scale) for comparison to llama
                let mut qn = conv0[qo..qo + head_k_dim].to_vec();
                let mut kn = conv0[ko..ko + head_k_dim].to_vec();
                l2_normalize(&mut qn);
                l2_normalize(&mut kn);
                let zh = vh * head_v_dim;
                let zslice = &z_all[zh..zh + 3];
                let silu0 = zslice[0] * (1.0 / (1.0 + (-zslice[0]).exp()));
                eprintln!(
                    "GDN L0 v_head={vh} k_head={kh}: k·q={:.6} beta={:.5} z[0..3]={:?} silu(z0)={:.4} qn[0..3]={:?} kn[0..3]={:?}",
                    kqv,
                    sigmoid(b_all[vh]),
                    zslice,
                    silu0,
                    &qn[..3],
                    &kn[..3],
                );
                let _ = (qh, kh2, &conv0[vo..vo + 3]);
            }
            eprintln!(
                "GDN L0 t0 OUTLIER: idx={bi} v_head={v_head} j={j} core={bv:.5} | v={vval:.5} beta={beta:.5} k·q={kq:.6} | conv_v_max={:.4} conv_q_max={:.4} z_max={:.4} ssm_norm[0]={:.4}",
                mabs(&conv0[key_dim * 2..qkv_out_len]),
                mabs(&conv0[..key_dim]),
                mabs(&z_all[..value_dim.min(z_all.len())]),
                layer.ssm_norm.first().copied().unwrap_or(0.0),
            );
            eprintln!(
                "GDN L0 SUMS (vs llama conv=4714 gdn_out=97 z=-35772 node55=-29.6): conv={:.1} core_pre={:.2} z={:.1}",
                ssum(&conv_all),
                ssum(&core_all),
                ssum(&z_all),
            );
        }
        if !layer.ssm_norm.is_empty() && layer.ssm_norm.len() == head_v_dim {
            for t in 0..kk {
                for head in 0..num_v_heads {
                    let start = t * value_dim + head * head_v_dim;
                    let end = start + head_v_dim;
                    gated_rms_norm(
                        &mut core_all[start..end],
                        &layer.ssm_norm,
                        &z_all[start..end],
                        cfg.rms_norm_eps,
                    );
                }
            }
        }
        if layer_idx == 0 && crate::inference::trace_vals_enabled() {
            let _mabs = |v: &[f32]| v.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
            let _ssum = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
            let hd = head_v_dim;
            eprintln!(
                "GDN L0 core_post head0={:?} head46={:?} head47={:?} (llama h46[-0.0044,-0.0048,0.0012] h47[-0.0035,-0.0000,-0.0012])",
                &core_all[..3.min(core_all.len())],
                &core_all[46 * hd..46 * hd + 3],
                &core_all[47 * hd..47 * hd + 3],
            );
            // llama node_55 rows: head0 [0.0001,-0.0030,-0.0008] head1 [-0.0003,-0.0091,-0.0027]
        }

        let mut residual_all = vec![0.0_f32; kk * h];
        if !weight_is_empty(&layer.ssm_out) {
            let out_len = weight_output_dim(&layer.ssm_out, value_dim);
            if out_len > 0 {
                let mut proj_all = vec![0.0_f32; kk * out_len];
                gemm_weight(
                    &layer.ssm_out,
                    out_len,
                    value_dim,
                    &core_all,
                    &mut proj_all,
                    kk,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ssm_out: {:?}", e)))?;
                let copy_len = h.min(out_len);
                for t in 0..kk {
                    residual_all[t * h..t * h + copy_len]
                        .copy_from_slice(&proj_all[t * out_len..t * out_len + copy_len]);
                }
            }
        } else {
            let copy_len = h.min(value_dim);
            for t in 0..kk {
                residual_all[t * h..t * h + copy_len]
                    .copy_from_slice(&core_all[t * value_dim..t * value_dim + copy_len]);
            }
        }
        if layer_idx == 0 && crate::inference::trace_vals_enabled() {
            eprintln!(
                "GDN L0 residual(=linear_attn_out) t0[0..6]={:?} (llama [-0.0381,-0.0049,-0.0200,..])",
                &residual_all[..6.min(residual_all.len())],
            );
        }
        Ok(residual_all)
    }
}
