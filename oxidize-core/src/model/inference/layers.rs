use super::*;

/// On-device attention when gpu_native is active (default ON). Set `OX_GPU_ATTN=0`
/// to force CPU attention for parity debugging.
pub(super) fn ox_gpu_attn_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("OX_GPU_ATTN") {
        Ok(v) if v == "0" || v.is_empty() => false,
        Ok(_) => true,
        Err(_) => true,
    })
}

/// Cached check for true continuous-batching decode (env `OX_BATCHED_DECODE`).
/// When set, the paged runtime / batched-decode bench may run N decode tokens
/// (one per running sequence) through [`InferenceModel::forward_batch`] as a
/// single set of batched GEMMs, amortizing the weight reads across sequences.
///
/// OFF by default → `forward_batch` is only reachable through the bench or a
/// future flagged runtime branch, so every existing path stays byte-identical.
/// CPU/backend-agnostic: when `OX_GPU_ATTN` is set it owns the device KV cache
/// and takes precedence; `OX_BATCHED_DECODE` only governs the host-side path.
pub fn ox_batched_decode_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("OX_BATCHED_DECODE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    })
}

impl InferenceModel {
    /// Return the raw quantized byte slice of a Q4K weight matrix, or `None`
    /// if the storage variant is not Q4K (S or M) quantised.
    #[allow(dead_code)]
    pub(crate) fn q4k_bytes(ws: &WeightStorage) -> Option<&[u8]> {
        match ws {
            WeightStorage::Quantized(
                GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M,
                data,
            ) => Some(data.as_slice()),
            WeightStorage::MmapQuantized(
                GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M,
                mmap,
                offset,
                size,
            ) => Some(&mmap[*offset..*offset + *size]),
            _ => None,
        }
    }

    /// Like `q4k_bytes` but also accepts Q6_K (used in mixed-precision models such as Q4_K_M).
    #[allow(dead_code)]
    pub(super) fn q4k_or_q6k_bytes(ws: &WeightStorage) -> Option<&[u8]> {
        match ws {
            WeightStorage::Quantized(
                GgufQuantizationType::Q4_K_S
                | GgufQuantizationType::Q4_K_M
                | GgufQuantizationType::Q6_K,
                data,
            ) => Some(data.as_slice()),
            WeightStorage::MmapQuantized(
                GgufQuantizationType::Q4_K_S
                | GgufQuantizationType::Q4_K_M
                | GgufQuantizationType::Q6_K,
                mmap,
                offset,
                size,
            ) => Some(&mmap[*offset..*offset + *size]),
            _ => None,
        }
    }

    /// Returns true if ALL weights required for the standard attention + dense
    /// FFN path of `layer` are Q4K-or-Q6K quantised and the runtime has an
    /// active CUDA device.  Layers that fail this check fall back to the CPU path.
    #[cfg(feature = "cuda")]
    pub(crate) fn layer_can_use_gpu_native(layer: &LayerWeights, cfg: &InferenceConfig) -> bool {
        // Must be a plain attention layer (not shortconv / Mamba / MLA).
        if !layer.shortconv_in_proj.is_empty()
            || !layer.attn_qkv.is_empty()
            || !layer.mla_kv_a_mqa.is_empty()
        {
            return false;
        }
        if layer.attn_q.is_empty() || layer.attn_k.is_empty() || layer.attn_v.is_empty() {
            return false;
        }
        // No attention biases handled yet.
        if !layer.attn_q_bias.is_empty()
            || !layer.attn_k_bias.is_empty()
            || !layer.attn_v_bias.is_empty()
        {
            return false;
        }
        // Per-head Q/K norm (Qwen3-style QK-RMSNorm). The fused GPU path applies
        // per-head [head_dim] rms_norm before RoPE in gpu_attn_block_fused_q4k.
        // Without this, Qwen3 models fall back to the full CPU decode path (~10×
        // slower than llama.cpp on GPU). Set `OX_GPU_QK_NORM=0` to force CPU.
        if !layer.attn_q_norm.is_empty() || !layer.attn_k_norm.is_empty() {
            let qk_norm_gpu = std::env::var("OX_GPU_QK_NORM")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(true);
            if !qk_norm_gpu {
                return false;
            }
            // Opt-in path: only per-head [head_dim] norms are representable.
            let n_q = cfg.num_attention_heads.max(1);
            let n_kv = cfg.num_key_value_heads.max(1);
            let q_hd = layer.attn_q.output_dim(cfg.hidden_size) / n_q;
            let kv_hd = if !layer.attn_k.is_empty() {
                layer.attn_k.output_dim(cfg.hidden_size) / n_kv
            } else {
                0
            };
            if !layer.attn_q_norm.is_empty() && layer.attn_q_norm.len() != q_hd {
                return false;
            }
            if !layer.attn_k_norm.is_empty() && layer.attn_k_norm.len() != kv_hd {
                return false;
            }
        }
        // Dense FFN required.
        if layer.ffn_gate.is_empty() || layer.ffn_up.is_empty() || layer.ffn_down.is_empty() {
            return false;
        }
        // Must not use GeGLU (SwiGLU only — our silu_mul_f32_kernel implements that).
        if cfg.gelu_ffn {
            return false;
        }
        // All projections must be Q4K or Q6K (mixed-precision models like Q4_K_M use Q6K for some tensors).
        for ws in [
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.ffn_gate,
            &layer.ffn_up,
            &layer.ffn_down,
        ] {
            if Self::q4k_or_q6k_bytes(ws).is_none() {
                return false;
            }
        }
        // Hidden size must be aligned to a Q4K block (256 values per block).
        if !cfg.hidden_size.is_multiple_of(256) || !cfg.intermediate_size.is_multiple_of(256) {
            return false;
        }
        // Attention projection sizes must also be aligned (Q/K have different sizes in GQA).
        let q_len = layer.attn_q.output_dim(cfg.hidden_size);
        let kv_len = if !layer.attn_k.is_empty() {
            layer.attn_k.output_dim(cfg.hidden_size)
        } else {
            256
        };
        let attn_out_len = if !layer.attn_output.is_empty() {
            layer.attn_output.output_dim(cfg.hidden_size)
        } else {
            256
        };
        if !q_len.is_multiple_of(256)
            || !kv_len.is_multiple_of(256)
            || !attn_out_len.is_multiple_of(256)
        {
            return false;
        }
        // CUDA must be active at runtime.
        crate::gpu_dispatch::active_gpu().is_some()
    }

    /// Run layers `range` against the hidden state currently in
    /// `workspace.x[..hidden_size]`, mutating it in place. `pos` is the
    /// absolute position for KV cache writes / RoPE.
    pub fn run_layer_range_in_workspace(
        &mut self,
        pos: usize,
        range: std::ops::Range<usize>,
    ) -> Result<(), ModelError> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let k = cfg.num_key_value_heads;
        // Silence unused warnings for paths that don't reference both names.
        let _ = (n, k);

        // GPU-native path: keep the hidden state resident on the GPU across all
        // layers in this range.  The CPU only touches q/k/v (for rope + KV cache
        // + attention) and the attention result (for the wo upload).  All other
        // memory traffic (rms_norm, gate/up/silu/down/residual) stays on the GPU,
        // reducing CPU↔GPU round-trips from 252 to ~56 per token for a 28-layer
        // Llama-3 style model with Q4_K_M weights.
        #[cfg(feature = "cuda")]
        let gpu_native: bool = {
            let all_eligible = range.clone().all(|i| {
                self.layers
                    .get(i)
                    .is_some_and(|l| Self::layer_can_use_gpu_native(l, cfg))
            });
            if all_eligible && !range.is_empty() {
                match crate::cuda::gpu_init_activation_buffers(h, cfg.intermediate_size) {
                    Ok(()) => {
                        #[cfg(feature = "cuda")]
                        let embed_ok = self
                            .pending_embed_token
                            .take()
                            .and_then(|token| {
                                crate::cuda::gpu_embed_token(
                                    &self.tok_embeddings,
                                    h,
                                    cfg.vocab_size,
                                    token,
                                    cfg.embedding_scale,
                                )
                                .ok()
                            })
                            .is_some();
                        #[cfg(feature = "cuda")]
                        let upload_ok = if embed_ok {
                            true
                        } else {
                            crate::cuda::gpu_upload_hidden(&self.workspace.x[..h]).is_ok()
                        };
                        #[cfg(not(feature = "cuda"))]
                        let upload_ok = false;
                        upload_ok
                    }
                    Err(_) => false,
                }
            } else {
                false
            }
        };
        #[cfg(not(feature = "cuda"))]
        let gpu_native = false;

        // On-device attention when gpu_native is active: the hybrid path that
        // downloads Q/K/V to CPU for attention costs ~32 stream syncs per token
        // (one per layer) and is the dominant decode bottleneck vs llama.cpp.
        // When all layers are gpu_native-eligible, run RoPE + KV + flash-decode on
        // the GPU by default. Set OX_GPU_ATTN=0 to force the legacy CPU-attention
        // path for parity debugging.
        #[cfg(feature = "cuda")]
        let gpu_attn: bool = if gpu_native && ox_gpu_attn_enabled() {
            let kv_cfg = self.kv_cache.config();
            let kv_token_size = kv_cfg.token_size();
            // `gpu_kv_init` runs once per decode token but only (re)allocates on a
            // geometry change; it deliberately leaves `kv_seq_len` MONOTONIC. The
            // new-sequence reset is driven HERE, gated on `pos == 0`, so a fresh
            // prompt never reads stale F16 rows left by a prior generation.
            match crate::cuda::gpu_kv_init(kv_cfg.layer_count, kv_token_size, kv_cfg.context_size) {
                Ok(()) => {
                    if pos == 0 {
                        // Hard error: a reset failure means the device cache state
                        // is undefined; do NOT silently fall through to a host
                        // cache the GPU path never populates.
                        crate::cuda::gpu_kv_reset().map_err(|e| {
                            ModelError::InferenceFailed(format!("gpu_kv_reset: {e}"))
                        })?;
                        let _ = crate::cuda::gpu_decode_graph_reset(kv_cfg.layer_count);
                    }
                    true
                }
                Err(_) => false,
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let gpu_attn = false;
        let _ = gpu_attn;

        #[cfg(feature = "cuda")]
        if gpu_native && pos > 0 {
            let _ = crate::cuda::gpu_decode_graph_set_token(
                pos as u32,
                cfg.context_size as u32,
                self.pending_embed_token.unwrap_or(0) as u32,
            );
        }

        let ws = &mut self.workspace;

        for layer_idx in range {
            let layer = &self.layers[layer_idx];

            // Detect LFM2 short-convolution layers (have shortconv.in_proj, no attention).
            let is_shortconv = !layer.shortconv_in_proj.is_empty();
            // Detect Mamba layers (have attn_qkv but no attn_q)
            let is_mamba = !layer.attn_qkv.is_empty() && layer.attn_q.is_empty();

            // Determine which norm weight to use for FFN
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

            // Per-layer RoPE theta and sliding-window size (Gemma local/global mix).
            let layer_rope = cfg.layer_rope_theta(layer_idx);
            let layer_window = cfg.layer_sliding_window(layer_idx);

            #[cfg(feature = "cuda")]
            let mut decode_graph_launched = false;
            #[cfg(not(feature = "cuda"))]
            let decode_graph_launched = false;

            if is_shortconv {
                // ---- LFM2 short-convolution token mixing ----
                // operator_norm -> in_proj -> (B,C,x) -> Bx=B*x ->
                // causal depthwise conv1d (kernel = l_cache) -> y=C*conv -> out_proj
                let l_cache = cfg.shortconv_l_cache.max(1);
                let d = layer.shortconv_in_proj.output_dim(h) / 3;

                let normed = &mut ws.hidden_b[..h];
                normed.fill(0.0_f32);
                rms_norm_f32(&ws.x[..h], &layer.attn_norm, cfg.rms_norm_eps, normed)
                    .map_err(|e| ModelError::InferenceFailed(format!("shortconv_norm: {:?}", e)))?;
                let bcx = &mut ws.shortconv_bcx[..3 * d];
                bcx.fill(0.0_f32);
                gemv_weight(&layer.shortconv_in_proj, 3 * d, h, normed, bcx).map_err(|e| {
                    ModelError::InferenceFailed(format!("shortconv_in_proj: {:?}", e))
                })?;

                // Bx = B * x   (B = bcx[0..d], C = bcx[d..2d], x = bcx[2d..3d])
                let bx = &mut ws.shortconv_bx[..d];
                for i in 0..d {
                    bx[i] = bcx[i] * bcx[2 * d + i];
                }

                // Causal depthwise conv1d. Weights laid out [l_cache, d] tap-major;
                // the last tap aligns with the current token (llama.cpp ssm_conv order).
                let conv_out = &mut ws.conv_out[..d];
                let have_conv = layer.shortconv_conv.len() == l_cache * d;
                {
                    let buf = &self.ssm_conv_buffers[layer_idx];
                    if have_conv {
                        // Weights are channel-major: [d, l_cache] with the l_cache
                        // taps contiguous per channel. Last tap = current token.
                        for c in 0..d {
                            let base = c * l_cache;
                            let mut sum = layer.shortconv_conv[base + (l_cache - 1)] * bx[c];
                            for j in 1..l_cache {
                                if let Some(prev) = buf.past_frame(j) {
                                    sum += layer.shortconv_conv[base + (l_cache - 1 - j)] * prev[c];
                                }
                            }
                            conv_out[c] = sum;
                        }
                    } else {
                        conv_out.copy_from_slice(bx);
                    }
                }

                // y = C * conv_out
                for i in 0..d {
                    conv_out[i] *= bcx[d + i];
                }

                // out_proj: [d] -> [h]
                let attn_out = &mut ws.hidden_a[..h];
                attn_out.fill(0.0_f32);
                gemv_weight(&layer.shortconv_out_proj, h, d, conv_out, attn_out).map_err(|e| {
                    ModelError::InferenceFailed(format!("shortconv_out_proj: {:?}", e))
                })?;

                self.ssm_conv_buffers[layer_idx].push(bx);

                for i in 0..h {
                    ws.x[i] += attn_out[i];
                }
            } else if is_mamba {
                // ---- Mamba/SSM layer ----
                let mamba_out = {
                    let normed = &mut ws.hidden_a[..h];
                    normed.fill(0.0_f32);
                    rms_norm_f32(&ws.x[..h], &layer.attn_norm, cfg.rms_norm_eps, normed)
                        .map_err(|e| ModelError::InferenceFailed(format!("mamba_norm: {:?}", e)))?;

                    // Gate branch
                    let gate_len = if !layer.attn_gate.is_empty() {
                        layer.attn_gate.output_dim(h)
                    } else {
                        0
                    };
                    if gate_len > 0 {
                        let gate = &mut ws.intermediate_a[..gate_len];
                        gate.fill(0.0_f32);
                        gemv_weight(&layer.attn_gate, gate_len, h, normed, gate).map_err(|e| {
                            ModelError::InferenceFailed(format!("attn_gate: {:?}", e))
                        })?;
                    }

                    // SSM branch projection: [h] -> [qkv_out_len]
                    let qkv_out_len = layer.attn_qkv.output_dim(h);
                    let x_proj = &mut ws.q_full[..qkv_out_len];
                    x_proj.fill(0.0_f32);
                    gemv_weight(&layer.attn_qkv, qkv_out_len, h, normed, x_proj)
                        .map_err(|e| ModelError::InferenceFailed(format!("attn_qkv: {:?}", e)))?;

                    // Causal conv1d over qkv_out_len channels
                    let conv_kernel = 4_usize;
                    let conv_out = &mut ws.conv_out[..qkv_out_len];
                    conv_out.fill(0.0_f32);
                    if !layer.ssm_conv1d.is_empty()
                        && layer.ssm_conv1d.len() == conv_kernel * qkv_out_len
                    {
                        let buffer = &self.ssm_conv_buffers[layer_idx];
                        for c in 0..qkv_out_len {
                            let mut sum = 0.0_f32;
                            // Tap-major [kernel, channels]; newest input uses the last tap.
                            sum +=
                                layer.ssm_conv1d[(conv_kernel - 1) * qkv_out_len + c] * x_proj[c];
                            for b in 1..conv_kernel {
                                if let Some(prev) = buffer.past_frame(b) {
                                    let weight_idx = (conv_kernel - 1 - b) * qkv_out_len + c;
                                    sum += layer.ssm_conv1d[weight_idx] * prev[c];
                                }
                            }
                            conv_out[c] = sum;
                        }
                    } else {
                        conv_out.copy_from_slice(x_proj);
                    }

                    self.ssm_conv_buffers[layer_idx].push(x_proj);

                    // SiLU activation
                    for val in conv_out.iter_mut() {
                        *val = *val * (1.0_f32 / (1.0_f32 + (-*val).exp()));
                    }

                    // Split into SSM input and gate
                    let half = qkv_out_len / 2;
                    let mut mamba_out = vec![0.0_f32; half];
                    let mut x_ssm = conv_out[..half].to_vec();
                    let z_gate: Vec<f32> = if qkv_out_len > half {
                        conv_out[half..].to_vec()
                    } else {
                        vec![0.0_f32; half]
                    };

                    // Group RMSNorm on x_ssm
                    if !layer.ssm_norm.is_empty() && !x_ssm.is_empty() {
                        let group_size = layer.ssm_norm.len();
                        if x_ssm.len().is_multiple_of(group_size) {
                            let num_groups = x_ssm.len() / group_size;
                            for g in 0..num_groups {
                                let start = g * group_size;
                                let end = start + group_size;
                                let mut normed_group = vec![0.0_f32; group_size];
                                rms_norm_f32(
                                    &x_ssm[start..end],
                                    &layer.ssm_norm,
                                    cfg.rms_norm_eps,
                                    &mut normed_group,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("ssm_norm: {:?}", e))
                                })?;
                                x_ssm[start..end].copy_from_slice(&normed_group);
                            }
                        }
                    }

                    // Selective Scan SSM
                    let state_dim = self.ssm_states[layer_idx].len();
                    if state_dim > 0
                        && !layer.ssm_a.is_empty()
                        && !layer.ssm_alpha.is_empty()
                        && !layer.ssm_beta.is_empty()
                    {
                        // Compute Bx: ssm_beta maps x_ssm -> state
                        // ssm_beta is [x_ssm_len, state_dim], stored row-major
                        let mut bx = vec![0.0_f32; state_dim];
                        let x_ssm_len = x_ssm.len();
                        if layer.ssm_beta.len() == x_ssm_len * state_dim {
                            for (j, &x_value) in x_ssm.iter().enumerate().take(x_ssm_len) {
                                for (i, bx_value) in bx.iter_mut().enumerate().take(state_dim) {
                                    *bx_value += layer.ssm_beta[j * state_dim + i] * x_value;
                                }
                            }
                        }

                        // Update state: h = h * exp(A * delta) + Bx * delta
                        for (i, &bx_value) in bx.iter().enumerate().take(state_dim) {
                            let a = layer.ssm_a[i % layer.ssm_a.len()];
                            let a = if std::env::var_os("OXIDIZE_SSM_A_DIRECT").is_some() {
                                a
                            } else {
                                -a.exp()
                            };
                            let dt = if !layer.ssm_dt_bias.is_empty() {
                                let b = layer.ssm_dt_bias[i % layer.ssm_dt_bias.len()];
                                (1.0_f32 + b.exp()).ln() // softplus
                            } else {
                                0.01_f32
                            };
                            let decay = (a * dt).exp();
                            self.ssm_states[layer_idx][i] =
                                self.ssm_states[layer_idx][i] * decay + bx_value * dt;
                        }

                        // Compute output: y = C * h = ssm_alpha * state
                        // ssm_alpha is [y_len, state_dim]
                        let y_len = layer.ssm_alpha.len() / state_dim;
                        let mut y_ssm = vec![0.0_f32; y_len];
                        if layer.ssm_alpha.len() == y_len * state_dim {
                            for (j, y_value) in y_ssm.iter_mut().enumerate().take(y_len) {
                                for (i, &state_value) in self.ssm_states[layer_idx]
                                    .iter()
                                    .enumerate()
                                    .take(state_dim)
                                {
                                    *y_value += layer.ssm_alpha[j * state_dim + i] * state_value;
                                }
                            }
                        }

                        // Pad or truncate y_ssm to match the Mamba inner width.
                        if y_ssm.len() >= mamba_out.len() {
                            let out_len = mamba_out.len();
                            mamba_out.copy_from_slice(&y_ssm[..out_len]);
                        } else {
                            mamba_out[..y_ssm.len()].copy_from_slice(&y_ssm);
                        }
                    }

                    // Apply gate: y = y * silu(z_gate or gate)
                    let gate_to_use: Vec<f32> = if gate_len > 0 && gate_len == mamba_out.len() {
                        // Use attn_gate if available
                        let silu_gate = &mut ws.intermediate_a[..gate_len];
                        for val in silu_gate.iter_mut() {
                            *val = *val * (1.0_f32 / (1.0_f32 + (-*val).exp()));
                        }
                        silu_gate[..mamba_out.len()].to_vec()
                    } else if z_gate.len() == mamba_out.len() {
                        // Use second half of qkv projection
                        z_gate.clone()
                    } else {
                        vec![]
                    };

                    if gate_to_use.len() == mamba_out.len() {
                        for i in 0..mamba_out.len() {
                            mamba_out[i] *= gate_to_use[i];
                        }
                    }

                    // Final output projection
                    let mut residual = vec![0.0_f32; h];
                    if !layer.ssm_out.is_empty() {
                        let out_len = layer.ssm_out.output_dim(mamba_out.len());
                        if out_len > 0 {
                            let mut projected = vec![0.0_f32; out_len];
                            gemv_weight(
                                &layer.ssm_out,
                                out_len,
                                mamba_out.len(),
                                &mamba_out,
                                &mut projected,
                            )
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("ssm_out: {:?}", e))
                            })?;
                            let copy_len = h.min(projected.len());
                            residual[..copy_len].copy_from_slice(&projected[..copy_len]);
                        }
                    } else {
                        let copy_len = h.min(mamba_out.len());
                        residual[..copy_len].copy_from_slice(&mamba_out[..copy_len]);
                    }
                    residual
                };

                for i in 0..h {
                    ws.x[i] += mamba_out[i];
                }
            } else if !layer.mla_kv_a_mqa.is_empty() {
                let kv_layer_idx = self
                    .kv_layer_map
                    .get(layer_idx)
                    .copied()
                    .flatten()
                    .unwrap_or(layer_idx);
                ws.hidden_a[..h].fill(0.0_f32);
                {
                    let kv_cache = &mut self.kv_cache;
                    Self::deepseek_mla_layer(kv_cache, layer, cfg, kv_layer_idx, pos, ws)?;
                }
                for i in 0..h {
                    ws.x[i] += ws.hidden_a[i];
                }
            } else if !layer.attn_q.is_empty() {
                // ---- Standard attention ----
                #[cfg(feature = "cuda")]
                if gpu_native && gpu_attn && pos > 0 {
                    decode_graph_launched =
                        crate::cuda::gpu_decode_layer_graph_begin(layer_idx, pos, cfg.context_size)
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("cuda_graph_begin: {e}"))
                            })?;
                }

                if !decode_graph_launched {
                    // Look up the KV cache index for this attention layer.  Non-attention
                    // layers are skipped above, so this entry is always Some for this path.
                    let kv_layer_idx = self
                        .kv_layer_map
                        .get(layer_idx)
                        .copied()
                        .flatten()
                        .unwrap_or(layer_idx);
                    let attn_out = &mut ws.hidden_a[..h];
                    attn_out.fill(0.0_f32);
                    {
                        // Compute dynamic dimensions (shared by both CPU and GPU paths).
                        let q_len = layer.attn_q.output_dim(h);
                        let kv_len = if !layer.attn_k.is_empty() {
                            layer.attn_k.output_dim(h)
                        } else {
                            0
                        };
                        let attn_output_input_len = if !layer.attn_output.is_empty() {
                            layer.attn_output.output_dim(h)
                        } else {
                            0
                        };

                        let q_full = &mut ws.q_full[..q_len];
                        q_full.fill(0.0_f32);
                        let k_vec = &mut ws.k_vec[..kv_len];
                        k_vec.fill(0.0_f32);
                        let v_vec = &mut ws.v_vec[..kv_len];
                        v_vec.fill(0.0_f32);

                        // ---- RMS-norm + QKV projections ----
                        // GPU-native path: GPU keeps the hidden state; runs rms_norm and
                        // the three GEMVs entirely on-device, then DMA-copies only q, k, v
                        // (≪ the full hidden state) to CPU for rope + attention.
                        #[cfg(feature = "cuda")]
                        let used_gpu_qkv = if gpu_native && gpu_attn {
                            // OX_GPU_ATTN: the QKV projections run INSIDE the fused
                            // attention block (gpu_attn_block_fused_q4k, below), reading
                            // the device-resident hidden state directly. Skip the
                            // host-marshalling QKV download here; just mark the CPU QKV
                            // island as bypassed. The host q_full/k_vec/v_vec are left
                            // as-is (their VALUES are unused by the fused path — only the
                            // derived dims/geometry feed the fused call).
                            true
                        } else if gpu_native {
                            let wq = Self::q4k_or_q6k_bytes(&layer.attn_q).ok_or_else(|| {
                                ModelError::InferenceFailed("gpu_native: wq not Q4K/Q6K".into())
                            })?;
                            let wk = Self::q4k_or_q6k_bytes(&layer.attn_k).ok_or_else(|| {
                                ModelError::InferenceFailed("gpu_native: wk not Q4K/Q6K".into())
                            })?;
                            let wv = Self::q4k_or_q6k_bytes(&layer.attn_v).ok_or_else(|| {
                                ModelError::InferenceFailed("gpu_native: wv not Q4K/Q6K".into())
                            })?;
                            crate::cuda::gpu_attn_rms_and_qkv_q4k(
                                &layer.attn_norm,
                                cfg.rms_norm_eps,
                                wq,
                                q_len,
                                h,
                                wk,
                                kv_len,
                                wv,
                                q_full,
                                k_vec,
                                v_vec,
                            )
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("gpu_attn_qkv: {e}"))
                            })?;
                            true
                        } else {
                            false
                        };
                        #[cfg(not(feature = "cuda"))]
                        let used_gpu_qkv = false;

                        if !used_gpu_qkv {
                            let normed = &mut ws.hidden_b[..h];
                            normed.fill(0.0_f32);
                            rms_norm_f32(&ws.x[..h], &layer.attn_norm, cfg.rms_norm_eps, normed)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("rms_norm: {:?}", e))
                                })?;
                            // Run Q, K, V projections as ONE fused parallel region —
                            // they share the same normed input and write to
                            // non-overlapping buffers (q_full, k_vec, v_vec).
                            gemv_weight_fused(
                                vec![
                                    (&layer.attn_q, q_len, &mut *q_full),
                                    (
                                        &layer.attn_k,
                                        if layer.attn_k.is_empty() { 0 } else { kv_len },
                                        &mut *k_vec,
                                    ),
                                    (
                                        &layer.attn_v,
                                        if layer.attn_v.is_empty() { 0 } else { kv_len },
                                        &mut *v_vec,
                                    ),
                                ],
                                h,
                                normed,
                            )
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("attn_qkv: {:?}", e))
                            })?;
                        }
                        let glue_t0 =
                            crate::tensor::decode_profile_enabled().then(std::time::Instant::now);

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

                        // In MLA-style attention, attn_q output is split into Q and auxiliary projection.
                        let q_len_actual = if attn_output_input_len > 0 {
                            q_len.min(attn_output_input_len)
                        } else if q_len > h {
                            h
                        } else {
                            q_len
                        };
                        let q = &mut ws.q_full[..q_len_actual];

                        // Compute head dimensions based on ACTUAL Q length after splitting
                        let q_len_used = q.len();
                        let (q_head_dim, q_heads, kv_head_dim, kv_heads) =
                            attention_head_dims(cfg, layer, q_len_used, kv_len);

                        // Apply per-head Q/K norm if available
                        if !layer.attn_q_norm.is_empty() && q.len() == layer.attn_q_norm.len() {
                            let normed_q = &mut ws.flash_q[..q.len()];
                            rms_norm_f32(q, &layer.attn_q_norm, cfg.rms_norm_eps, normed_q)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("q_norm: {:?}", e))
                                })?;
                            q.copy_from_slice(normed_q);
                        } else if !layer.attn_q_norm.is_empty()
                            && q_head_dim == layer.attn_q_norm.len()
                        {
                            for head in 0..q_heads {
                                let start = head * q_head_dim;
                                let end = start + q_head_dim;
                                if end > q.len() {
                                    break;
                                }
                                let normed_head = &mut ws.head_scratch[..q_head_dim];
                                normed_head.fill(0.0_f32);
                                rms_norm_f32(
                                    &q[start..end],
                                    &layer.attn_q_norm,
                                    cfg.rms_norm_eps,
                                    normed_head,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("q_norm: {:?}", e))
                                })?;
                                q[start..end].copy_from_slice(normed_head);
                            }
                        }
                        if !layer.attn_k_norm.is_empty() && k_vec.len() == layer.attn_k_norm.len() {
                            let normed_k = &mut ws.attn_result[..k_vec.len()];
                            rms_norm_f32(k_vec, &layer.attn_k_norm, cfg.rms_norm_eps, normed_k)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("k_norm: {:?}", e))
                                })?;
                            k_vec.copy_from_slice(normed_k);
                        } else if !layer.attn_k_norm.is_empty()
                            && kv_head_dim == layer.attn_k_norm.len()
                        {
                            for head in 0..kv_heads {
                                let start = head * kv_head_dim;
                                let end = start + kv_head_dim;
                                if end > k_vec.len() {
                                    break;
                                }
                                let normed_head = &mut ws.head_scratch[..kv_head_dim];
                                normed_head.fill(0.0_f32);
                                rms_norm_f32(
                                    &k_vec[start..end],
                                    &layer.attn_k_norm,
                                    cfg.rms_norm_eps,
                                    normed_head,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("k_norm: {:?}", e))
                                })?;
                                k_vec[start..end].copy_from_slice(normed_head);
                            }
                        }

                        // Attention output buffer (filled by either the GPU or CPU path
                        // below). Bound here so it stays in scope for the wo step.
                        let attn_result = &mut ws.attn_result[..q_len_used];
                        attn_result.fill(0.0_f32);

                        // ---- On-device attention (OX_GPU_ATTN) ----
                        // When enabled and this is a standard GQA layer (the gpu_native
                        // gate already guarantees no biases, no per-head norm, and
                        // q_head_dim == kv_head_dim == head_dim), run RoPE + KV append +
                        // flash-attention entirely on the GPU. The device F16 KV cache
                        // is authoritative for the WHOLE run, so the host KvCache is NOT
                        // updated here (OX_GPU_ATTN is a whole-run, all-or-nothing mode).
                        //
                        // CRITICAL (gating-fallback): because the host KvCache is never
                        // populated under gpu_attn, the CPU attention island below would
                        // read an all-zero host cache and silently produce garbage. So a
                        // per-token failure of `gpu_attn_rope_append_flash` is NOT a safe
                        // fallback — it is a HARD error. The static geometry guards
                        // (q_head_dim == kv_head_dim, head-count products) are checked in
                        // the `if` condition below and cannot flip token-to-token; the
                        // only remaining reasons the call can Err are a transient CUDA
                        // fault or an unsupported runtime configuration (e.g. KV-context
                        // wraparound on very long generations). In both cases we abort the
                        // run instead of corrupting output. The `else { false }` branch
                        // (static geometry ineligible) is unreachable under the gpu_native
                        // gate (which guarantees q_head_dim == kv_head_dim for every layer
                        // in range), but is kept as a defensive uniform fallback: if it
                        // ever triggers, gpu_attn would have to be false for the whole
                        // range anyway, so the pure-CPU path (host cache warm) is used.
                        #[cfg(feature = "cuda")]
                        let used_gpu_attn = if gpu_attn
                            && q_head_dim == kv_head_dim
                            && q_head_dim * q_heads == q_len_used
                            && kv_head_dim * kv_heads == kv_len
                        {
                            let rope_dim = cfg.effective_rope_dim().min(q_head_dim);
                            // Fully fused, device-resident attention block: rms_norm +
                            // QKV GEMV + RoPE + KV append + flash decode + Wo GEMV +
                            // residual add, all inside ONE with_gpu() closure with NO
                            // host round-trip. Subsumes the former three calls
                            // (gpu_attn_rms_and_qkv_q4k + gpu_attn_rope_append_flash +
                            // gpu_wo_residual_q4k). The host glue between them (biases,
                            // per-head q/k norms) is guaranteed inert by the gpu_native
                            // gate, so fusing is sound.
                            let wq = Self::q4k_or_q6k_bytes(&layer.attn_q).ok_or_else(|| {
                                ModelError::InferenceFailed("gpu_native: wq not Q4K/Q6K".into())
                            })?;
                            let wk = Self::q4k_or_q6k_bytes(&layer.attn_k).ok_or_else(|| {
                                ModelError::InferenceFailed("gpu_native: wk not Q4K/Q6K".into())
                            })?;
                            let wv = Self::q4k_or_q6k_bytes(&layer.attn_v).ok_or_else(|| {
                                ModelError::InferenceFailed("gpu_native: wv not Q4K/Q6K".into())
                            })?;
                            let wo =
                                Self::q4k_or_q6k_bytes(&layer.attn_output).ok_or_else(|| {
                                    ModelError::InferenceFailed("gpu_native: wo not Q4K/Q6K".into())
                                })?;
                            match crate::cuda::gpu_attn_block_fused_q4k(
                                &layer.attn_norm,
                                cfg.rms_norm_eps,
                                wq,
                                q_len_used,
                                h,
                                wk,
                                kv_len,
                                wv,
                                q_head_dim,
                                q_heads,
                                kv_heads,
                                rope_dim,
                                layer_rope,
                                pos,
                                kv_layer_idx,
                                layer_window,
                                // Qwen3-style per-head QK-norm (empty for Llama/Mistral).
                                &layer.attn_q_norm,
                                &layer.attn_k_norm,
                                wo,
                                h,
                                attn_output_input_len,
                            ) {
                                Ok(()) => true,
                                // Hard error: the host cache is stale under gpu_attn, so
                                // there is no safe CPU fallback for this token.
                                Err(e) => {
                                    return Err(ModelError::InferenceFailed(format!(
                                        "gpu_attn (whole-run mode, no safe host-cache fallback): {e}"
                                    )));
                                }
                            }
                        } else if gpu_attn {
                            // gpu_attn is on but this layer's geometry is ineligible for
                            // the fused path (q_head_dim != kv_head_dim, or head-count
                            // products don't match q_len_used/kv_len). The QKV download was
                            // SKIPPED above (used_gpu_qkv=true under gpu_native && gpu_attn),
                            // so host q/k/v are all-zero and the CPU attention island below
                            // would silently produce garbage. Abort instead of corrupting
                            // output — matching the whole-run, no-host-cache-fallback policy.
                            return Err(ModelError::InferenceFailed(
                                "gpu_attn: layer geometry ineligible for fused attention but \
                             QKV download was skipped; no safe CPU fallback"
                                    .into(),
                            ));
                        } else {
                            false
                        };
                        #[cfg(not(feature = "cuda"))]
                        let used_gpu_attn = false;

                        if !used_gpu_attn {
                            // apply RoPE to Q (partial RoPE: first rope_dim elements per head)
                            let q_rope_len = cfg.effective_rope_dim().min(q_head_dim);
                            for head in 0..q_heads {
                                let off = head * q_head_dim;
                                if off + q_head_dim > q.len() {
                                    break;
                                }
                                let rotated = &mut ws.head_scratch[..q_rope_len];
                                rotated.fill(0.0_f32);
                                cfg.apply_rope_head(
                                    &q[off..off + q_rope_len],
                                    pos,
                                    q_rope_len,
                                    layer_rope,
                                    rotated,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("rope q: {:?}", e))
                                })?;
                                q[off..off + q_rope_len].copy_from_slice(rotated);
                            }

                            // apply RoPE to K (partial RoPE)
                            let k_rope_len = cfg.effective_rope_dim().min(kv_head_dim);
                            for head in 0..kv_heads {
                                let off = head * kv_head_dim;
                                if off + kv_head_dim > k_vec.len() {
                                    break;
                                }
                                let rotated = &mut ws.head_scratch[..k_rope_len];
                                rotated.fill(0.0_f32);
                                cfg.apply_rope_head(
                                    &k_vec[off..off + k_rope_len],
                                    pos,
                                    k_rope_len,
                                    layer_rope,
                                    rotated,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("rope k: {:?}", e))
                                })?;
                                k_vec[off..off + k_rope_len].copy_from_slice(rotated);
                            }

                            // --- Env-gated attention debug dump (OX_ATTN_DUMP), CPU path ---
                            // Snapshot the post-RoPE Q/K/V (and the post-RMSNorm input to
                            // QKV, recomputed here so it is available even when the QKV GEMV
                            // ran on the GPU) for the FIRST decode token's FIRST eligible
                            // attention layer. One-shot, fully behind the env flag; the
                            // attn_out vector is appended after the attention compute below.
                            // The GPU fused path emits the SAME labels/order/sizes so the
                            // two files diff cleanly.
                            let attn_dump_cpu: Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
                                if crate::attn_dump::should_dump() {
                                    let mut norm_in = vec![0.0_f32; h];
                                    rms_norm_f32(
                                        &ws.x[..h],
                                        &layer.attn_norm,
                                        cfg.rms_norm_eps,
                                        &mut norm_in,
                                    )
                                    .map_err(|e| {
                                        ModelError::InferenceFailed(format!(
                                            "attn_dump rms_norm: {:?}",
                                            e
                                        ))
                                    })?;
                                    Some((norm_in, q.to_vec(), k_vec.to_vec(), v_vec.to_vec()))
                                } else {
                                    None
                                };

                            let cache_token_size = self.kv_cache.config().token_size();
                            if cache_token_size == kv_len {
                                self.kv_cache.set(kv_layer_idx, pos, k_vec, v_vec).map_err(
                                    |e| ModelError::InferenceFailed(format!("kv set: {:?}", e)),
                                )?;
                            } else {
                                let key_row = &mut ws.kv_keys_copy[..cache_token_size];
                                let value_row = &mut ws.kv_values_copy[..cache_token_size];
                                key_row.fill(0.0_f32);
                                value_row.fill(0.0_f32);
                                key_row[..kv_len].copy_from_slice(k_vec);
                                value_row[..kv_len].copy_from_slice(v_vec);
                                self.kv_cache
                                    .set(kv_layer_idx, pos, key_row, value_row)
                                    .map_err(|e| {
                                        ModelError::InferenceFailed(format!("kv set: {:?}", e))
                                    })?;
                            }

                            let seq_len = pos + 1;

                            // compute attention using parallel flash attention decode over heads
                            // (`attn_result` was bound + zeroed above the GPU/CPU branch).

                            // For MLA-style where q_head_dim > kv_head_dim, truncate Q heads into scratch.
                            // Otherwise pass the Q projection directly and avoid a per-layer allocation.
                            let q_for_flash: &[f32] = if q_head_dim > kv_head_dim {
                                let q_truncated = &mut ws.flash_q[..q_heads * kv_head_dim];
                                for head in 0..q_heads {
                                    let src_start = head * q_head_dim;
                                    let dst_start = head * kv_head_dim;
                                    q_truncated[dst_start..dst_start + kv_head_dim]
                                        .copy_from_slice(&q[src_start..src_start + kv_head_dim]);
                                }
                                q_truncated
                            } else {
                                q
                            };

                            // Borrow the KV prefix in its storage dtype when the logical
                            // prefix is still contiguous in storage (F32 directly, F16 as
                            // half bits converted in-kernel); otherwise dequantize-copy
                            // into workspace buffers. Borrowing avoids materializing an
                            // f32 prefix copy per layer per token, and F16 also halves
                            // the attention DRAM reads vs an F32 cache.
                            let cache_token_size = self.kv_cache.config().token_size();
                            let f16_keys = if cache_token_size == kv_len {
                                self.kv_cache
                                    .f16_layer_key_prefix(kv_layer_idx, seq_len)
                                    .map_err(|e| {
                                        ModelError::InferenceFailed(format!(
                                            "kv borrow f16 keys: {:?}",
                                            e
                                        ))
                                    })?
                            } else {
                                None
                            };
                            let f16_values = if cache_token_size == kv_len {
                                self.kv_cache
                                    .f16_layer_value_prefix(kv_layer_idx, seq_len)
                                    .map_err(|e| {
                                        ModelError::InferenceFailed(format!(
                                            "kv borrow f16 values: {:?}",
                                            e
                                        ))
                                    })?
                            } else {
                                None
                            };
                            if let (Some(key16), Some(value16)) = (f16_keys, f16_values) {
                                // Sliding-window attention: a local layer attends only to
                                // the most recent `layer_window` positions (see the F32
                                // branch below for why slicing preserves the mask).
                                let (eff_seq_len, key16, value16) =
                                    if layer_window > 0 && seq_len > layer_window {
                                        let skip = (seq_len - layer_window) * kv_len;
                                        (layer_window, &key16[skip..], &value16[skip..])
                                    } else {
                                        (seq_len, key16, value16)
                                    };
                                if let Some(t0) = glue_t0 {
                                    crate::tensor::decode_profile_record(
                                        "pre_attn_glue",
                                        t0.elapsed().as_nanos() as u64,
                                    );
                                }
                                let attn_t0 = crate::tensor::decode_profile_enabled()
                                    .then(std::time::Instant::now);
                                flash_attention_decode_heads_f16(
                                    q_for_flash,
                                    key16,
                                    value16,
                                    eff_seq_len,
                                    kv_head_dim,
                                    kv_len,
                                    q_heads,
                                    kv_heads,
                                    attn_result,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!(
                                        "flash attention heads (f16): {:?}",
                                        e
                                    ))
                                })?;
                                if let Some(t0) = attn_t0 {
                                    crate::tensor::decode_profile_record(
                                        "attention",
                                        t0.elapsed().as_nanos() as u64,
                                    );
                                }
                            } else {
                                let borrowed_key_cache = if cache_token_size == kv_len {
                                    self.kv_cache
                                        .f32_layer_key_prefix(kv_layer_idx, seq_len)
                                        .map_err(|e| {
                                            ModelError::InferenceFailed(format!(
                                                "kv borrow keys: {:?}",
                                                e
                                            ))
                                        })?
                                } else {
                                    None
                                };
                                let borrowed_value_cache = if cache_token_size == kv_len {
                                    self.kv_cache
                                        .f32_layer_value_prefix(kv_layer_idx, seq_len)
                                        .map_err(|e| {
                                            ModelError::InferenceFailed(format!(
                                                "kv borrow values: {:?}",
                                                e
                                            ))
                                        })?
                                } else {
                                    None
                                };

                                let key_cache: &[f32];
                                let value_cache: &[f32];
                                if cache_token_size == kv_len
                                    && let (Some(keys), Some(values)) =
                                        (borrowed_key_cache, borrowed_value_cache)
                                {
                                    key_cache = keys;
                                    value_cache = values;
                                } else {
                                    let key_copy = &mut ws.kv_keys_copy[..seq_len * kv_len];
                                    let value_copy = &mut ws.kv_values_copy[..seq_len * kv_len];
                                    if cache_token_size == kv_len {
                                        self.kv_cache
                                            .copy_layer_keys(kv_layer_idx, seq_len, key_copy)
                                            .map_err(|e| {
                                                ModelError::InferenceFailed(format!(
                                                    "kv copy keys: {:?}",
                                                    e
                                                ))
                                            })?;
                                        self.kv_cache
                                            .copy_layer_values(kv_layer_idx, seq_len, value_copy)
                                            .map_err(|e| {
                                                ModelError::InferenceFailed(format!(
                                                    "kv copy values: {:?}",
                                                    e
                                                ))
                                            })?;
                                    } else {
                                        self.kv_cache
                                            .copy_layer_key_prefix_values(
                                                kv_layer_idx,
                                                seq_len,
                                                kv_len,
                                                key_copy,
                                            )
                                            .map_err(|e| {
                                                ModelError::InferenceFailed(format!(
                                                    "kv copy keys: {:?}",
                                                    e
                                                ))
                                            })?;
                                        self.kv_cache
                                            .copy_layer_value_prefix_values(
                                                kv_layer_idx,
                                                seq_len,
                                                kv_len,
                                                value_copy,
                                            )
                                            .map_err(|e| {
                                                ModelError::InferenceFailed(format!(
                                                    "kv copy values: {:?}",
                                                    e
                                                ))
                                            })?;
                                    }
                                    key_cache = key_copy;
                                    value_cache = value_copy;
                                }

                                // Sliding-window attention: a local layer attends only to the
                                // most recent `layer_window` positions. RoPE encodes absolute
                                // positions, so slicing off the oldest rows yields the
                                // windowed-causal mask with relative positions preserved.
                                let (eff_seq_len, key_cache, value_cache) =
                                    if layer_window > 0 && seq_len > layer_window {
                                        let skip = (seq_len - layer_window) * kv_len;
                                        (layer_window, &key_cache[skip..], &value_cache[skip..])
                                    } else {
                                        (seq_len, key_cache, value_cache)
                                    };
                                if let Some(t0) = glue_t0 {
                                    crate::tensor::decode_profile_record(
                                        "pre_attn_glue",
                                        t0.elapsed().as_nanos() as u64,
                                    );
                                }
                                let attn_t0 = crate::tensor::decode_profile_enabled()
                                    .then(std::time::Instant::now);
                                flash_attention_decode_heads_f32(
                                    q_for_flash,
                                    key_cache,
                                    value_cache,
                                    eff_seq_len,
                                    kv_head_dim,
                                    kv_len,
                                    q_heads,
                                    kv_heads,
                                    attn_result,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!(
                                        "flash attention heads: {:?}",
                                        e
                                    ))
                                })?;
                                if let Some(t0) = attn_t0 {
                                    crate::tensor::decode_profile_record(
                                        "attention",
                                        t0.elapsed().as_nanos() as u64,
                                    );
                                }
                            }
                            // --- Env-gated attention debug dump (OX_ATTN_DUMP), CPU path ---
                            // Emit the full labeled block now that attn_result is computed.
                            // `attn_dump_cpu` is Some only for the one-shot first eligible
                            // layer; write_block claims the one-shot token and no-ops if the
                            // GPU path already claimed it (the two are mutually exclusive per
                            // run — OX_GPU_ATTN selects one attention path).
                            if let Some((norm_in, q_rope, k_cur, v_cur)) = attn_dump_cpu {
                                crate::attn_dump::write_block(
                                    "cpu",
                                    pos,
                                    layer_idx,
                                    kv_layer_idx,
                                    &norm_in,
                                    &q_rope,
                                    &k_cur,
                                    &v_cur,
                                    attn_result,
                                );
                            }
                        } // end `if !used_gpu_attn` (CPU attention island)

                        // Reconcile attention result size with attn_output expected input
                        let attn_input = if attn_output_input_len > 0
                            && attn_result.len() != attn_output_input_len
                        {
                            if attn_result.len() >= attn_output_input_len {
                                &attn_result[..attn_output_input_len]
                            } else {
                                let padded = &mut ws.q_full[..attn_output_input_len];
                                padded.fill(0.0_f32);
                                padded[..attn_result.len()].copy_from_slice(attn_result);
                                &padded[..attn_output_input_len]
                            }
                        } else {
                            &attn_result[..]
                        };

                        if !layer.attn_output.is_empty() && attn_output_input_len > 0 {
                            // GPU-native: upload attn_result to GPU, run wo GEMV + residual
                            // entirely on device.  No CPU residual add needed.
                            #[cfg(feature = "cuda")]
                            let used_gpu_wo = if gpu_native && used_gpu_attn {
                                // Wo GEMV + residual add were already performed inside
                                // gpu_attn_block_fused_q4k (device-resident). Nothing to
                                // do here — just mark the CPU/GPU wo fallback as bypassed.
                                true
                            } else if gpu_native {
                                let wo = Self::q4k_or_q6k_bytes(&layer.attn_output).ok_or_else(
                                    || ModelError::InferenceFailed("gpu_native: wo not Q4K".into()),
                                )?;
                                crate::cuda::gpu_wo_residual_q4k(
                                    attn_input,
                                    wo,
                                    h,
                                    attn_output_input_len,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("gpu_wo_residual: {e}"))
                                })?;
                                true
                            } else {
                                false
                            };
                            #[cfg(not(feature = "cuda"))]
                            let used_gpu_wo = false;

                            if !used_gpu_wo {
                                gemv_weight(
                                    &layer.attn_output,
                                    h,
                                    attn_output_input_len,
                                    attn_input,
                                    attn_out,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("attn_output: {:?}", e))
                                })?;
                                if !layer.attn_output_bias.is_empty() {
                                    for (i, out) in attn_out.iter_mut().enumerate() {
                                        *out += layer.attn_output_bias
                                            [i % layer.attn_output_bias.len()];
                                    }
                                }
                            }
                        }
                    }

                    // Gemma sandwich norm: normalize the attention output before the
                    // residual add (post_attention_norm).  Not used in GPU-native path.
                    if !gpu_native && cfg.sandwich_norm && !layer.post_attention_norm.is_empty() {
                        let normed_attn = &mut ws.hidden_b[..h];
                        rms_norm_f32(
                            attn_out,
                            &layer.post_attention_norm,
                            cfg.rms_norm_eps,
                            normed_attn,
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("post_attn_norm: {:?}", e))
                        })?;
                        attn_out.copy_from_slice(normed_attn);
                    }

                    // CPU residual add (skipped when GPU-native handled it above).
                    if !gpu_native {
                        for i in 0..h {
                            ws.x[i] += attn_out[i];
                        }
                    }
                } // !decode_graph_launched
            }

            // ---- FFN (shared between Mamba and standard layers) ----
            let has_dense_ffn = !layer.ffn_gate.is_empty()
                && !layer.ffn_up.is_empty()
                && !layer.ffn_down.is_empty()
                && !ffn_norm_weight.is_empty();
            let has_moe = cfg.num_experts > 0
                && !layer.ffn_gate_exps.is_empty()
                && !layer.ffn_up_exps.is_empty()
                && !layer.ffn_down_exps.is_empty()
                && !layer.ffn_gate_inp.is_empty()
                && !ffn_norm_weight.is_empty();

            // GPU-native dense FFN: rms-norm + gate/up/silu/down + residual stay on
            // the GPU; hidden state is NOT downloaded between layers.
            #[cfg(feature = "cuda")]
            let gpu_ran_ffn = if decode_graph_launched {
                true
            } else if gpu_native && has_dense_ffn && !has_moe && !cfg.sandwich_norm {
                let gate = Self::q4k_or_q6k_bytes(&layer.ffn_gate).ok_or_else(|| {
                    ModelError::InferenceFailed("gpu_native: ffn_gate not Q4K/Q6K".into())
                })?;
                let up = Self::q4k_or_q6k_bytes(&layer.ffn_up).ok_or_else(|| {
                    ModelError::InferenceFailed("gpu_native: ffn_up not Q4K/Q6K".into())
                })?;
                let down = Self::q4k_or_q6k_bytes(&layer.ffn_down).ok_or_else(|| {
                    ModelError::InferenceFailed("gpu_native: ffn_down not Q4K/Q6K".into())
                })?;
                crate::cuda::gpu_ffn_q4k(
                    ffn_norm_weight,
                    cfg.rms_norm_eps,
                    gate,
                    cfg.intermediate_size,
                    h,
                    up,
                    cfg.intermediate_size,
                    down,
                    h,
                    cfg.intermediate_size,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("gpu_ffn: {e}")))?;
                true
            } else {
                false
            };
            #[cfg(not(feature = "cuda"))]
            let gpu_ran_ffn = false;

            #[cfg(feature = "cuda")]
            if gpu_native && gpu_attn && pos > 0 && !decode_graph_launched {
                crate::cuda::gpu_decode_layer_graph_end(layer_idx, pos, cfg.context_size)
                    .map_err(|e| ModelError::InferenceFailed(format!("cuda_graph_end: {e}")))?;
            }

            if (has_dense_ffn || has_moe) && !gpu_ran_ffn {
                let ffn_out = &mut ws.hidden_a[..h];
                ffn_out.fill(0.0_f32);
                {
                    let normed = &mut ws.hidden_b[..h];
                    normed.fill(0.0_f32);
                    rms_norm_f32(&ws.x[..h], ffn_norm_weight, cfg.rms_norm_eps, normed)
                        .map_err(|e| ModelError::InferenceFailed(format!("ffn_norm: {:?}", e)))?;

                    if has_moe {
                        let moe_i = if cfg.expert_intermediate_size > 0 {
                            cfg.expert_intermediate_size
                        } else {
                            cfg.intermediate_size
                        };
                        let n_sel = cfg.num_experts_per_tok.max(1).min(cfg.num_experts);
                        let gate_scratch = &mut ws.moe_gate_all[..n_sel * moe_i];
                        let up_scratch = &mut ws.moe_up_all[..n_sel * moe_i];
                        let expert_out = &mut ws.moe_down_all[..n_sel * h];
                        Self::moe_ffn_forward_single(
                            layer,
                            cfg,
                            normed,
                            ffn_out,
                            gate_scratch,
                            up_scratch,
                            expert_out,
                            &mut ws.moe_router_logits[..cfg.num_experts],
                            &mut ws.moe_expert_scores[..cfg.num_experts],
                        )?;
                        if !layer.ffn_gate_shexp.is_empty()
                            && !layer.ffn_up_shexp.is_empty()
                            && !layer.ffn_down_shexp.is_empty()
                        {
                            let shexp_i = if cfg.expert_intermediate_size > 0 {
                                cfg.expert_intermediate_size
                            } else {
                                cfg.intermediate_size
                            };
                            let gate = &mut ws.intermediate_a[..shexp_i];
                            let up = &mut ws.intermediate_b[..shexp_i];
                            let shexp_out = &mut ws.intermediate_c[..h];
                            gate.fill(0.0_f32);
                            up.fill(0.0_f32);
                            shexp_out.fill(0.0_f32);
                            gemv_weight(&layer.ffn_gate_shexp, shexp_i, h, normed, gate).map_err(
                                |e| ModelError::InferenceFailed(format!("shexp gate: {:?}", e)),
                            )?;
                            gemv_weight(&layer.ffn_up_shexp, shexp_i, h, normed, up).map_err(
                                |e| ModelError::InferenceFailed(format!("shexp up: {:?}", e)),
                            )?;
                            apply_swiglu_inplace_f32(gate, up);
                            gemv_weight(&layer.ffn_down_shexp, h, shexp_i, gate, shexp_out)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("shexp down: {:?}", e))
                                })?;
                            if !layer.ffn_gate_inp_shexp.is_empty() {
                                let gate_logit = &mut ws.moe_router_logits[..1];
                                gate_logit[0] = 0.0_f32;
                                gemv_weight(&layer.ffn_gate_inp_shexp, 1, h, normed, gate_logit)
                                    .map_err(|e| {
                                        ModelError::InferenceFailed(format!(
                                            "shexp router gate: {:?}",
                                            e
                                        ))
                                    })?;
                                let scale = 1.0_f32 / (1.0 + (-gate_logit[0]).exp());
                                for val in shexp_out.iter_mut() {
                                    *val *= scale;
                                }
                            }
                            for i in 0..h {
                                ffn_out[i] += shexp_out[i];
                            }
                        }
                    } else {
                        let gate = &mut ws.intermediate_a[..cfg.intermediate_size];
                        gate.fill(0.0_f32);
                        let up = &mut ws.intermediate_b[..cfg.intermediate_size];
                        up.fill(0.0_f32);
                        // Gate and up share the normed input; run both as ONE
                        // fused parallel region (two nested regions stole work
                        // from each other and halved streaming throughput).
                        gemv_weight_fused(
                            vec![
                                (&layer.ffn_gate, cfg.intermediate_size, &mut *gate),
                                (&layer.ffn_up, cfg.intermediate_size, &mut *up),
                            ],
                            h,
                            normed,
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("ffn_gate_up: {:?}", e))
                        })?;

                        // GeGLU for Gemma, otherwise SwiGLU (AVX2 fast path).
                        if cfg.gelu_ffn {
                            apply_geglu_inplace_f32(gate, up);
                        } else {
                            apply_swiglu_inplace_f32(gate, up);
                        }

                        gemv_weight(&layer.ffn_down, h, cfg.intermediate_size, gate, ffn_out)
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("ffn_down: {:?}", e))
                            })?;
                        if !layer.ffn_down_bias.is_empty() {
                            for (i, out) in ffn_out.iter_mut().enumerate() {
                                *out += layer.ffn_down_bias[i % layer.ffn_down_bias.len()];
                            }
                        }
                    }
                }

                // Gemma sandwich norm: normalize the FFN output before residual.
                if cfg.sandwich_norm && !layer.post_ffn_norm.is_empty() {
                    let normed_ffn = &mut ws.hidden_b[..h];
                    rms_norm_f32(ffn_out, &layer.post_ffn_norm, cfg.rms_norm_eps, normed_ffn)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("post_ffn_norm: {:?}", e))
                        })?;
                    ffn_out.copy_from_slice(normed_ffn);
                }

                for i in 0..h {
                    ws.x[i] += ffn_out[i];
                }
            }
            if self.eagle3_capture_layers.contains(&layer_idx) {
                if self.eagle3_layer_hiddens.len() <= layer_idx {
                    self.eagle3_layer_hiddens
                        .resize(self.config.layer_count, None);
                }
                self.eagle3_layer_hiddens[layer_idx] = Some(ws.x[..h].to_vec());
            }
            if trace_fwd_enabled() {
                let sum: f64 = ws.x[..h].iter().map(|v| *v as f64).sum();
                eprintln!("TRACE inf pos={pos} layer={layer_idx} sum={sum:.9e}");
            }
        }

        // gpu_native leaves hidden on device; final_head uses gpu_final_head_device_resident.
        Ok(())
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
                gemv_f32(&data[start..end], rows, cols, input, output)
                    .map_err(|e| format!("{:?}", e))
            }
            WeightStorage::Quantized(qtype, data) => {
                let (block_width, block_size) = weight_block_info(*qtype);
                let blocks_per_row = cols / block_width;
                let per_head = rows * blocks_per_row * block_size;
                let start = head * per_head;
                let end = start + per_head;
                gemv_quantized_f32(*qtype, &data[start..end], rows, cols, input, output)
                    .map_err(|e| format!("{:?}", e))
            }
            WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                let data = &mmap[*offset..*offset + *size];
                let (block_width, block_size) = weight_block_info(*qtype);
                let blocks_per_row = cols / block_width;
                let per_head = rows * blocks_per_row * block_size;
                let start = head * per_head;
                let end = start + per_head;
                gemv_quantized_f32(*qtype, &data[start..end], rows, cols, input, output)
                    .map_err(|e| format!("{:?}", e))
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
                    return Err(format!(
                        "v_b head {head} out of range (bytes {end} > {})",
                        data.len()
                    ));
                }
                let mut deq = vec![0.0_f32; per_head_elems];
                dequantize_scalar(*qtype, &data[start..end], &mut deq)
                    .map_err(|e| format!("dequant v_b: {:?}", e))?;
                w_host.copy_from_slice(&deq);
            }
            WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                let data = &mmap[*offset..*offset + *size];
                let per_head_bytes = data.len() / n_heads.max(1);
                let start = head * per_head_bytes;
                let end = (head + 1) * per_head_bytes;
                if end > data.len() {
                    return Err(format!(
                        "v_b head {head} out of range (bytes {end} > {})",
                        data.len()
                    ));
                }
                let mut deq = vec![0.0_f32; per_head_elems];
                dequantize_scalar(*qtype, &data[start..end], &mut deq)
                    .map_err(|e| format!("dequant v_b: {:?}", e))?;
                w_host.copy_from_slice(&deq);
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

    #[allow(clippy::too_many_lines)]
    fn deepseek_mla_layer(
        kv_cache: &mut KvCache,
        layer: &LayerWeights,
        cfg: &InferenceConfig,
        kv_layer_idx: usize,
        pos: usize,
        ws: &mut Workspace,
    ) -> Result<(), ModelError> {
        let h = cfg.hidden_size;
        let x = &ws.x[..h];
        let attn_out = &mut ws.hidden_a[..h];
        let n_heads = cfg.num_attention_heads;
        let kv_lora = layer.mla_kv_a_norm.len();
        let q_lora_dim = layer.mla_q_a.output_dim(h);
        let q_len = layer.mla_q_b.output_dim(q_lora_dim);
        let k_head_dim = cfg.kv_head_dim();
        let kv_pe_dim = layer.mla_kv_a_mqa.output_dim(h).saturating_sub(kv_lora);
        let k_nope_dim = layer.mla_k_b.output_dim(kv_lora) / n_heads.max(1);
        let v_head_dim = layer.mla_v_b.output_dim(kv_lora) / n_heads.max(1);
        let q_pe_dim = k_head_dim.saturating_sub(k_nope_dim);

        let normed = &mut ws.hidden_b[..h];
        normed.fill(0.0_f32);
        rms_norm_f32(x, &layer.attn_norm, cfg.rms_norm_eps, normed)
            .map_err(|e| ModelError::InferenceFailed(format!("mla attn_norm: {:?}", e)))?;

        let q_lora = layer.mla_q_a.output_dim(h);
        let c_q = &mut ws.intermediate_a[..q_lora];
        c_q.fill(0.0_f32);
        gemv_weight(&layer.mla_q_a, q_lora, h, normed, c_q)
            .map_err(|e| ModelError::InferenceFailed(format!("mla q_a: {:?}", e)))?;
        if !layer.mla_q_a_norm.is_empty() {
            let normed_q = &mut ws.intermediate_b[..q_lora];
            normed_q.copy_from_slice(c_q);
            rms_norm_f32(normed_q, &layer.mla_q_a_norm, cfg.rms_norm_eps, c_q)
                .map_err(|e| ModelError::InferenceFailed(format!("mla q_a_norm: {:?}", e)))?;
        }

        let q = &mut ws.q_full[..q_len];
        q.fill(0.0_f32);
        gemv_weight(&layer.mla_q_b, q_len, q_lora, c_q, q)
            .map_err(|e| ModelError::InferenceFailed(format!("mla q_b: {:?}", e)))?;

        let kv_out = layer.mla_kv_a_mqa.output_dim(h);
        let kv_pe = &mut ws.intermediate_c[..kv_out];
        kv_pe.fill(0.0_f32);
        gemv_weight(&layer.mla_kv_a_mqa, kv_out, h, normed, kv_pe)
            .map_err(|e| ModelError::InferenceFailed(format!("mla kv_a: {:?}", e)))?;

        let c_kv = &mut ws.mamba_scratch[..kv_lora];
        c_kv.copy_from_slice(&kv_pe[..kv_lora]);
        if !layer.mla_kv_a_norm.is_empty() {
            let c_kv_tmp = &mut ws.intermediate_b[..kv_lora];
            c_kv_tmp.copy_from_slice(c_kv);
            rms_norm_f32(c_kv_tmp, &layer.mla_kv_a_norm, cfg.rms_norm_eps, c_kv)
                .map_err(|e| ModelError::InferenceFailed(format!("mla kv_norm: {:?}", e)))?;
        }

        let k_pe_raw = &kv_pe[kv_lora..kv_lora + kv_pe_dim];
        let k_pe_rope = &mut ws.flash_q[..kv_pe_dim];
        cfg.apply_rope_head(k_pe_raw, pos, kv_pe_dim, cfg.rope_theta, k_pe_rope)
            .map_err(|e| ModelError::InferenceFailed(format!("mla k_pe rope: {:?}", e)))?;

        let total_k = n_heads * k_head_dim;
        let total_v = n_heads * v_head_dim;
        let k_store = &mut ws.k_vec[..total_k];
        let v_store = &mut ws.v_vec[..total_v];
        k_store.fill(0.0_f32);
        v_store.fill(0.0_f32);

        for head in 0..n_heads {
            let k_off = head * k_head_dim;
            Self::gemv_weight_head(
                &layer.mla_k_b,
                k_nope_dim,
                kv_lora,
                head,
                n_heads,
                c_kv,
                &mut k_store[k_off..k_off + k_nope_dim],
            )
            .map_err(|e| ModelError::InferenceFailed(format!("mla k_b h{head}: {e}")))?;
            let rope_off = k_off + k_nope_dim;
            let copy = q_pe_dim
                .min(kv_pe_dim)
                .min(k_head_dim.saturating_sub(k_nope_dim));
            k_store[rope_off..rope_off + copy].copy_from_slice(&k_pe_rope[..copy]);

            let v_off = head * v_head_dim;
            Self::mla_v_b_head(
                &layer.mla_v_b,
                kv_lora,
                v_head_dim,
                head,
                n_heads,
                c_kv,
                &mut v_store[v_off..v_off + v_head_dim],
            )
            .map_err(|e| ModelError::InferenceFailed(format!("mla v_b h{head}: {e}")))?;

            let q_off = head * k_head_dim;
            if q_off + k_head_dim <= q.len() && q_pe_dim > 0 {
                let q_pe = &mut q[q_off + k_nope_dim..q_off + k_head_dim];
                let rotated = &mut ws.head_scratch[..q_pe_dim];
                rotated.fill(0.0_f32);
                cfg.apply_rope_head(q_pe, pos, q_pe_dim, cfg.rope_theta, rotated)
                    .map_err(|e| ModelError::InferenceFailed(format!("mla q_pe: {:?}", e)))?;
                q_pe.copy_from_slice(&rotated[..q_pe.len()]);
            }
        }

        let mut v_padded = vec![0.0_f32; total_k];
        for head in 0..n_heads {
            let v_off = head * v_head_dim;
            let k_off = head * k_head_dim;
            v_padded[k_off..k_off + v_head_dim]
                .copy_from_slice(&v_store[v_off..v_off + v_head_dim]);
        }
        kv_cache
            .set(kv_layer_idx, pos, k_store, &v_padded)
            .map_err(|e| ModelError::InferenceFailed(format!("mla kv set: {:?}", e)))?;

        let seq_len = pos + 1;
        let attn_result = &mut ws.attn_result[..total_k];
        attn_result.fill(0.0_f32);
        let scale = 1.0_f32 / (k_head_dim as f32).sqrt();

        for head in 0..n_heads {
            let q_off = head * k_head_dim;
            let k_off = head * k_head_dim;
            let _v_off = head * v_head_dim;
            let q_h = &q[q_off..q_off + k_head_dim];
            let mut scores = vec![0.0_f32; seq_len];
            for t in 0..seq_len {
                let mut k_t = vec![0.0_f32; total_k];
                kv_cache
                    .get_key(kv_layer_idx, t, &mut k_t)
                    .map_err(|e| ModelError::InferenceFailed(format!("mla get_k: {:?}", e)))?;
                let mut dot = 0.0_f32;
                for i in 0..k_head_dim {
                    dot += q_h[i] * k_t[q_off + i];
                }
                scores[t] = dot * scale;
            }
            let max_s = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0_f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in &mut scores {
                    *s /= sum;
                }
            }
            for i in 0..v_head_dim {
                let mut acc = 0.0_f32;
                for t in 0..seq_len {
                    let mut v_t = vec![0.0_f32; total_k];
                    kv_cache
                        .get_value(kv_layer_idx, t, &mut v_t)
                        .map_err(|e| ModelError::InferenceFailed(format!("mla get_v: {:?}", e)))?;
                    acc += scores[t] * v_t[k_off + i];
                }
                attn_result[q_off + i] = acc;
            }
        }

        let attn_in_len = layer.attn_output.output_dim(h);
        let attn_input = if attn_in_len > 0 && attn_result.len() >= attn_in_len {
            &attn_result[..attn_in_len]
        } else {
            &attn_result[..total_k.min(attn_result.len())]
        };
        gemv_weight(
            &layer.attn_output,
            h,
            attn_input.len(),
            attn_input,
            attn_out,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mla attn_out: {:?}", e)))?;
        Ok(())
    }
}
