use super::*;

impl LayerWiseModel {
    pub(super) fn forward_single(
        &mut self,
        token: Token,
        pos: usize,
    ) -> Result<Logits, ModelError> {
        self.trace_state("fwd1-entry", pos);
        let cfg = self.config.clone();
        let h = cfg.hidden_size;

        let mut x = vec![0.0_f32; h];
        let token_idx = (token as usize).min(cfg.vocab_size.saturating_sub(1));
        if std::env::var("OXIDIZE_DEBUG_LAYERS").is_ok() && pos == 0 {
            eprintln!("token_idx={token_idx}");
        }
        match &self.tok_embeddings {
            WeightStorage::F32(data) => {
                x.copy_from_slice(&data[token_idx * h..(token_idx + 1) * h]);
            }
            WeightStorage::Quantized(qtype, data) => {
                lookup_quantized_embedding(h, *qtype, data, token_idx, &mut x);
            }
            WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                let data = &mmap[*offset..*offset + *size];
                lookup_quantized_embedding(h, *qtype, data, token_idx, &mut x);
            }
        }

        debug_hidden("embed", pos, &x);
        scale_hidden(&mut x, cfg.embedding_scale);

        for layer_idx in 0..cfg.layer_count {
            self.ensure_layer_loaded(layer_idx)
                .map_err(|e| ModelError::InferenceFailed(format!("layer load: {}", e)))?;
            // SAFETY: layer weights live in a fixed cache slot; extending the
            // ref via raw pointer lets us call `&mut self` methods in the same scope.
            let layer = unsafe {
                let layer_ptr = self.layer_ref(layer_idx) as *const LayerWeights;
                &*layer_ptr
            };
            let parallel = cfg.architecture.uses_parallel_attn_ffn();
            let pre_residual = if parallel { x.clone() } else { Vec::new() };
            let ffn_norm_weight = select_ffn_norm(&cfg, layer);
            let is_shortconv = !weight_is_empty(&layer.shortconv_in_proj);
            let is_mamba = !weight_is_empty(&layer.attn_qkv) && layer.attn_q.is_empty();
            let is_mla = !weight_is_empty(&layer.mla_kv_a_mqa);

            if is_shortconv {
                let mixed = self.run_shortconv_layer(layer_idx, layer, &x, &cfg)?;
                for (xi, out) in x.iter_mut().zip(mixed.iter()).take(h) {
                    *xi += out;
                }
            } else if is_mamba {
                let mamba_out = self.run_mamba_layer(layer_idx, layer, &x, &cfg)?;
                for (xi, out) in x.iter_mut().zip(mamba_out.iter()).take(h) {
                    *xi += out;
                }
                debug_hidden(&format!("layer {layer_idx} after gdn"), pos, &x);
            } else if is_mla {
                let mut attn_out = self.run_mla_layer(layer_idx, layer, &x, pos, &cfg)?;
                apply_sandwich_norm(&cfg, &layer.post_attention_norm, &mut attn_out)?;
                for (xi, out) in x.iter_mut().zip(attn_out.iter()).take(h) {
                    *xi += out;
                }
                debug_hidden(&format!("layer {layer_idx} after mla"), pos, &x);
            } else if !weight_is_empty(&layer.attn_q) {
                let mut attn_out = self.run_attention_layer(layer_idx, layer, &x, pos, &cfg)?;
                apply_sandwich_norm(&cfg, &layer.post_attention_norm, &mut attn_out)?;
                for (xi, out) in x.iter_mut().zip(attn_out.iter()).take(h) {
                    *xi += out;
                }
                debug_hidden(&format!("layer {layer_idx} after attn"), pos, &x);
            }

            let has_dense_ffn = !weight_is_empty(&layer.ffn_gate)
                && !weight_is_empty(&layer.ffn_up)
                && !weight_is_empty(&layer.ffn_down)
                && !ffn_norm_weight.is_empty();
            let has_moe = cfg.num_experts > 0
                && !weight_is_empty(&layer.ffn_gate_exps)
                && !weight_is_empty(&layer.ffn_up_exps)
                && !weight_is_empty(&layer.ffn_down_exps)
                && !weight_is_empty(&layer.ffn_gate_inp)
                && !ffn_norm_weight.is_empty();
            if has_dense_ffn || has_moe {
                let ffn_in: &[f32] = if parallel { &pre_residual } else { &x };
                let mut ffn_out = vec![0.0_f32; h];
                {
                    let mut normed = vec![0.0_f32; h];
                    rms_norm_model(ffn_in, ffn_norm_weight, cfg.rms_norm_eps, &mut normed, &cfg)?;
                    if has_moe {
                        let moe_i = if cfg.expert_intermediate_size > 0 {
                            cfg.expert_intermediate_size
                        } else {
                            cfg.intermediate_size
                        };
                        let n_sel = cfg.num_experts_per_tok.max(1).min(cfg.num_experts);
                        let mut gate_scratch = vec![0.0_f32; n_sel * moe_i];
                        let mut up_scratch = vec![0.0_f32; n_sel * moe_i];
                        let mut expert_out = vec![0.0_f32; n_sel * h];
                        let mut router_logits = vec![0.0_f32; cfg.num_experts];
                        let mut expert_scores = vec![(0usize, 0.0_f32); cfg.num_experts];
                        let moe_weights = MoeFfnWeights {
                            gate_inp: &layer.ffn_gate_inp,
                            gate_exps: &layer.ffn_gate_exps,
                            up_exps: &layer.ffn_up_exps,
                            down_exps: &layer.ffn_down_exps,
                            exp_probs_b: &layer.ffn_exp_probs_b,
                        };
                        moe_ffn_forward_weights(
                            &moe_weights,
                            &cfg,
                            &normed,
                            &mut ffn_out,
                            &mut gate_scratch,
                            &mut up_scratch,
                            &mut expert_out,
                            &mut router_logits,
                            &mut expert_scores,
                        )?;
                        if !weight_is_empty(&layer.ffn_gate_shexp)
                            && !weight_is_empty(&layer.ffn_up_shexp)
                            && !weight_is_empty(&layer.ffn_down_shexp)
                        {
                            let shexp_i = moe_i;
                            let mut gate = vec![0.0_f32; shexp_i];
                            let mut up = vec![0.0_f32; shexp_i];
                            let mut shexp_out = vec![0.0_f32; h];
                            gemv_weight(&layer.ffn_gate_shexp, shexp_i, h, &normed, &mut gate)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("shexp gate: {:?}", e))
                                })?;
                            gemv_weight(&layer.ffn_up_shexp, shexp_i, h, &normed, &mut up)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("shexp up: {:?}", e))
                                })?;
                            let mut swiglu = vec![0.0_f32; shexp_i];
                            activate_ffn(&gate, &up, cfg.gelu_ffn, &mut swiglu)?;
                            gemv_weight(&layer.ffn_down_shexp, h, shexp_i, &swiglu, &mut shexp_out)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("shexp down: {:?}", e))
                                })?;
                            if !weight_is_empty(&layer.ffn_gate_inp_shexp) {
                                let mut gate_logit = vec![0.0_f32; 1];
                                gemv_weight(
                                    &layer.ffn_gate_inp_shexp,
                                    1,
                                    h,
                                    &normed,
                                    &mut gate_logit,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!(
                                        "shexp router gate: {:?}",
                                        e
                                    ))
                                })?;
                                let scale = sigmoid(gate_logit[0]);
                                for val in shexp_out.iter_mut() {
                                    *val *= scale;
                                }
                            }
                            for (out, val) in ffn_out.iter_mut().zip(shexp_out.iter()) {
                                *out += val;
                            }
                        }
                    } else {
                        let mut gate = vec![0.0_f32; cfg.intermediate_size];
                        let mut up = vec![0.0_f32; cfg.intermediate_size];
                        gemv_weight(
                            &layer.ffn_gate,
                            cfg.intermediate_size,
                            h,
                            &normed,
                            &mut gate,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("ffn_gate: {:?}", e)))?;
                        gemv_weight(&layer.ffn_up, cfg.intermediate_size, h, &normed, &mut up)
                            .map_err(|e| ModelError::InferenceFailed(format!("ffn_up: {:?}", e)))?;
                        let mut swiglu = vec![0.0_f32; cfg.intermediate_size];
                        activate_ffn(&gate, &up, cfg.gelu_ffn, &mut swiglu)?;
                        gemv_weight(
                            &layer.ffn_down,
                            h,
                            cfg.intermediate_size,
                            &swiglu,
                            &mut ffn_out,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("ffn_down: {:?}", e)))?;
                        if !layer.ffn_down_bias.is_empty() {
                            for (i, out) in ffn_out.iter_mut().enumerate() {
                                *out += layer.ffn_down_bias[i % layer.ffn_down_bias.len()];
                            }
                        }
                    }
                }
                apply_sandwich_norm(&cfg, &layer.post_ffn_norm, &mut ffn_out)?;
                for (xi, out) in x.iter_mut().zip(ffn_out.iter()).take(h) {
                    *xi += out;
                }
            }

            debug_hidden(&format!("layer {layer_idx}"), pos, &x);
            trace_fwd("single", pos, layer_idx, &x);
        }

        let mut normed = vec![0.0_f32; h];
        rms_norm_model(&x, &self.norm_weight, cfg.rms_norm_eps, &mut normed, &cfg)?;
        let mut logits = vec![0.0_f32; cfg.vocab_size];
        gemv_weight(&self.output_weight, cfg.vocab_size, h, &normed, &mut logits)
            .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))?;

        if std::env::var("OXIDIZE_DEBUG_LOGITS").is_ok() {
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!(
                "OXIDIZE_DEBUG pos={pos} top5: {:?}",
                indexed
                    .iter()
                    .take(5)
                    .map(|(id, val)| (id, val))
                    .collect::<Vec<_>>()
            );
        }

        self.ssm_pos = pos + 1;
        Ok(logits)
    }

    /// Process a window of consecutive tokens layer-major with batched weight
    /// passes (TileRT/MTP-style): every dense projection reads its weights once
    /// per window instead of once per token. Recurrent pieces (conv ring,
    /// delta-rule state, KV attention) run sequentially per token inside each
    /// layer, so results match the token-by-token path.
    ///
    /// Returns logits for every token when `want_all_logits` (speculative
    /// verification), or just the last token's logits otherwise (prefill —
    /// skipping the ~1B-param LM head for all non-final tokens).
    pub(super) fn forward_window(
        &mut self,
        tokens: &[Token],
        start_pos: usize,
        want_all_logits: bool,
    ) -> Result<Vec<Logits>, ModelError> {
        let kk = tokens.len();
        if kk == 0 {
            return Err(ModelError::EmptyInput);
        }
        if kk == 1 {
            let logits = self.forward_single(tokens[0], start_pos)?;
            return Ok(vec![logits]);
        }
        let xs = self.forward_window_states(tokens, start_pos)?;
        let cfg = self.config.clone();
        let h = cfg.hidden_size;

        // Final norm + LM head, batched over the tokens that need logits.
        let needed: Vec<usize> = if want_all_logits {
            (0..kk).collect()
        } else {
            vec![kk - 1]
        };
        let nb = needed.len();
        let mut normed_all = vec![0.0_f32; nb * h];
        for (j, &t) in needed.iter().enumerate() {
            let mut normed = vec![0.0_f32; h];
            rms_norm_model(
                &xs[t * h..(t + 1) * h],
                &self.norm_weight,
                cfg.rms_norm_eps,
                &mut normed,
                &cfg,
            )?;
            normed_all[j * h..(j + 1) * h].copy_from_slice(&normed);
        }
        let mut logits_all = vec![0.0_f32; nb * cfg.vocab_size];
        self.lm_head_logits_batch(&normed_all, nb, &mut logits_all)?;
        Ok(needed
            .iter()
            .enumerate()
            .map(|(j, _)| logits_all[j * cfg.vocab_size..(j + 1) * cfg.vocab_size].to_vec())
            .collect())
    }

    /// Batched final-normed hidden states for a window of tokens. This is the
    /// training entry point: it advances KV/SSM state exactly like
    /// `forward_window` but returns the post-final-norm hidden state for every
    /// position (`tokens.len() * hidden_size`, row-major by position) instead
    /// of computing LM-head logits.
    pub fn forward_normed_hidden(
        &mut self,
        tokens: &[Token],
        start_pos: usize,
    ) -> Result<Vec<f32>, ModelError> {
        let kk = tokens.len();
        if kk == 0 {
            return Err(ModelError::EmptyInput);
        }
        let xs = self.forward_window_states(tokens, start_pos)?;
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let mut normed_all = vec![0.0_f32; kk * h];
        for t in 0..kk {
            rms_norm_model(
                &xs[t * h..(t + 1) * h],
                &self.norm_weight,
                cfg.rms_norm_eps,
                &mut normed_all[t * h..(t + 1) * h],
                &cfg,
            )?;
        }
        Ok(normed_all)
    }

    /// LM-head logits for `count` rows of final-normed hidden states
    /// (`normed_all` is `count * hidden_size`, `logits_out` is
    /// `count * vocab_size`). Uses the batched GEMM weight path.
    pub fn lm_head_logits_batch(
        &self,
        normed_all: &[f32],
        count: usize,
        logits_out: &mut [f32],
    ) -> Result<(), ModelError> {
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        if normed_all.len() != count * h || logits_out.len() != count * vocab {
            return Err(ModelError::InferenceFailed(format!(
                "lm_head_logits_batch: normed={} logits={} expected {}x{h} and {}x{vocab}",
                normed_all.len(),
                logits_out.len(),
                count,
                count
            )));
        }
        gemm_weight(&self.output_weight, vocab, h, normed_all, logits_out, count)
            .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))
    }

    /// Run the transformer stack over a window of tokens, returning the
    /// pre-final-norm hidden state for every position (kk * hidden_size).
    /// Advances KV cache and SSM state to `start_pos + tokens.len()`.
    pub(super) fn forward_window_states(
        &mut self,
        tokens: &[Token],
        start_pos: usize,
    ) -> Result<Vec<f32>, ModelError> {
        let kk = tokens.len();
        if kk == 0 {
            return Err(ModelError::EmptyInput);
        }
        let cfg = self.config.clone();
        let h = cfg.hidden_size;

        let mut xs = vec![0.0_f32; kk * h];
        for (t, &token) in tokens.iter().enumerate() {
            let token_idx = (token as usize).min(cfg.vocab_size.saturating_sub(1));
            let x_t = &mut xs[t * h..(t + 1) * h];
            match &self.tok_embeddings {
                WeightStorage::F32(data) => {
                    x_t.copy_from_slice(&data[token_idx * h..(token_idx + 1) * h]);
                }
                WeightStorage::Quantized(qtype, data) => {
                    lookup_quantized_embedding(h, *qtype, data, token_idx, x_t);
                }
                WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                    let data = &mmap[*offset..*offset + *size];
                    lookup_quantized_embedding(h, *qtype, data, token_idx, x_t);
                }
            }
        }

        for t in 0..kk {
            trace_fwd("embd", start_pos + t, usize::MAX, &xs[t * h..(t + 1) * h]);
        }
        scale_hidden(&mut xs, cfg.embedding_scale);
        for layer_idx in 0..cfg.layer_count {
            self.ensure_layer_loaded(layer_idx)
                .map_err(|e| ModelError::InferenceFailed(format!("layer load: {}", e)))?;
            // SAFETY: same lifetime extension as forward_single (see above).
            let layer = unsafe {
                let layer_ptr = self.layer_ref(layer_idx) as *const LayerWeights;
                &*layer_ptr
            };
            let parallel = cfg.architecture.uses_parallel_attn_ffn();
            let pre_residual = if parallel { xs.clone() } else { Vec::new() };
            let ffn_norm_weight = select_ffn_norm(&cfg, layer);
            let is_shortconv = !weight_is_empty(&layer.shortconv_in_proj);
            let is_mamba = !weight_is_empty(&layer.attn_qkv) && layer.attn_q.is_empty();
            let is_mla = !weight_is_empty(&layer.mla_kv_a_mqa);

            if is_shortconv {
                for t in 0..kk {
                    let x_t = xs[t * h..(t + 1) * h].to_vec();
                    let mixed = self.run_shortconv_layer(layer_idx, layer, &x_t, &cfg)?;
                    for (xi, out) in xs[t * h..(t + 1) * h].iter_mut().zip(mixed.iter()) {
                        *xi += out;
                    }
                }
            } else if is_mamba {
                let residual = self.run_mamba_layer_batch(layer_idx, layer, &xs, kk, &cfg)?;
                for (xi, out) in xs.iter_mut().zip(residual.iter()) {
                    *xi += out;
                }
            } else if is_mla {
                for t in 0..kk {
                    let x_t = xs[t * h..(t + 1) * h].to_vec();
                    let mut attn_out =
                        self.run_mla_layer(layer_idx, layer, &x_t, start_pos + t, &cfg)?;
                    apply_sandwich_norm(&cfg, &layer.post_attention_norm, &mut attn_out)?;
                    for (xi, out) in xs[t * h..(t + 1) * h].iter_mut().zip(attn_out.iter()) {
                        *xi += out;
                    }
                }
            } else if !weight_is_empty(&layer.attn_q) {
                // Full-attention layers stay per token: KV append order matters.
                for t in 0..kk {
                    let x_t = xs[t * h..(t + 1) * h].to_vec();
                    let mut attn_out =
                        self.run_attention_layer(layer_idx, layer, &x_t, start_pos + t, &cfg)?;
                    apply_sandwich_norm(&cfg, &layer.post_attention_norm, &mut attn_out)?;
                    for (xi, out) in xs[t * h..(t + 1) * h].iter_mut().zip(attn_out.iter()) {
                        *xi += out;
                    }
                }
            }

            let has_dense_ffn = !weight_is_empty(&layer.ffn_gate)
                && !weight_is_empty(&layer.ffn_up)
                && !weight_is_empty(&layer.ffn_down)
                && !ffn_norm_weight.is_empty();
            let has_moe = cfg.num_experts > 0
                && !weight_is_empty(&layer.ffn_gate_exps)
                && !weight_is_empty(&layer.ffn_up_exps)
                && !weight_is_empty(&layer.ffn_down_exps)
                && !weight_is_empty(&layer.ffn_gate_inp)
                && !ffn_norm_weight.is_empty();
            if !(has_dense_ffn || has_moe) {
                continue;
            }

            let mut normed_all = vec![0.0_f32; kk * h];
            {
                let ffn_src = if parallel {
                    pre_residual.as_slice()
                } else {
                    xs.as_slice()
                };
                for t in 0..kk {
                    let mut normed = vec![0.0_f32; h];
                    rms_norm_model(
                        &ffn_src[t * h..(t + 1) * h],
                        ffn_norm_weight,
                        cfg.rms_norm_eps,
                        &mut normed,
                        &cfg,
                    )?;
                    normed_all[t * h..(t + 1) * h].copy_from_slice(&normed);
                }
            }
            let mut ffn_all = vec![0.0_f32; kk * h];

            if has_moe {
                let moe_i = if cfg.expert_intermediate_size > 0 {
                    cfg.expert_intermediate_size
                } else {
                    cfg.intermediate_size
                };
                let n_sel = cfg.num_experts_per_tok.max(1).min(cfg.num_experts);
                let mut gate_scratch = vec![0.0_f32; n_sel * moe_i];
                let mut up_scratch = vec![0.0_f32; n_sel * moe_i];
                let mut expert_out = vec![0.0_f32; n_sel * h];
                let mut router_logits = vec![0.0_f32; cfg.num_experts];
                let mut expert_scores = vec![(0usize, 0.0_f32); cfg.num_experts];
                let moe_weights = MoeFfnWeights {
                    gate_inp: &layer.ffn_gate_inp,
                    gate_exps: &layer.ffn_gate_exps,
                    up_exps: &layer.ffn_up_exps,
                    down_exps: &layer.ffn_down_exps,
                    exp_probs_b: &layer.ffn_exp_probs_b,
                };
                for t in 0..kk {
                    moe_ffn_forward_weights(
                        &moe_weights,
                        &cfg,
                        &normed_all[t * h..(t + 1) * h],
                        &mut ffn_all[t * h..(t + 1) * h],
                        &mut gate_scratch,
                        &mut up_scratch,
                        &mut expert_out,
                        &mut router_logits,
                        &mut expert_scores,
                    )?;
                }
                if !weight_is_empty(&layer.ffn_gate_shexp)
                    && !weight_is_empty(&layer.ffn_up_shexp)
                    && !weight_is_empty(&layer.ffn_down_shexp)
                {
                    let shexp_i = moe_i;
                    let mut gate_all = vec![0.0_f32; kk * shexp_i];
                    let mut up_all = vec![0.0_f32; kk * shexp_i];
                    gemm_weight(
                        &layer.ffn_gate_shexp,
                        shexp_i,
                        h,
                        &normed_all,
                        &mut gate_all,
                        kk,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("shexp gate: {:?}", e)))?;
                    gemm_weight(
                        &layer.ffn_up_shexp,
                        shexp_i,
                        h,
                        &normed_all,
                        &mut up_all,
                        kk,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("shexp up: {:?}", e)))?;
                    let mut swiglu_all = vec![0.0_f32; kk * shexp_i];
                    for t in 0..kk {
                        let mut swiglu = vec![0.0_f32; shexp_i];
                        activate_ffn(
                            &gate_all[t * shexp_i..(t + 1) * shexp_i],
                            &up_all[t * shexp_i..(t + 1) * shexp_i],
                            cfg.gelu_ffn,
                            &mut swiglu,
                        )?;
                        swiglu_all[t * shexp_i..(t + 1) * shexp_i].copy_from_slice(&swiglu);
                    }
                    let mut shexp_out_all = vec![0.0_f32; kk * h];
                    gemm_weight(
                        &layer.ffn_down_shexp,
                        h,
                        shexp_i,
                        &swiglu_all,
                        &mut shexp_out_all,
                        kk,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("shexp down: {:?}", e)))?;
                    if !weight_is_empty(&layer.ffn_gate_inp_shexp) {
                        let mut gate_logit_all = vec![0.0_f32; kk];
                        gemm_weight(
                            &layer.ffn_gate_inp_shexp,
                            1,
                            h,
                            &normed_all,
                            &mut gate_logit_all,
                            kk,
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("shexp router gate: {:?}", e))
                        })?;
                        for t in 0..kk {
                            let scale = sigmoid(gate_logit_all[t]);
                            for val in shexp_out_all[t * h..(t + 1) * h].iter_mut() {
                                *val *= scale;
                            }
                        }
                    }
                    for (out, val) in ffn_all.iter_mut().zip(shexp_out_all.iter()) {
                        *out += val;
                    }
                }
            } else {
                let i_size = cfg.intermediate_size;
                let mut gate_all = vec![0.0_f32; kk * i_size];
                let mut up_all = vec![0.0_f32; kk * i_size];
                gemm_weight(&layer.ffn_gate, i_size, h, &normed_all, &mut gate_all, kk)
                    .map_err(|e| ModelError::InferenceFailed(format!("ffn_gate: {:?}", e)))?;
                gemm_weight(&layer.ffn_up, i_size, h, &normed_all, &mut up_all, kk)
                    .map_err(|e| ModelError::InferenceFailed(format!("ffn_up: {:?}", e)))?;
                let mut swiglu_all = vec![0.0_f32; kk * i_size];
                for t in 0..kk {
                    let mut swiglu = vec![0.0_f32; i_size];
                    activate_ffn(
                        &gate_all[t * i_size..(t + 1) * i_size],
                        &up_all[t * i_size..(t + 1) * i_size],
                        cfg.gelu_ffn,
                        &mut swiglu,
                    )?;
                    swiglu_all[t * i_size..(t + 1) * i_size].copy_from_slice(&swiglu);
                }
                gemm_weight(&layer.ffn_down, h, i_size, &swiglu_all, &mut ffn_all, kk)
                    .map_err(|e| ModelError::InferenceFailed(format!("ffn_down: {:?}", e)))?;
                if !layer.ffn_down_bias.is_empty() {
                    for t in 0..kk {
                        for (i, out) in ffn_all[t * h..(t + 1) * h].iter_mut().enumerate() {
                            *out += layer.ffn_down_bias[i % layer.ffn_down_bias.len()];
                        }
                    }
                }
            }

            for t in 0..kk {
                apply_sandwich_norm(&cfg, &layer.post_ffn_norm, &mut ffn_all[t * h..(t + 1) * h])?;
            }
            for (xi, out) in xs.iter_mut().zip(ffn_all.iter()) {
                *xi += out;
            }
            for t in 0..kk {
                trace_fwd("window", start_pos + t, layer_idx, &xs[t * h..(t + 1) * h]);
            }
        }

        self.ssm_pos = start_pos + kk;
        Ok(xs)
    }

    /// LFM2 short-convolution mixer. Returns the residual to add (not applied).
    pub(super) fn run_shortconv_layer(
        &mut self,
        layer_idx: usize,
        layer: &LayerWeights,
        x: &[f32],
        cfg: &InferenceConfig,
    ) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden_size;
        let l_cache = cfg.shortconv_l_cache.max(1);
        let out_dim = layer.shortconv_in_proj.output_dim(h);
        if out_dim == 0 || !out_dim.is_multiple_of(3) {
            return Err(ModelError::InferenceFailed(format!(
                "shortconv in_proj width {out_dim} is not divisible by 3"
            )));
        }
        let d = out_dim / 3;
        let mut normed = vec![0.0_f32; h];
        rms_norm_model(x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed, cfg)?;
        let mut bcx = vec![0.0_f32; 3 * d];
        gemv_weight(&layer.shortconv_in_proj, 3 * d, h, &normed, &mut bcx)
            .map_err(|e| ModelError::InferenceFailed(format!("shortconv_in_proj: {e}")))?;
        let mut bx = vec![0.0_f32; d];
        for i in 0..d {
            bx[i] = bcx[i] * bcx[2 * d + i];
        }
        let mut conv_out = vec![0.0_f32; d];
        let have_conv = layer.shortconv_conv.len() == l_cache * d;
        self.prepare_conv_ring(layer_idx, l_cache, d);
        if have_conv {
            let buf = &self.ssm_conv_buffers[layer_idx];
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
            conv_out.copy_from_slice(&bx);
        }
        for i in 0..d {
            conv_out[i] *= bcx[d + i];
        }
        let mut mixed = vec![0.0_f32; h];
        gemv_weight(&layer.shortconv_out_proj, h, d, &conv_out, &mut mixed)
            .map_err(|e| ModelError::InferenceFailed(format!("shortconv_out_proj: {e}")))?;
        self.ssm_conv_buffers[layer_idx].push(&bx);
        Ok(mixed)
    }

    fn prepare_conv_ring(&mut self, layer_idx: usize, capacity: usize, dim: usize) {
        let ready = self
            .ssm_conv_buffers
            .get(layer_idx)
            .is_some_and(|buf| buf.dim == dim && buf.capacity >= capacity && dim > 0);
        if !ready {
            self.ssm_conv_buffers[layer_idx] = ConvHistoryRing::new(capacity.max(1), dim);
        }
    }
}
