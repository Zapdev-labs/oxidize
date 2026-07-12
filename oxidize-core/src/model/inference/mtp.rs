use super::*;

impl InferenceModel {
    /// Generate draft tokens with the native in-GGUF MTP/nextn block.
    ///
    /// `start_token` and `start_hidden` must describe the same committed target
    /// position. The first MTP step predicts the token after `start_token`; each
    /// accepted MTP row then feeds its sampled token and post-head-norm hidden row
    /// back into the next MTP step.
    pub fn draft_mtp_tokens(
        &mut self,
        start_token: Token,
        start_hidden: &[f32],
        max_tokens: usize,
        sampling: crate::sampling::SamplingConfig,
        random: &mut dyn FnMut() -> f32,
        quantspec_draft_kv: bool,
    ) -> Result<(Vec<Token>, Vec<Logits>), ModelError> {
        if max_tokens == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if self.mtp.is_none() {
            return Err(ModelError::InferenceFailed(
                "model does not contain a usable MTP/nextn block".to_string(),
            ));
        }
        let h = self.config.hidden_size;
        if start_hidden.len() != h {
            return Err(ModelError::InferenceFailed(format!(
                "MTP hidden width mismatch: expected {h}, got {}",
                start_hidden.len()
            )));
        }

        let mtp_kv_config = KvCacheConfig {
            layer_count: 1,
            context_size: max_tokens.max(1),
            head_count: self.config.num_key_value_heads,
            head_dim: self.config.kv_head_dim(),
            dtype: if quantspec_draft_kv {
                DType::I8
            } else {
                DType::F32
            },
            quantization: if quantspec_draft_kv {
                crate::kv_cache::KvQuantization::TurboQuant
            } else {
                crate::kv_cache::KvQuantization::default()
            },
        };
        let mut mtp_kv = KvCache::new(mtp_kv_config)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp kv_cache: {e:?}")))?;

        let mut draft_tokens = Vec::with_capacity(max_tokens);
        let mut draft_logits = Vec::with_capacity(max_tokens);
        let mut current_token = start_token;
        let mut current_hidden = start_hidden.to_vec();
        for pos in 0..max_tokens {
            let (logits, next_hidden) =
                self.mtp_forward_one(current_token, &current_hidden, pos, &mut mtp_kv)?;
            let token = crate::sampling::sample(&logits, sampling, random())
                .map_err(|e| ModelError::InferenceFailed(format!("MTP sample: {e:?}")))?;
            draft_tokens.push(token);
            draft_logits.push(logits);
            current_token = token;
            current_hidden = next_hidden;
        }

        Ok((draft_tokens, draft_logits))
    }

    fn mtp_forward_one(
        &mut self,
        token: Token,
        previous_hidden: &[f32],
        pos: usize,
        mtp_kv: &mut KvCache,
    ) -> Result<(Logits, Vec<f32>), ModelError> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| ModelError::InferenceFailed("missing MTP/nextn weights".to_string()))?;
        let h = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let rms_norm_eps = self.config.rms_norm_eps;

        let embed_storage = if mtp.embed_tokens.is_empty() {
            &self.tok_embeddings
        } else {
            &mtp.embed_tokens
        };
        let mut token_embedding = vec![0.0_f32; h];
        lookup_embedding_from_storage(embed_storage, h, vocab_size, token, &mut token_embedding);

        let mut embed_normed = vec![0.0_f32; h];
        rms_norm_f32(
            &token_embedding,
            &mtp.enorm,
            rms_norm_eps,
            &mut embed_normed,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp enorm: {e:?}")))?;
        let mut hidden_normed = vec![0.0_f32; h];
        rms_norm_f32(
            previous_hidden,
            &mtp.hnorm,
            rms_norm_eps,
            &mut hidden_normed,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp hnorm: {e:?}")))?;

        let mut concat = vec![0.0_f32; h * 2];
        concat[..h].copy_from_slice(&embed_normed);
        concat[h..].copy_from_slice(&hidden_normed);

        let mut fused = vec![0.0_f32; h];
        gemv_weight(&mtp.eh_proj, h, h * 2, &concat, &mut fused)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp eh_proj: {e}")))?;
        self.workspace.x[..h].copy_from_slice(&fused);

        self.run_mtp_layer_in_workspace(pos, mtp_kv)?;

        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| ModelError::InferenceFailed("missing MTP/nextn weights".to_string()))?;
        let norm_weight = if mtp.shared_head_norm.is_empty() {
            &self.norm_weight
        } else {
            &mtp.shared_head_norm
        };
        let head_weight = if mtp.shared_head_head.is_empty() {
            &self.output_weight
        } else {
            &mtp.shared_head_head
        };

        let x = self.workspace.x[..h].to_vec();
        let mut mtp_hidden = vec![0.0_f32; h];
        rms_norm_f32(&x, norm_weight, rms_norm_eps, &mut mtp_hidden)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp shared_head_norm: {e:?}")))?;
        let mut logits = vec![0.0_f32; vocab_size];
        gemv_weight(head_weight, vocab_size, h, &mtp_hidden, &mut logits)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp shared_head: {e}")))?;
        Ok((logits, mtp_hidden))
    }

    fn run_mtp_layer_in_workspace(
        &mut self,
        pos: usize,
        mtp_kv: &mut KvCache,
    ) -> Result<(), ModelError> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| ModelError::InferenceFailed("missing MTP/nextn weights".to_string()))?;
        let layer = &mtp.layer;
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let k = cfg.num_key_value_heads;
        let mut x = self.workspace.x[..h].to_vec();

        let mut normed = vec![0.0_f32; h];
        rms_norm_f32(&x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp attn_norm: {e:?}")))?;

        let qg_len = layer.attn_q.output_dim(h);
        let kv_len = layer.attn_k.output_dim(h);
        let attn_output_input_len = layer.attn_output.output_dim(h);
        if qg_len == 0 || kv_len == 0 || attn_output_input_len == 0 {
            return Err(ModelError::InferenceFailed(format!(
                "invalid MTP attention dims qg={qg_len} kv={kv_len} out_in={attn_output_input_len}"
            )));
        }

        let mut qg = vec![0.0_f32; qg_len];
        let mut k_vec = vec![0.0_f32; kv_len];
        let mut v_vec = vec![0.0_f32; kv_len];
        gemv_weight_fused(
            vec![
                (&layer.attn_q, qg_len, &mut qg[..]),
                (&layer.attn_k, kv_len, &mut k_vec[..]),
                (&layer.attn_v, kv_len, &mut v_vec[..]),
            ],
            h,
            &normed,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp qkv: {e}")))?;
        if !layer.attn_q_bias.is_empty() {
            for (i, q) in qg.iter_mut().enumerate() {
                *q += layer.attn_q_bias[i % layer.attn_q_bias.len()];
            }
        }
        if !layer.attn_k_bias.is_empty() {
            for (i, value) in k_vec.iter_mut().enumerate() {
                *value += layer.attn_k_bias[i % layer.attn_k_bias.len()];
            }
        }
        if !layer.attn_v_bias.is_empty() {
            for (i, value) in v_vec.iter_mut().enumerate() {
                *value += layer.attn_v_bias[i % layer.attn_v_bias.len()];
            }
        }

        let q_len = qg_len.min(attn_output_input_len);
        let gate = (qg_len >= q_len.saturating_mul(2)).then(|| qg[q_len..q_len + q_len].to_vec());
        let mut q = qg[..q_len].to_vec();
        let q_head_dim = if n > 0 && q_len.is_multiple_of(n) {
            q_len / n
        } else {
            q_len
        };
        let q_heads = q_len.checked_div(q_head_dim.max(1)).unwrap_or(1);
        let kv_head_dim = if k > 0 && kv_len.is_multiple_of(k) {
            kv_len / k
        } else {
            kv_len
        };
        let kv_heads = kv_len.checked_div(kv_head_dim.max(1)).unwrap_or(1);

        if !layer.attn_q_norm.is_empty() && q.len() == layer.attn_q_norm.len() {
            let mut normed_q = vec![0.0_f32; q.len()];
            rms_norm_f32(&q, &layer.attn_q_norm, cfg.rms_norm_eps, &mut normed_q)
                .map_err(|e| ModelError::InferenceFailed(format!("mtp q_norm: {e:?}")))?;
            q.copy_from_slice(&normed_q);
        } else if !layer.attn_q_norm.is_empty() && q_head_dim == layer.attn_q_norm.len() {
            let mut normed_head = vec![0.0_f32; q_head_dim];
            for head in 0..q_heads {
                let start = head * q_head_dim;
                let end = start + q_head_dim;
                if end > q.len() {
                    break;
                }
                rms_norm_f32(
                    &q[start..end],
                    &layer.attn_q_norm,
                    cfg.rms_norm_eps,
                    &mut normed_head,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("mtp q_norm: {e:?}")))?;
                q[start..end].copy_from_slice(&normed_head);
            }
        }
        if !layer.attn_k_norm.is_empty() && k_vec.len() == layer.attn_k_norm.len() {
            let mut normed_k = vec![0.0_f32; k_vec.len()];
            rms_norm_f32(&k_vec, &layer.attn_k_norm, cfg.rms_norm_eps, &mut normed_k)
                .map_err(|e| ModelError::InferenceFailed(format!("mtp k_norm: {e:?}")))?;
            k_vec.copy_from_slice(&normed_k);
        } else if !layer.attn_k_norm.is_empty() && kv_head_dim == layer.attn_k_norm.len() {
            let mut normed_head = vec![0.0_f32; kv_head_dim];
            for head in 0..kv_heads {
                let start = head * kv_head_dim;
                let end = start + kv_head_dim;
                if end > k_vec.len() {
                    break;
                }
                rms_norm_f32(
                    &k_vec[start..end],
                    &layer.attn_k_norm,
                    cfg.rms_norm_eps,
                    &mut normed_head,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("mtp k_norm: {e:?}")))?;
                k_vec[start..end].copy_from_slice(&normed_head);
            }
        }

        let q_rope_len = cfg.effective_rope_dim().min(q_head_dim);
        let mut rope_scratch = vec![0.0_f32; q_rope_len.max(kv_head_dim)];
        for head in 0..q_heads {
            let off = head * q_head_dim;
            if off + q_head_dim > q.len() {
                break;
            }
            let rotated = &mut rope_scratch[..q_rope_len];
            cfg.apply_rope_head(
                &q[off..off + q_rope_len],
                pos,
                q_rope_len,
                cfg.rope_theta,
                rotated,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("mtp rope q: {e:?}")))?;
            q[off..off + q_rope_len].copy_from_slice(rotated);
        }
        let k_rope_len = cfg.effective_rope_dim().min(kv_head_dim);
        for head in 0..kv_heads {
            let off = head * kv_head_dim;
            if off + kv_head_dim > k_vec.len() {
                break;
            }
            let rotated = &mut rope_scratch[..k_rope_len];
            cfg.apply_rope_head(
                &k_vec[off..off + k_rope_len],
                pos,
                k_rope_len,
                cfg.rope_theta,
                rotated,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("mtp rope k: {e:?}")))?;
            k_vec[off..off + k_rope_len].copy_from_slice(rotated);
        }

        mtp_kv
            .set(0, pos, &k_vec, &v_vec)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp kv set: {e:?}")))?;
        let seq_len = pos + 1;
        let key_cache = mtp_kv
            .f32_layer_key_prefix(0, seq_len)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp kv keys: {e:?}")))?
            .ok_or_else(|| ModelError::InferenceFailed("MTP KV cache is not f32".to_string()))?;
        let value_cache = mtp_kv
            .f32_layer_value_prefix(0, seq_len)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp kv values: {e:?}")))?
            .ok_or_else(|| ModelError::InferenceFailed("MTP KV cache is not f32".to_string()))?;

        let q_for_flash = if q_head_dim > kv_head_dim {
            let mut truncated = vec![0.0_f32; q_heads * kv_head_dim];
            for head in 0..q_heads {
                let src = head * q_head_dim;
                let dst = head * kv_head_dim;
                truncated[dst..dst + kv_head_dim].copy_from_slice(&q[src..src + kv_head_dim]);
            }
            truncated
        } else {
            q.clone()
        };
        let mut attn_result = vec![0.0_f32; q_for_flash.len()];
        flash_attention_decode_heads_f32(
            &q_for_flash,
            key_cache,
            value_cache,
            seq_len,
            kv_head_dim,
            kv_len,
            q_heads,
            kv_heads,
            &mut attn_result,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp attention: {e:?}")))?;
        if let Some(gate) = gate.as_ref()
            && gate.len() == attn_result.len()
        {
            for (out, gate_value) in attn_result.iter_mut().zip(gate.iter()) {
                let sigmoid = 1.0_f32 / (1.0 + (-*gate_value).exp());
                *out *= sigmoid;
            }
        }

        let attn_input = if attn_result.len() == attn_output_input_len {
            attn_result
        } else {
            let mut padded = vec![0.0_f32; attn_output_input_len];
            let copy = padded.len().min(attn_result.len());
            padded[..copy].copy_from_slice(&attn_result[..copy]);
            padded
        };
        let mut attn_out = vec![0.0_f32; h];
        gemv_weight(
            &layer.attn_output,
            h,
            attn_output_input_len,
            &attn_input,
            &mut attn_out,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp attn_output: {e}")))?;
        if !layer.attn_output_bias.is_empty() {
            for (i, out) in attn_out.iter_mut().enumerate() {
                *out += layer.attn_output_bias[i % layer.attn_output_bias.len()];
            }
        }
        for i in 0..h {
            x[i] += attn_out[i];
        }

        let ffn_norm_weight = if !layer.post_attention_norm.is_empty() {
            &layer.post_attention_norm
        } else {
            &layer.ffn_norm
        };
        if ffn_norm_weight.is_empty() {
            return Err(ModelError::InferenceFailed(
                "MTP block is missing post_attention_norm/ffn_norm".to_string(),
            ));
        }
        let mut ffn_normed = vec![0.0_f32; h];
        rms_norm_f32(&x, ffn_norm_weight, cfg.rms_norm_eps, &mut ffn_normed)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp ffn_norm: {e:?}")))?;
        let mut gate = vec![0.0_f32; cfg.intermediate_size];
        let mut up = vec![0.0_f32; cfg.intermediate_size];
        gemv_weight_fused(
            vec![
                (&layer.ffn_gate, cfg.intermediate_size, &mut gate[..]),
                (&layer.ffn_up, cfg.intermediate_size, &mut up[..]),
            ],
            h,
            &ffn_normed,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp ffn gate/up: {e}")))?;
        if cfg.gelu_ffn {
            apply_geglu_inplace_f32(&mut gate, &up);
        } else {
            apply_swiglu_inplace_f32(&mut gate, &up);
        }
        let mut ffn_out = vec![0.0_f32; h];
        gemv_weight(
            &layer.ffn_down,
            h,
            cfg.intermediate_size,
            &gate,
            &mut ffn_out,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp ffn_down: {e}")))?;
        if !layer.ffn_down_bias.is_empty() {
            for (i, out) in ffn_out.iter_mut().enumerate() {
                *out += layer.ffn_down_bias[i % layer.ffn_down_bias.len()];
            }
        }
        for i in 0..h {
            x[i] += ffn_out[i];
        }

        self.workspace.x[..h].copy_from_slice(&x);
        Ok(())
    }
}
