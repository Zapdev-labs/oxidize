use super::*;

impl InferenceModel {
    /// MoE FFN forward for a single token.
    /// 1. Compute router logits: normed @ ffn_gate_inp
    /// 2. Softmax + top-k expert selection
    /// 3. For each selected expert, compute gate/up/down
    /// 4. Weighted sum of expert outputs
    pub(super) fn moe_ffn_forward_single(
        layer: &LayerWeights,
        cfg: &InferenceConfig,
        normed: &[f32],
        ffn_out: &mut [f32],
        gate_scratch: &mut [f32],
        up_scratch: &mut [f32],
        expert_out: &mut [f32],
        router_logits: &mut [f32],
        expert_scores: &mut [(usize, f32)],
    ) -> Result<(), ModelError> {
        moe_ffn_forward_weights(
            &MoeFfnWeights::from_layer(layer),
            cfg,
            normed,
            ffn_out,
            gate_scratch,
            up_scratch,
            expert_out,
            router_logits,
            expert_scores,
        )
    }
}

/// LongCat-2.0 ScMoE router + expert forward for one token. Returns the MoE
/// branch's output in `moe_out` — the caller ([`super::layers`]'s
/// `run_longcat_layer_range`) STASHES this and adds it to the residual only
/// after the pair's second sub-block dense FFN (see `LongcatFlashDecoderLayer`
/// in SGLang's `longcat_flash.py`: `x = moe_hidden_states + hidden_states`
/// where `moe_hidden_states` was computed from sub-block 0's post-attention
/// state, before sub-block 1 ever runs).
///
/// Router/TopK semantics pinned down from SGLang (current `main`,
/// `python/sglang/srt/models/longcat_flash.py` +
/// `python/sglang/srt/layers/moe/topk.py`):
///   - `LongcatFlashRouter` is a plain `ReplicatedLinear(hidden_size,
///     n_routed_experts + zero_expert_num)` run in F32
///     (`rounter_params_dtype`), producing raw logits over ALL
///     `n_routed + zero_expert_num` slots (768 routed + 128 identity
///     zero-experts here) — `LongcatFlashRouter.forward` just returns
///     `self.classifier(hidden_states.to(f32))`, no activation applied yet.
///   - `LongcatFlashMoE.__init__` builds
///     `TopK(top_k=12, renormalize=False, use_grouped_topk=False,
///     correction_bias=self.router.e_score_correction_bias.data)` and rebinds
///     `self.topk.forward = self.topk.forward_native`. `forward_native` sets
///     `torch_native = True` and calls `select_experts`, which — since
///     `use_grouped_topk=False` and there is no `custom_routing_function` —
///     takes the `torch_native` branch straight to
///     `fused_topk_native = fused_topk_torch_native`. LongCat's `TopK(...)`
///     call never passes `scoring_func`, so it keeps `TopKConfig`'s default
///     `scoring_func: str = "softmax"` (NOT sigmoid — that default is easy to
///     miss since most other DeepSeek-family routers in the same file use
///     sigmoid gating). With `correction_bias` set, `fused_topk_torch_native`
///     computes exactly:
/// ```text
/// scores = softmax(logits, dim=-1)                 # unbiased, all 896
/// scores_for_choice = scores + correction_bias      # bias added for SELECTION
///                                                    # only — the DeepSeek-V3
///                                                    # no-aux-loss trick
/// topk_ids = torch.topk(scores_for_choice, k=12)[1] # selection uses the
///                                                    # BIASED score
/// topk_weights = scores.gather(1, topk_ids)         # weights are the
///                                                    # UNBIASED softmax probs
/// ```
///     `renormalize=False` on the `TopK`/`fused_topk_torch_native` call means
///     those 12 gathered weights are used AS-IS — no softmax renormalization
///     over the selected subset (unlike Mixtral/Qwen `norm_topk_prob=True`).
///   - `LongcatFlashMoE.forward`: routed experts (`topk_idx < n_routed_experts`)
///     are summed by `self.experts(...)` with their (unbiased) weight, then
///     `final_hidden_states *= self.routed_scaling_factor` (baked into
///     `cfg.expert_weights_scale`, 9.0 for LongCat-2.0). ONLY AFTER that scale
///     is applied does `zero_experts_compute_triton` add the zero-expert
///     contribution: for `zero_expert_type="identity"` it returns
///     `weight * hidden_states` per selected zero-expert slot (the untouched
///     router input, not run through any FFN) — added UNSCALED by
///     `routed_scaling_factor`.
#[allow(clippy::too_many_arguments)]
pub(super) fn longcat_moe_forward(
    layer: &MoeFfnWeights<'_>,
    cfg: &InferenceConfig,
    normed: &[f32],
    moe_out: &mut [f32],
    gate_scratch: &mut [f32],
    up_scratch: &mut [f32],
    expert_out: &mut [f32],
    router_logits: &mut [f32],
    expert_scores: &mut [(usize, f32)],
) -> Result<(), ModelError> {
    let h = cfg.hidden_size;
    let i_size = if cfg.expert_intermediate_size > 0 {
        cfg.expert_intermediate_size
    } else {
        cfg.intermediate_size
    };
    let n_routed = cfg.num_experts;
    let total = n_routed + cfg.zero_expert_count;
    let top_k = cfg.num_experts_per_tok.max(1).min(total.max(1));

    // 1. Router logits (F32 classifier) over all `total` slots.
    router_logits.fill(0.0_f32);
    gemv_weight(layer.gate_inp, total, h, normed, router_logits)
        .map_err(|e| ModelError::InferenceFailed(format!("longcat router: {:?}", e)))?;

    // 2. Softmax over ALL slots — these are the UNBIASED per-expert weights.
    let max_logit = router_logits
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum_exp = 0.0_f32;
    for v in router_logits.iter_mut() {
        *v = (*v - max_logit).exp();
        sum_exp += *v;
    }
    if sum_exp > 0.0 {
        for v in router_logits.iter_mut() {
            *v /= sum_exp;
        }
    }

    // 3. Selection score = unbiased softmax + correction bias. Bias is used for
    // SELECTION only; the gathered weight below reads the unbiased score.
    for (i, &w) in router_logits.iter().enumerate() {
        let bias = layer.exp_probs_b.get(i).copied().unwrap_or(0.0);
        expert_scores[i] = (i, w + bias);
    }

    // 4. Top-k by selection score. `renormalize=False`: no renorm over the
    // selected subset (unlike the generic Mixtral/Qwen/DeepSeek MoE path).
    let compare_score = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    };
    if top_k < expert_scores.len() {
        let (selected, _, _) = expert_scores.select_nth_unstable_by(top_k, compare_score);
        selected.sort_by(compare_score);
    } else {
        expert_scores.sort_by(compare_score);
    }

    // 5. Split the top-k selection into routed experts (idx < n_routed) and
    // zero (identity) experts (idx >= n_routed). Zero-expert contributions are
    // added directly here (unscaled); routed weights are pre-scaled by
    // `expert_weights_scale` (routed_scaling_factor) so the shared expert-GEMV
    // plumbing below applies it uniformly — scalar multiplication distributes
    // over the sum, so `scale * sum(w_i * e_i)` and `sum(scale*w_i * e_i)` are
    // the same value, matching `final_hidden_states *= routed_scaling_factor`
    // being applied to the routed SUM in SGLang.
    let routed_scale = if cfg.expert_weights_scale > 0.0 {
        cfg.expert_weights_scale
    } else {
        1.0
    };
    let mut routed_idx: Vec<usize> = Vec::with_capacity(top_k);
    let mut routed_weight: Vec<f32> = Vec::with_capacity(top_k);
    moe_out.fill(0.0_f32);
    for &(idx, _score) in expert_scores.iter().take(top_k) {
        let w = router_logits[idx];
        if idx < n_routed {
            routed_idx.push(idx);
            routed_weight.push(routed_scale * w);
        } else {
            for (out, &x) in moe_out.iter_mut().zip(normed.iter()) {
                *out += w * x;
            }
        }
    }
    if routed_idx.is_empty() {
        return Ok(());
    }
    let n_sel = routed_idx.len();

    // 6. Routed expert FFN (SiLU-gated), batched quantized path when available,
    // per-expert f32 fallback otherwise — same GEMV plumbing as the generic
    // MoE path (`moe_ffn_forward_weights`), just without its fused gate+up
    // micro-opt (ponytail: LongCat's routed-expert FFN is a small slice of
    // total decode time next to the 8192-wide dense FFN/attention; add the
    // fused thread-local buffer trick here if profiling shows it matters).
    if let (Some((gq, gm)), Some((uq, um)), Some((dq, dm))) = (
        expert_matrix(layer.gate_exps),
        expert_matrix(layer.up_exps),
        expert_matrix(layer.down_exps),
    ) {
        let gate_all = &mut gate_scratch[..n_sel * i_size];
        let up_all = &mut up_scratch[..n_sel * i_size];
        gate_all.fill(0.0_f32);
        up_all.fill(0.0_f32);
        gemv_quantized_experts_f32(
            gq,
            gm,
            n_routed,
            &routed_idx,
            i_size,
            h,
            normed,
            0,
            gate_all,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("longcat moe gate: {:?}", e)))?;
        gemv_quantized_experts_f32(uq, um, n_routed, &routed_idx, i_size, h, normed, 0, up_all)
            .map_err(|e| ModelError::InferenceFailed(format!("longcat moe up: {:?}", e)))?;
        for (g, u) in gate_all.iter_mut().zip(up_all.iter()) {
            let sigmoid = 1.0_f32 / (1.0 + (-*g).exp());
            *g = *g * sigmoid * *u;
        }
        let down_all = &mut expert_out[..n_sel * h];
        down_all.fill(0.0_f32);
        gemv_quantized_experts_f32(
            dq,
            dm,
            n_routed,
            &routed_idx,
            h,
            i_size,
            gate_all,
            i_size,
            down_all,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("longcat moe down: {:?}", e)))?;
        for (slot, &weight) in routed_weight.iter().enumerate() {
            let d = &down_all[slot * h..(slot + 1) * h];
            for (out, &val) in moe_out.iter_mut().zip(d.iter()) {
                *out += weight * val;
            }
        }
        return Ok(());
    }

    // Fallback: per-expert FFN for f32 expert weights (small/test models).
    for (slot, &expert_idx) in routed_idx.iter().enumerate() {
        let weight = routed_weight[slot];
        let gate = &mut gate_scratch[..i_size];
        let up = &mut up_scratch[..i_size];
        gate.fill(0.0_f32);
        up.fill(0.0_f32);
        expert_out[..h].fill(0.0_f32);
        gemv_expert_weight(
            layer.gate_exps,
            expert_idx,
            n_routed,
            i_size,
            h,
            normed,
            gate,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("longcat moe gate: {:?}", e)))?;
        gemv_expert_weight(layer.up_exps, expert_idx, n_routed, i_size, h, normed, up)
            .map_err(|e| ModelError::InferenceFailed(format!("longcat moe up: {:?}", e)))?;
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            let sigmoid = 1.0_f32 / (1.0 + (-*g).exp());
            *g = *g * sigmoid * *u;
        }
        gemv_expert_weight(
            layer.down_exps,
            expert_idx,
            n_routed,
            h,
            i_size,
            gate,
            &mut expert_out[..h],
        )
        .map_err(|e| ModelError::InferenceFailed(format!("longcat moe down: {:?}", e)))?;
        for (out, &val) in moe_out.iter_mut().zip(expert_out[..h].iter()) {
            *out += weight * val;
        }
    }
    Ok(())
}
