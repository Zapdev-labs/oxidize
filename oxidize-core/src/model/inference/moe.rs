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
