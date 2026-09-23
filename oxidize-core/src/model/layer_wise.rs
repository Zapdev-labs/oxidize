#![allow(clippy::needless_range_loop, clippy::manual_checked_ops, dead_code)]

use crate::conversion::normalize_gguf_tensor_name;
use crate::flash_attention::flash_attention_decode_f32;
use crate::gguf::{GgufQuantizationType, GgufTensorInfo, MappedGgufFile};
use crate::inference::{
    InferenceConfig, MoeFfnWeights, WeightStorage, lookup_quantized_embedding,
    moe_ffn_forward_weights,
};
use crate::kv_cache::KvCache;
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::quantization::{dequantize_scalar, quant_block_layout, quantized_size};
use crate::tensor::{
    apply_geglu_inplace_f32, apply_rope_f32, apply_swiglu_f32, gemm_quantized_f32, gemv_f32,
    gemv_quantized_f32, rms_norm_f32,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

#[path = "layer_wise/attention.rs"]
mod attention;
#[path = "layer_wise/cache.rs"]
mod cache;
#[path = "layer_wise/forward.rs"]
mod forward;
#[path = "layer_wise/loading.rs"]
mod loading;
#[path = "layer_wise/ssm.rs"]
mod ssm;
#[path = "layer_wise/weights.rs"]
mod weights;

use cache::{GgufTensorRef, LayerCache};
use ssm::ConvHistoryRing;
use weights::LayerWeights;

#[derive(Debug, Clone, PartialEq)]
pub struct LayerWiseModel {
    config: InferenceConfig,
    mmap: Arc<MappedGgufFile>,
    layer_tensors: Vec<HashMap<String, GgufTensorRef>>,
    tok_embeddings: WeightStorage,
    tok_embeddings_cols: usize,
    norm_weight: Vec<f32>,
    output_weight: WeightStorage,
    kv_cache: KvCache,
    ssm_states: Vec<Vec<f32>>,
    ssm_conv_buffers: Vec<ConvHistoryRing>,
    /// Number of tokens applied to the recurrent (GDN) state so far.
    ssm_pos: usize,
    /// Snapshots of (position, ssm_states, conv rings) for speculative
    /// rollback: unlike the KV cache, recurrent state is not
    /// position-addressable, so rewinding requires restoring a checkpoint.
    /// Two entries are live per speculative round (the rollback target set at
    /// the pre-verify rewind, plus the forward_many entry position).
    ssm_checkpoints: Vec<(usize, Vec<Vec<f32>>, Vec<ConvHistoryRing>)>,
    cache: LayerCache,
}

fn quant_block_info(qtype: GgufQuantizationType) -> (usize, usize) {
    match qtype {
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => (256, 144),
        GgufQuantizationType::Q6_K => (256, 210),
        GgufQuantizationType::Q8_0 => (32, 34),
        _ => (1, 4),
    }
}

fn weight_is_empty(ws: &WeightStorage) -> bool {
    match ws {
        WeightStorage::F32(v) => v.is_empty(),
        WeightStorage::Quantized(_, v) => v.is_empty(),
        WeightStorage::MmapQuantized(_, _, _, size) => *size == 0,
    }
}

fn weight_output_dim(ws: &WeightStorage, input_dim: usize) -> usize {
    match ws {
        WeightStorage::F32(v) => v.len() / input_dim,
        WeightStorage::Quantized(qtype, v) => {
            let (block_width, block_size) = quant_block_info(*qtype);
            let bytes_per_row = (input_dim / block_width) * block_size;
            if bytes_per_row == 0 {
                return 0;
            }
            v.len() / bytes_per_row
        }
        WeightStorage::MmapQuantized(qtype, _, _, size) => {
            let (block_width, block_size) = quant_block_info(*qtype);
            let bytes_per_row = (input_dim / block_width) * block_size;
            if bytes_per_row == 0 {
                return 0;
            }
            size / bytes_per_row
        }
    }
}

fn weight_input_dim(ws: &WeightStorage, output_dim: usize) -> usize {
    if output_dim == 0 {
        return 0;
    }
    match ws {
        WeightStorage::F32(v) => v.len() / output_dim,
        WeightStorage::Quantized(qtype, v) => {
            let (block_width, block_size) = quant_block_info(*qtype);
            let bytes_per_row = v.len() / output_dim;
            if block_size == 0 {
                return 0;
            }
            (bytes_per_row / block_size) * block_width
        }
        WeightStorage::MmapQuantized(qtype, _, _, size) => {
            let (block_width, block_size) = quant_block_info(*qtype);
            let bytes_per_row = size / output_dim;
            if block_size == 0 {
                return 0;
            }
            (bytes_per_row / block_size) * block_width
        }
    }
}

fn gemv_weight(
    storage: &WeightStorage,
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    match storage {
        WeightStorage::F32(data) => {
            gemv_f32(data, rows, cols, input, output).map_err(|e| format!("{:?}", e))
        }
        WeightStorage::Quantized(qtype, data) => {
            gemv_quantized_f32(*qtype, data, rows, cols, input, output)
                .map_err(|e| format!("{:?}", e))
        }
        WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
            let data = &mmap[*offset..*offset + *size];
            gemv_quantized_f32(*qtype, data, rows, cols, input, output)
                .map_err(|e| format!("{:?}", e))
        }
    }
}

/// Batched [`gemv_weight`]: `inputs` is `batch` row-major vectors of `cols`,
/// `outputs` is `batch` vectors of `rows`. Quantized storage goes through the
/// fused GEMM kernels, which decode each weight block once and reuse it across
/// the whole batch — this is what amortizes weight reads during prefill and
/// speculative verification.
fn gemm_weight(
    storage: &WeightStorage,
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), String> {
    match storage {
        WeightStorage::F32(data) => {
            for t in 0..batch {
                gemv_f32(
                    data,
                    rows,
                    cols,
                    &inputs[t * cols..(t + 1) * cols],
                    &mut outputs[t * rows..(t + 1) * rows],
                )
                .map_err(|e| format!("{:?}", e))?;
            }
            Ok(())
        }
        WeightStorage::Quantized(qtype, data) => {
            gemm_quantized_f32(*qtype, data, rows, cols, inputs, outputs, batch)
                .map_err(|e| format!("{:?}", e))
        }
        WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
            let data = &mmap[*offset..*offset + *size];
            gemm_quantized_f32(*qtype, data, rows, cols, inputs, outputs, batch)
                .map_err(|e| format!("{:?}", e))
        }
    }
}

fn l2_normalize(v: &mut [f32]) {
    let mut sum = 0.0_f32;
    for x in v.iter() {
        sum += x * x;
    }
    let inv = 1.0_f32 / (sum + 1e-6_f32).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0_f32 {
        x
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

fn gated_rms_norm(x: &mut [f32], weight: &[f32], gate: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    // llama.cpp's GDN gated RMSNorm uses a near-zero eps; oxidize's model eps
    // (1e-6) over-floors near-orthogonal-qk heads whose delta output is tiny.
    let eps = std::env::var("OXIDIZE_GDN_EPS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(eps);
    if std::env::var_os("OXIDIZE_GDN_GATE_FIRST").is_some() {
        // HF Qwen3NextRMSNormGated order (gate before norm).
        for i in 0..n {
            let g = gate.get(i).copied().unwrap_or(0.0_f32);
            let silu = g * (1.0_f32 / (1.0_f32 + (-g).exp()));
            x[i] *= silu;
        }
        let mut var = 0.0_f32;
        for val in x.iter() {
            var += val * val;
        }
        var /= n as f32;
        let inv = 1.0_f32 / (var + eps).sqrt();
        for i in 0..n {
            let w = weight.get(i).copied().unwrap_or(1.0_f32);
            x[i] = x[i] * inv * w;
        }
        return;
    }
    // Gate-after order (matches llama.cpp's qwen3next graph): rmsnorm * weight * silu(gate).
    let mut var = 0.0_f32;
    for val in x.iter() {
        var += val * val;
    }
    var /= n as f32;
    let inv = 1.0_f32 / (var + eps).sqrt();
    for i in 0..n {
        let w = weight.get(i).copied().unwrap_or(1.0_f32);
        let g = gate.get(i).copied().unwrap_or(0.0_f32);
        let silu = g * (1.0_f32 / (1.0_f32 + (-g).exp()));
        x[i] = x[i] * inv * w * silu;
    }
}

/// Pre-FFN norm. Gemma sandwich mode keeps `post_attention_norm` for the
/// attention output, so the FFN reads `ffn_norm` instead.
fn select_ffn_norm<'a>(cfg: &InferenceConfig, layer: &'a LayerWeights) -> &'a [f32] {
    if cfg.sandwich_norm {
        &layer.ffn_norm
    } else if !layer.post_attention_norm.is_empty() {
        &layer.post_attention_norm
    } else {
        &layer.ffn_norm
    }
}

fn apply_sandwich_norm(
    cfg: &InferenceConfig,
    weight: &[f32],
    data: &mut [f32],
) -> Result<(), ModelError> {
    if !cfg.sandwich_norm || weight.is_empty() || data.is_empty() {
        return Ok(());
    }
    let mut tmp = vec![0.0_f32; data.len()];
    rms_norm_model(data, weight, cfg.rms_norm_eps, &mut tmp, cfg)?;
    data.copy_from_slice(&tmp);
    Ok(())
}

/// SwiGLU, or Gemma's tanh-GELU GeGLU when `gelu` is set.
fn activate_ffn(
    gate: &[f32],
    up: &[f32],
    gelu: bool,
    output: &mut [f32],
) -> Result<(), ModelError> {
    if gelu {
        if gate.len() != output.len() || up.len() < gate.len() {
            return Err(ModelError::InferenceFailed(
                "geglu: dimension mismatch".to_owned(),
            ));
        }
        output.copy_from_slice(gate);
        apply_geglu_inplace_f32(output, up);
        Ok(())
    } else {
        apply_swiglu_f32(gate, up, output)
            .map_err(|e| ModelError::InferenceFailed(format!("swiglu: {:?}", e)))
    }
}

/// ALiBi slope for `head`, matching the MLX path: `-(2^(-8/n))^(head+1)`.
pub(crate) fn alibi_slope(head: usize, n_heads: usize) -> f32 {
    let n = n_heads.max(1);
    let base = 2.0_f32.powf(-(8.0_f32 / n as f32));
    -(base.powf(head as f32 + 1.0))
}

fn scale_hidden(xs: &mut [f32], scale: f32) {
    if scale != 1.0 {
        for v in xs.iter_mut() {
            *v *= scale;
        }
    }
}

/// KV cache geometry. MLA stores the decompressed per-head K/V
/// (`n_heads * kv_head_dim`), not the compressed latent.
pub(super) fn kv_cache_geometry(config: &InferenceConfig) -> (usize, usize) {
    let head_dim = config.kv_head_dim().max(1);
    if config.architecture.uses_mla() {
        (config.num_attention_heads.max(1), head_dim)
    } else {
        (config.num_key_value_heads.max(1), head_dim)
    }
}

fn rms_norm_model(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
    cfg: &InferenceConfig,
) -> Result<(), ModelError> {
    if cfg.rms_norm_weight_plus_one {
        let hidden_dim = output.len();
        if hidden_dim == 0 || input.len() != hidden_dim || weight.len() != hidden_dim {
            return Err(ModelError::InferenceFailed(
                "rms_norm: dimension mismatch".to_owned(),
            ));
        }
        let sum_sq: f32 = input.iter().map(|v| v * v).sum();
        let inv_rms = 1.0_f32 / (sum_sq / hidden_dim as f32 + eps).sqrt();
        for ((value, w), out) in input.iter().zip(weight.iter()).zip(output.iter_mut()) {
            *out = value * inv_rms * (1.0_f32 + w);
        }
        Ok(())
    } else {
        rms_norm_f32(input, weight, eps, output)
            .map_err(|e| ModelError::InferenceFailed(format!("rms_norm: {:?}", e)))
    }
}

fn split_gated_query_proj(q_full: &[f32], head_dim: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    if head_dim == 0 || !q_full.len().is_multiple_of(2 * head_dim) {
        return None;
    }
    let num_heads = q_full.len() / (2 * head_dim);
    let mut query = vec![0.0_f32; num_heads * head_dim];
    let mut gate = vec![0.0_f32; num_heads * head_dim];
    for head in 0..num_heads {
        let base = head * 2 * head_dim;
        query[head * head_dim..(head + 1) * head_dim]
            .copy_from_slice(&q_full[base..base + head_dim]);
        gate[head * head_dim..(head + 1) * head_dim]
            .copy_from_slice(&q_full[base + head_dim..base + 2 * head_dim]);
    }
    Some((query, gate))
}

fn debug_vec(label: &str, x: &[f32]) {
    if std::env::var("OXIDIZE_DEBUG_LAYERS").is_err() {
        return;
    }
    let nan_count = x.iter().filter(|v| v.is_nan()).count();
    let inf_count = x.iter().filter(|v| v.is_infinite()).count();
    let max_abs = x
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    let large = x
        .iter()
        .filter(|v| v.is_finite() && v.abs() > 1000.0)
        .count();
    eprintln!("{label} nan={nan_count} inf={inf_count} max_abs={max_abs} gt1k={large}");
}

/// Per-layer hidden-state checksum tracing (OXIDIZE_TRACE_FWD=1) for
/// diffing the batched window path against the per-token path.
fn trace_fwd(path: &str, pos: usize, layer: usize, x: &[f32]) {
    if crate::inference::trace_fwd_enabled() {
        let sum: f64 = x.iter().map(|v| *v as f64).sum();
        // OXIDIZE_TRACE_VALS=1 also prints the first 8 residual values so the
        // stream can be diffed value-for-value against a reference (llama.cpp
        // eval-callback) — sums alone can match by luck.
        if crate::inference::trace_vals_enabled() {
            let head: Vec<String> = x.iter().take(8).map(|v| format!("{v:.5}")).collect();
            eprintln!(
                "TRACE {path} pos={pos} layer={layer} sum={sum:.9e} vals=[{}]",
                head.join(",")
            );
        } else {
            eprintln!("TRACE {path} pos={pos} layer={layer} sum={sum:.9e}");
        }
    }
}

fn debug_hidden(label: &str, pos: usize, x: &[f32]) {
    if pos == 0 {
        debug_vec(label, x);
    }
}

impl LayerWiseModel {
    /// Tokens processed per batched layer-major pass in `forward`/`forward_many`.
    /// Larger windows amortize weight reads further but grow activation scratch
    /// linearly; 16 keeps scratch in the tens of MB for typical models.
    const FORWARD_WINDOW: usize = 16;

    /// Window size for batched forward; `OXIDIZE_WINDOW_BISECT=off` forces the
    /// per-token path everywhere, `=fwd` only in `forward`, `=many` only in
    /// `forward_many` (debugging / A-B reference).
    fn forward_window_size_for(caller: &str) -> usize {
        let v = std::env::var("OXIDIZE_WINDOW_BISECT").unwrap_or_default();
        if v.contains("off") || v.contains(caller) {
            1
        } else {
            Self::FORWARD_WINDOW
        }
    }

    /// Access the model's inference configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

impl Model for LayerWiseModel {
    fn rewind_to(&mut self, consumed_tokens: usize) -> Result<(), ModelError> {
        let position = if consumed_tokens == 0 {
            0
        } else {
            consumed_tokens.saturating_sub(1)
        };
        if consumed_tokens == 0 {
            // Fresh session: recurrent GDN state is not position-addressable
            // like the KV cache, so it must be cleared explicitly or every new
            // request is conditioned on leftover state from the previous one.
            for state in self.ssm_states.iter_mut() {
                state.fill(0.0_f32);
            }
            for buffer in self.ssm_conv_buffers.iter_mut() {
                let dim = buffer.dim;
                if dim > 0 {
                    *buffer = ConvHistoryRing::new(4, dim);
                }
            }
            self.ssm_pos = 0;
            self.ssm_checkpoints.clear();
        } else if consumed_tokens == self.ssm_pos {
            // State already sits at the target position. Capture a checkpoint:
            // the speculative loop rewinds here, forwards the pending token,
            // verifies a draft window, then rolls back to this position.
            self.push_ssm_checkpoint(consumed_tokens);
        } else if let Some(idx) = self
            .ssm_checkpoints
            .iter()
            .position(|(p, _, _)| *p == consumed_tokens)
        {
            // Speculative rollback: restore the recurrent state snapshot taken
            // before the rejected draft window was processed.
            let (_, states, bufs) = &self.ssm_checkpoints[idx];
            self.ssm_states = states.clone();
            self.ssm_conv_buffers = bufs.clone();
            self.ssm_pos = consumed_tokens;
            self.trace_state("restore", consumed_tokens);
        } else if self.ssm_states.iter().any(|s| s.len() > 1) {
            // GDN model rewinding to a position we have no snapshot for: the
            // recurrent state cannot be reconstructed. Warn instead of failing
            // so non-speculative callers keep the (previous) best-effort
            // behavior, but generation quality may degrade past this point.
            eprintln!(
                "layer-wise: rewind_to({consumed_tokens}) without a GDN checkpoint (state at {}); recurrent state may be stale",
                self.ssm_pos
            );
        }
        self.kv_cache
            .rewind_to(position)
            .map_err(|e| ModelError::InferenceFailed(format!("{e:?}")))
    }

    fn forward(&mut self, tokens: &[Token], session: &mut Session) -> Result<Logits, ModelError> {
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
        let window = Self::forward_window_size_for("fwd");
        let mut logits = Vec::new();
        let mut offset = 0;
        while offset < tokens.len() {
            let end = (offset + window).min(tokens.len());
            logits = self
                .forward_window(&tokens[offset..end], start_pos + offset, false)?
                .pop()
                .unwrap_or_default();
            offset = end;
        }
        session.record_tokens(tokens.len());
        Ok(logits)
    }

    fn forward_many(
        &mut self,
        tokens: &[Token],
        session: &mut Session,
    ) -> Result<Vec<Logits>, ModelError> {
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
        // forward_many is the speculative-verification entry point: checkpoint
        // the recurrent state here so a rejected draft can rewind to start_pos.
        self.push_ssm_checkpoint(start_pos);
        let window = Self::forward_window_size_for("many");
        let mut all_logits = Vec::with_capacity(tokens.len());
        let mut offset = 0;
        while offset < tokens.len() {
            let end = (offset + window).min(tokens.len());
            all_logits.extend(self.forward_window(
                &tokens[offset..end],
                start_pos + offset,
                true,
            )?);
            offset = end;
        }
        session.record_tokens(tokens.len());
        Ok(all_logits)
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn context_size(&self) -> usize {
        self.config.context_size
    }
    fn layer_count(&self) -> usize {
        self.config.layer_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alibi_slopes_are_negative_and_decay() {
        let first = alibi_slope(0, 8);
        let second = alibi_slope(1, 8);
        assert!((first - -0.5).abs() < 1e-6, "{first}");
        assert!((second - -0.25).abs() < 1e-6, "{second}");
        assert!(second > first);
    }

    #[test]
    fn geglu_differs_from_swiglu() {
        let gate = [1.0_f32];
        let up = [2.0_f32];
        let mut swi = [0.0_f32];
        let mut ge = [0.0_f32];
        activate_ffn(&gate, &up, false, &mut swi).unwrap();
        activate_ffn(&gate, &up, true, &mut ge).unwrap();
        let silu = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        assert!((swi[0] - silu * 2.0).abs() < 1e-5);
        assert!((ge[0] - swi[0]).abs() > 1e-3);
    }

    #[test]
    fn sandwich_norm_uses_ffn_norm_not_post_attention() {
        let mut cfg = InferenceConfig::default();
        cfg.sandwich_norm = true;
        let mut layer = LayerWeights::default();
        layer.ffn_norm = vec![1.0, 2.0];
        layer.post_attention_norm = vec![3.0, 4.0];
        assert_eq!(select_ffn_norm(&cfg, &layer), &[1.0, 2.0]);
        cfg.sandwich_norm = false;
        assert_eq!(select_ffn_norm(&cfg, &layer), &[3.0, 4.0]);
    }

    #[test]
    fn mla_cache_stores_decompressed_heads() {
        let mut cfg = InferenceConfig::default();
        cfg.architecture = crate::inference::ModelArchitecture::DeepSeek;
        cfg.num_attention_heads = 8;
        cfg.num_key_value_heads = 1;
        cfg.hidden_size = 32;
        cfg.key_value_head_dim = 4;
        let (heads, dim) = kv_cache_geometry(&cfg);
        assert_eq!((heads, dim), (8, 4));
        cfg.architecture = crate::inference::ModelArchitecture::Llama;
        let (heads, dim) = kv_cache_geometry(&cfg);
        assert_eq!((heads, dim), (1, 4));
    }
}
