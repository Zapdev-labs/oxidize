#![allow(clippy::needless_range_loop, clippy::manual_checked_ops, dead_code)]

use crate::conversion::normalize_gguf_tensor_name;
use crate::flash_attention::flash_attention_decode_f32;
use crate::gguf::{GgufQuantizationType, MappedGgufFile};
use crate::inference::{
    InferenceConfig, MoeFfnWeights, WeightStorage, lookup_quantized_embedding,
    moe_ffn_forward_weights,
};
use crate::kv_cache::KvCache;
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::quantization::{dequantize_scalar, quantized_size};
use crate::tensor::{
    apply_rope_f32, apply_swiglu_f32, gemm_quantized_f32, gemv_f32, gemv_quantized_f32,
    rms_norm_f32,
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
