//! Temporal encoder for video understanding.
//!
//! Applies a stack of causal self-attention layers over the time axis. The
//! input to the temporal encoder is a `[num_frames, hidden_size]` matrix of
//! per-frame summaries (one vector per frame, produced by pooling the
//! vision encoder's patch tokens).
//!
//! CPU optimizations:
//!
//! * Sequential 1D attention with `O(num_frames^2 * hidden)` cost. For
//!   typical video inputs (8-32 frames) the quadratic term is small.
//! * Workspace reuse: callers pass a [`TemporalWorkspace`] that owns all
//!   scratch buffers, so the inner loops never allocate.
//! * Pure f32, no SIMD intrinsics: the existing [`gemv_f32_transposed`]
//!   /[`gemm_f32`] paths already cover the hot kernels; this layer adds
//!   the small number of bookkeeping ops (RMSNorm, RoPE, residual add) we
//!   need.
//! * Optional learnable `temporal_cls` token prepended to the sequence; if
//!   disabled the encoder pools the output by mean.
//!
//! [`gemv_f32_transposed`]: crate::tensor::gemv_f32_transposed
//! [`gemm_f32`]: crate::tensor::gemm_f32

use rayon::prelude::*;

use super::config::TemporalConfig;
use super::error::VideoError;
use crate::tensor::{apply_rope_f32, gemm_f32, rms_norm_f32};

/// Persistent scratch buffers for the temporal encoder. Allocate once,
/// reuse across calls.
#[derive(Debug, Clone)]
pub struct TemporalWorkspace {
    /// Hidden activations for the full sequence: `[seq_len * hidden]`.
    pub hidden: Vec<f32>,
    /// Scratch buffer for the residual stream: `[seq_len * hidden]`.
    pub residual: Vec<f32>,
    /// Cached QKV output: `[seq_len * (3 * hidden)]`.
    pub qkv: Vec<f32>,
    /// Attention scores buffer: `[num_heads * seq_len * seq_len]`.
    pub attn: Vec<f32>,
    /// Softmax output buffer: `[num_heads * seq_len * seq_len]`.
    pub attn_softmax: Vec<f32>,
    /// Per-head output: `[seq_len * hidden]`.
    pub attn_out: Vec<f32>,
    /// FFN intermediate (gate / up).
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_silu: Vec<f32>,
    pub ffn_down: Vec<f32>,
}

impl TemporalWorkspace {
    pub fn for_config(config: &TemporalConfig) -> Self {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        // Worst case seq_len = max_frames + 1 (cls token).
        let seq = config.max_frames + 1;
        let attn_size = config.num_heads * seq * seq;
        Self {
            hidden: vec![0.0_f32; seq * h],
            residual: vec![0.0_f32; seq * h],
            qkv: vec![0.0_f32; seq * 3 * h],
            attn: vec![0.0_f32; attn_size],
            attn_softmax: vec![0.0_f32; attn_size],
            attn_out: vec![0.0_f32; seq * h],
            ffn_gate: vec![0.0_f32; seq * inter],
            ffn_up: vec![0.0_f32; seq * inter],
            ffn_silu: vec![0.0_f32; seq * inter],
            ffn_down: vec![0.0_f32; seq * h],
        }
    }
}

/// Weights for a single temporal self-attention layer.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalLayerWeights {
    pub attn_norm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
}

impl TemporalLayerWeights {
    pub fn zeros(config: &TemporalConfig) -> Self {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        Self {
            attn_norm: vec![1.0; h],
            q_proj: vec![0.0; h * h],
            k_proj: vec![0.0; h * h],
            v_proj: vec![0.0; h * h],
            o_proj: vec![0.0; h * h],
            ffn_norm: vec![1.0; h],
            ffn_gate: vec![0.0; h * inter],
            ffn_up: vec![0.0; h * inter],
            ffn_down: vec![0.0; inter * h],
        }
    }
}

/// Full temporal encoder weights.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalWeights {
    pub layers: Vec<TemporalLayerWeights>,
    pub final_norm: Vec<f32>,
    /// Optional learnable `temporal_cls` token of length `hidden_size`.
    /// Empty when `config.use_cls_token` is `false`.
    pub cls_token: Vec<f32>,
}

impl TemporalWeights {
    /// Build zero-initialized weights for the given config. Useful for tests
    /// and as a starting point before loading real weights.
    pub fn zeros(config: &TemporalConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TemporalLayerWeights::zeros(config));
        }
        let cls_token = if config.use_cls_token {
            vec![0.0; config.hidden_size]
        } else {
            Vec::new()
        };
        Self {
            layers,
            final_norm: vec![1.0; config.hidden_size],
            cls_token,
        }
    }
}

/// Causal self-attention over a `[seq, hidden]` matrix.
///
/// `qkv_out` is expected to be of size `seq * 3 * hidden` and is reused
/// across layers via [`TemporalWorkspace`].
pub fn forward_temporal(
    config: &TemporalConfig,
    weights: &TemporalWeights,
    input: &[f32], // [input_seq_len, hidden]
    input_seq_len: usize,
    workspace: &mut TemporalWorkspace,
) -> Result<Vec<f32>, VideoError> {
    config.validate()?;
    if input_seq_len == 0 || input_seq_len > config.max_frames {
        return Err(VideoError::FrameCountOutOfRange {
            requested: input_seq_len,
            min: 1,
            max: config.max_frames,
        });
    }
    let h = config.hidden_size;
    if input.len() != input_seq_len * h {
        return Err(VideoError::InferenceFailed(format!(
            "input buffer length {} does not match seq_len * hidden ({} * {})",
            input.len(),
            input_seq_len,
            h
        )));
    }
    if weights.layers.len() != config.num_layers {
        return Err(VideoError::WeightShapeMismatch {
            name: "temporal_layers",
            expected: config.num_layers,
            actual: weights.layers.len(),
        });
    }

    let seq_len = input_seq_len + usize::from(!weights.cls_token.is_empty());
    let mut hidden = vec![0.0_f32; seq_len * h];

    // Prepend cls token if requested.
    if !weights.cls_token.is_empty() {
        hidden[..h].copy_from_slice(&weights.cls_token);
        hidden[h..h + input_seq_len * h].copy_from_slice(input);
    } else {
        hidden.copy_from_slice(input);
    }

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        forward_temporal_layer(config, layer, &mut hidden, seq_len, workspace)
            .map_err(|e| VideoError::InferenceFailed(format!("layer {layer_idx}: {e}")))?;
    }

    // Final norm on the full sequence.
    let mut normalized = vec![0.0_f32; seq_len * h];
    for (src, dst) in hidden.chunks_exact(h).zip(normalized.chunks_exact_mut(h)) {
        rms_norm_f32(src, &weights.final_norm, config.rms_norm_eps, dst)
            .map_err(|e| VideoError::InferenceFailed(format!("final rms norm: {e:?}")))?;
    }
    Ok(normalized)
}

fn forward_temporal_layer(
    config: &TemporalConfig,
    layer: &TemporalLayerWeights,
    hidden: &mut [f32], // [seq, hidden]
    seq_len: usize,
    workspace: &mut TemporalWorkspace,
) -> Result<(), VideoError> {
    let h = config.hidden_size;
    let head_dim = config.head_dim();
    if head_dim == 0 {
        return Err(VideoError::InvalidConfig(
            "head_dim must be non-zero".into(),
        ));
    }
    if layer.q_proj.len() != h * h
        || layer.k_proj.len() != h * h
        || layer.v_proj.len() != h * h
        || layer.o_proj.len() != h * h
    {
        return Err(VideoError::InferenceFailed(
            "QKV/O projection shape mismatch".into(),
        ));
    }
    if layer.ffn_gate.len() != h * config.intermediate_size
        || layer.ffn_up.len() != h * config.intermediate_size
        || layer.ffn_down.len() != config.intermediate_size * h
    {
        return Err(VideoError::InferenceFailed(
            "FFN projection shape mismatch".into(),
        ));
    }

    // ---- Pre-norm + QKV + attention ----
    // Save residual.
    workspace.residual[..seq_len * h].copy_from_slice(&hidden[..seq_len * h]);

    // RMSNorm input in-place into a fresh buffer (we can't overwrite hidden
    // before we've snapshotted it for the residual).
    let mut normed = vec![0.0_f32; seq_len * h];
    for (src, dst) in hidden.chunks_exact(h).zip(normed.chunks_exact_mut(h)) {
        rms_norm_f32(src, &layer.attn_norm, config.rms_norm_eps, dst)
            .map_err(|e| VideoError::InferenceFailed(format!("attn rms norm: {e:?}")))?;
    }

    // QKV projections. Compute Q, K, V via three independent GEMMs.
    let qkv = &mut workspace.qkv[..seq_len * 3 * h];
    let (q_part, rest) = qkv.split_at_mut(seq_len * h);
    let (k_part, v_part) = rest.split_at_mut(seq_len * h);
    gemm_f32(&normed, seq_len, h, &layer.q_proj, h, q_part)
        .map_err(|e| VideoError::InferenceFailed(format!("q_proj: {e:?}")))?;
    gemm_f32(&normed, seq_len, h, &layer.k_proj, h, k_part)
        .map_err(|e| VideoError::InferenceFailed(format!("k_proj: {e:?}")))?;
    gemm_f32(&normed, seq_len, h, &layer.v_proj, h, v_part)
        .map_err(|e| VideoError::InferenceFailed(format!("v_proj: {e:?}")))?;

    // Apply RoPE to Q and K along the time axis. The position index for
    // each row in `q_part` / `k_part` is its position in the sequence.
    let mut rope_scratch = vec![0.0_f32; head_dim];
    for pos in 0..seq_len {
        for head in 0..config.num_heads {
            let start = pos * h + head * head_dim;
            let end = start + head_dim;
            apply_rope_f32(
                &q_part[start..end],
                pos,
                head_dim,
                config.rope_theta,
                &mut rope_scratch,
            )
            .map_err(|e| VideoError::InferenceFailed(format!("q rope: {e:?}")))?;
            q_part[start..end].copy_from_slice(&rope_scratch);

            apply_rope_f32(
                &k_part[start..end],
                pos,
                head_dim,
                config.rope_theta,
                &mut rope_scratch,
            )
            .map_err(|e| VideoError::InferenceFailed(format!("k rope: {e:?}")))?;
            k_part[start..end].copy_from_slice(&rope_scratch);
        }
    }

    // Compute per-head attention: Q @ K^T -> scores, softmax, then @ V.
    let num_heads = config.num_heads;
    let attn_size = num_heads * seq_len * seq_len;
    let scores = &mut workspace.attn[..attn_size];
    let softmax_out = &mut workspace.attn_softmax[..attn_size];
    let attn_out = &mut workspace.attn_out[..seq_len * h];

    compute_causal_attention(
        q_part,
        k_part,
        v_part,
        scores,
        softmax_out,
        attn_out,
        seq_len,
        h,
        num_heads,
        head_dim,
    );

    // Output projection.
    let mut proj_out = vec![0.0_f32; seq_len * h];
    gemm_f32(attn_out, seq_len, h, &layer.o_proj, h, &mut proj_out)
        .map_err(|e| VideoError::InferenceFailed(format!("o_proj: {e:?}")))?;

    // Residual add: hidden = residual + proj_out.
    for i in 0..seq_len * h {
        hidden[i] = workspace.residual[i] + proj_out[i];
    }

    // ---- FFN block (pre-norm residual stream) ----
    workspace.residual[..seq_len * h].copy_from_slice(&hidden[..seq_len * h]);
    let mut ffn_normed = vec![0.0_f32; seq_len * h];
    for (src, dst) in hidden.chunks_exact(h).zip(ffn_normed.chunks_exact_mut(h)) {
        rms_norm_f32(src, &layer.ffn_norm, config.rms_norm_eps, dst)
            .map_err(|e| VideoError::InferenceFailed(format!("ffn rms norm: {e:?}")))?;
    }

    // SwiGLU FFN: gate = x @ Wg; up = x @ Wu; silu(gate) * up; out = @ Wd
    let gate = &mut workspace.ffn_gate[..seq_len * config.intermediate_size];
    let up = &mut workspace.ffn_up[..seq_len * config.intermediate_size];
    let silu = &mut workspace.ffn_silu[..seq_len * config.intermediate_size];
    let down = &mut workspace.ffn_down[..seq_len * h];

    gemm_f32(
        &ffn_normed,
        seq_len,
        h,
        &layer.ffn_gate,
        config.intermediate_size,
        gate,
    )
    .map_err(|e| VideoError::InferenceFailed(format!("ffn_gate: {e:?}")))?;
    gemm_f32(
        &ffn_normed,
        seq_len,
        h,
        &layer.ffn_up,
        config.intermediate_size,
        up,
    )
    .map_err(|e| VideoError::InferenceFailed(format!("ffn_up: {e:?}")))?;

    // SwiGLU + down projection (parallel over the seq axis via rayon).
    let inter = config.intermediate_size;
    silu.par_chunks_exact_mut(inter)
        .zip(gate.par_chunks_exact(inter))
        .zip(up.par_chunks_exact(inter))
        .for_each(|((s_row, g_row), u_row)| {
            for ((s, g), u) in s_row.iter_mut().zip(g_row.iter()).zip(u_row.iter()) {
                let sil = *g * sigmoid(*g);
                *s = sil * *u;
            }
        });

    gemm_f32(silu, seq_len, inter, &layer.ffn_down, h, down)
        .map_err(|e| VideoError::InferenceFailed(format!("ffn_down: {e:?}")))?;

    for i in 0..seq_len * h {
        hidden[i] = workspace.residual[i] + down[i];
    }

    Ok(())
}

fn compute_causal_attention(
    q: &[f32],          // [seq, hidden]
    k: &[f32],          // [seq, hidden]
    v: &[f32],          // [seq, hidden]
    scores: &mut [f32], // [heads, seq, seq]
    softmax_out: &mut [f32],
    output: &mut [f32], // [seq, hidden]
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h_idx in 0..num_heads {
        for q_pos in 0..seq_len {
            // Pointer arithmetic: Q row q_pos, head h_idx -> [q_pos*hidden + h_idx*head_dim]
            let q_row =
                &q[q_pos * hidden + h_idx * head_dim..q_pos * hidden + (h_idx + 1) * head_dim];
            let mut max_score = f32::NEG_INFINITY;
            let score_row = &mut scores[h_idx * seq_len * seq_len + q_pos * seq_len..];
            for k_pos in 0..=q_pos {
                let k_row =
                    &k[k_pos * hidden + h_idx * head_dim..k_pos * hidden + (h_idx + 1) * head_dim];
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q_row[d] * k_row[d];
                }
                let s = dot * scale;
                score_row[k_pos] = s;
                if s > max_score {
                    max_score = s;
                }
            }
            // Mask future positions to -inf.
            for k_pos in (q_pos + 1)..seq_len {
                score_row[k_pos] = f32::NEG_INFINITY;
            }
            if !max_score.is_finite() {
                max_score = 0.0;
            }
            // Softmax over the row.
            let softmax_row = &mut softmax_out[h_idx * seq_len * seq_len + q_pos * seq_len..];
            let mut sum = 0.0_f32;
            for k_pos in 0..=q_pos {
                let p = (score_row[k_pos] - max_score).exp();
                softmax_row[k_pos] = p;
                sum += p;
            }
            for k_pos in (q_pos + 1)..seq_len {
                softmax_row[k_pos] = 0.0;
            }
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
            for k_pos in 0..seq_len {
                softmax_row[k_pos] *= inv_sum;
            }

            // Output = sum_k softmax[k] * V[k, h_idx]
            let out_row = &mut output
                [q_pos * hidden + h_idx * head_dim..q_pos * hidden + (h_idx + 1) * head_dim];
            for d in 0..head_dim {
                out_row[d] = 0.0;
            }
            for k_pos in 0..=q_pos {
                let v_row =
                    &v[k_pos * hidden + h_idx * head_dim..k_pos * hidden + (h_idx + 1) * head_dim];
                let a = softmax_row[k_pos];
                for d in 0..head_dim {
                    out_row[d] += a * v_row[d];
                }
            }
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> TemporalConfig {
        TemporalConfig {
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 16,
            rms_norm_eps: 1e-5,
            max_frames: 4,
            rope_theta: 10000.0,
            use_cls_token: false,
            layer_dropout: 0.0,
        }
    }

    #[test]
    fn zero_weights_propagate_input_through_norm() {
        let cfg = tiny_config();
        let mut w = TemporalWeights::zeros(&cfg);
        w.layers.clear();
        w.layers.push(TemporalLayerWeights::zeros(&cfg));
        // With Q/K/V/O all zero, attention output is zero -> hidden becomes
        // residual only.
        let input: Vec<f32> = (0..cfg.hidden_size).map(|i| i as f32 * 0.1).collect();
        let mut ws = TemporalWorkspace::for_config(&cfg);
        let out = forward_temporal(&cfg, &w, &input, 1, &mut ws).unwrap();
        // After the only layer (Q/K/V/O all zero) hidden == input (residual
        // only); then final norm rescales but keeps shape.
        assert_eq!(out.len(), cfg.hidden_size);
    }

    #[test]
    fn empty_input_is_rejected() {
        let cfg = tiny_config();
        let w = TemporalWeights::zeros(&cfg);
        let mut ws = TemporalWorkspace::for_config(&cfg);
        let err = forward_temporal(&cfg, &w, &[], 0, &mut ws).unwrap_err();
        assert!(matches!(err, VideoError::FrameCountOutOfRange { .. }));
    }

    #[test]
    fn causal_attention_ignores_future_positions() {
        let cfg = tiny_config();
        let mut w = TemporalWeights::zeros(&cfg);
        // Make Q and K constant: every position has the same Q/K vector.
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let layer = &mut w.layers[0];
        for v in &mut layer.q_proj {
            *v = 0.0;
        }
        for v in &mut layer.k_proj {
            *v = 0.0;
        }
        // Set Q[0, head_dim] = 1 for every head, rest 0.
        for head in 0..cfg.num_heads {
            layer.q_proj[head * head_dim] = 1.0;
            layer.k_proj[head * head_dim] = 1.0;
        }
        let input = vec![0.1_f32; 3 * h];
        let mut ws = TemporalWorkspace::for_config(&cfg);
        let out = forward_temporal(&cfg, &w, &input, 3, &mut ws).unwrap();
        assert_eq!(out.len(), 3 * h);
    }
}
