//! DiffusionGemma (`diffusion-gemma`) block-diffusion inference on the OXK CPU kernels.
//!
//! DiffusionGemma is a Gemma-4 26B-A4B Mixture-of-Experts checkpoint trained as a discrete
//! **block-diffusion** denoiser rather than an autoregressive decoder. It generates a fixed
//! `CANVAS` of tokens in parallel by iteratively denoising them over `STEPS` forward passes,
//! attending **bidirectionally** within the canvas (`attention.causal = false`).
//!
//! This module is a self-contained, faithful port of the reference forward graph
//! (llama.cpp `src/models/diffusion-gemma.cpp`, PR #24427) implemented on top of oxidize's
//! quantized GEMV/GEMM kernels (the OXK kernels when built with `--features oxk` and run with
//! `OXIDIZE_GEMV=oxk`). Per-layer math mirrors Gemma-4:
//!   * QK-norm + scale-less V-norm, dual head dims (swa head_dim 256 / full head_dim 512),
//!     V = K on the global (full-attention) layers (no `attn_v`), NEOX rope with proportional
//!     `rope_freqs` on full layers, attention scale 1.0 (`f_attn_scale`).
//!   * Dual FFN per layer: a dense shared MLP (`ffn_*`) plus a routed 128-expert top-8 MoE
//!     (`ffn_*_exps`), summed; GELU-gated; sandwich RMS norms; per-layer output scalar.
//!   * Self-conditioning MLP feeding back the previous step's soft prediction (decoder phase).
//!   * Final logit softcapping (30.0); output head tied to `token_embd`.
//!
//! The denoise loop reproduces the reference sampler (linear temperature schedule,
//! EntropyBoundSampler accept, StableAndConfident stop).

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity,
    dead_code
)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use crate::gguf::{GgufQuantizationType, GgufTensorInfo, load_mapped_gguf};
use crate::quantization::QuantizationError;
use crate::tensor::{
    apply_geglu_inplace_f32, gemm_quantized_f32, gemv_f32, gemv_quantized_experts_f32,
    gemv_quantized_f32, rms_norm_f32, softmax_f32, GemmError, GemvError, RmsNormError,
    SoftmaxError,
};
use memmap2::Mmap;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Errors from DiffusionGemma load, forward, and denoise sampling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffusionGemmaError {
    Gemv(GemvError),
    Gemm(GemmError),
    RmsNorm(RmsNormError),
    Softmax(SoftmaxError),
    Quantization(QuantizationError),
    UnsupportedQuant(String),
}

impl std::fmt::Display for DiffusionGemmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gemv(e) => write!(f, "gemv: {e:?}"),
            Self::Gemm(e) => write!(f, "gemm: {e:?}"),
            Self::RmsNorm(e) => write!(f, "rms_norm: {e:?}"),
            Self::Softmax(e) => write!(f, "softmax: {e:?}"),
            Self::Quantization(e) => write!(f, "quantization: {e:?}"),
            Self::UnsupportedQuant(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for DiffusionGemmaError {}

impl From<GemvError> for DiffusionGemmaError {
    fn from(value: GemvError) -> Self {
        Self::Gemv(value)
    }
}
impl From<GemmError> for DiffusionGemmaError {
    fn from(value: GemmError) -> Self {
        Self::Gemm(value)
    }
}
impl From<RmsNormError> for DiffusionGemmaError {
    fn from(value: RmsNormError) -> Self {
        Self::RmsNorm(value)
    }
}
impl From<SoftmaxError> for DiffusionGemmaError {
    fn from(value: SoftmaxError) -> Self {
        Self::Softmax(value)
    }
}
impl From<QuantizationError> for DiffusionGemmaError {
    fn from(value: QuantizationError) -> Self {
        Self::Quantization(value)
    }
}

type DiffusionResult<T> = Result<T, DiffusionGemmaError>;

fn f32_cmp(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

// ---- architecture constants (from the GGUF metadata) ----
const N_LAYER: usize = 30;
const N_EMBD: usize = 2816;
const N_HEAD: usize = 16;
const N_VOCAB: usize = 262144;
const EPS: f32 = 1e-6;
const ROPE_FULL: f32 = 1_000_000.0;
const ROPE_SWA: f32 = 10_000.0;
const N_EXPERT: usize = 128;
const N_USED: usize = 8;
const EXPERT_FF: usize = 704;
const DENSE_FF: usize = 2112;
const SOFTCAP: f32 = 30.0;
pub const CANVAS: usize = 256;
pub const STEPS: usize = 48;
pub const MASK_TOKEN: u32 = 4;

// per-layer geometry: every 6th layer (il % 6 == 5) is a global full-attention layer.
fn is_swa(il: usize) -> bool {
    il % 6 != 5
}
fn head_dim(il: usize) -> usize {
    if is_swa(il) { 256 } else { 512 }
}
fn n_head_kv(il: usize) -> usize {
    if is_swa(il) { 8 } else { 2 }
}
fn rope_base(il: usize) -> f32 {
    if is_swa(il) { ROPE_SWA } else { ROPE_FULL }
}

/// True when OXK's quantized GEMV/GEMM kernels can consume this type directly.
fn quant_supported(q: GgufQuantizationType) -> bool {
    matches!(
        q,
        GgufQuantizationType::Q8_0
            | GgufQuantizationType::Q4_K_S
            | GgufQuantizationType::Q4_K_M
            | GgufQuantizationType::Q6_K
            | GgufQuantizationType::Q2_K
    )
}

/// A quantized weight matrix. `rows` outputs of `cols` inputs each. Normally an mmap slice; for
/// types OXK's kernels don't support (e.g. Q5_0) it is requantized to Q8_0 and held in `owned`
/// (Q8_0 is higher precision than Q5_0, so the requant is near-lossless and stays on the fast
/// SIMD path — ~4x less RAM and ~10x faster than a scalar f32 fallback).
#[derive(Clone)]
struct QW {
    q: GgufQuantizationType,
    off: usize,
    len: usize,
    rows: usize,
    cols: usize,
    owned: Option<Vec<u8>>,
}

/// A routed-experts tensor: `n_expert` matrices of `rows x cols` each, contiguous.
#[derive(Clone)]
struct EW {
    q: GgufQuantizationType,
    off: usize,
    len: usize,
    rows: usize,
    cols: usize,
    owned: Option<Vec<u8>>,
}

/// Requantize an OXK-unsupported buffer to Q8_0 bytes (via f32). `n` = element count.
fn requant_to_q8_0(
    q: GgufQuantizationType,
    bytes: &[u8],
    n: usize,
) -> DiffusionResult<Vec<u8>> {
    let f = dequant_any(q, bytes, n)?;
    let mut out = vec![0u8; (n / 32) * 34];
    crate::quantization::quantize_q8_0_scalar(&f, &mut out)?;
    Ok(out)
}

struct Layer {
    attn_norm: Vec<f32>,
    attn_q: QW,
    attn_q_norm: Vec<f32>,
    attn_k: QW,
    attn_k_norm: Vec<f32>,
    attn_v: Option<QW>, // absent on full layers (V = K)
    attn_output: QW,
    post_attention_norm: Vec<f32>,
    // dense shared MLP
    ffn_norm: Vec<f32>,
    ffn_gate: QW,
    ffn_up: QW,
    ffn_down: QW,
    post_ffw_norm_1: Vec<f32>,
    // routed MoE
    pre_ffw_norm_2: Vec<f32>,
    ffn_gate_inp: Vec<f32>,    // [N_EXPERT, N_EMBD] f32 router
    ffn_gate_inp_s: Vec<f32>,  // [N_EMBD] per-channel router-input scale
    ffn_gate_up_exps: EW,      // fused [2*EXPERT_FF, N_EMBD] per expert
    ffn_down_exps: EW,         // [N_EMBD, EXPERT_FF] per expert
    ffn_down_exps_s: Vec<f32>, // [N_EXPERT] per-expert output scale
    post_ffw_norm_2: Vec<f32>,
    post_ffw_norm: Vec<f32>,
    out_scale: f32, // layer_output_scale
}

pub struct DiffusionGemma {
    mmap: Arc<Mmap>,
    layers: Vec<Layer>,
    token_embd: QW, // [N_VOCAB, N_EMBD], also the tied output head
    output_norm: Vec<f32>,
    self_cond_norm: Vec<f32>,
    self_cond_gate: QW,
    self_cond_up: QW,
    self_cond_down: QW,   // Q5_0 -> auto-dequantized in QW.deq
    rope_freqs: Vec<f32>, // [256] proportional-rope factors for full layers
}

fn bytes_for(q: GgufQuantizationType, rows: usize, cols: usize) -> usize {
    let (bw, bs) = block_info(q);
    rows * (cols / bw) * bs
}

fn block_info(q: GgufQuantizationType) -> (usize, usize) {
    match q {
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => (256, 144),
        GgufQuantizationType::Q5_K_S | GgufQuantizationType::Q5_K_M => (256, 176),
        GgufQuantizationType::Q6_K => (256, 210),
        GgufQuantizationType::Q8_0 => (32, 34),
        GgufQuantizationType::Q5_0 => (32, 22),
        GgufQuantizationType::Q4_0 => (32, 18),
        GgufQuantizationType::F32 => (1, 4),
        GgufQuantizationType::F16 => (1, 2),
        _ => (1, 4),
    }
}

/// Dequantize a Q5_0 buffer to f32 (block = 32 values: f16 scale, u32 high-bits, 16 nibble bytes).
fn dequant_q5_0(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    let nblocks = n / 32;
    for b in 0..nblocks {
        let base = b * 22;
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let qh = u32::from_le_bytes([
            data[base + 2],
            data[base + 3],
            data[base + 4],
            data[base + 5],
        ]);
        let qs = &data[base + 6..base + 22];
        for i in 0..16 {
            let h0 = ((qh >> i) & 1) as u8;
            let h1 = ((qh >> (i + 16)) & 1) as u8;
            let lo = (qs[i] & 0x0F) | (h0 << 4);
            let hi = (qs[i] >> 4) | (h1 << 4);
            out[b * 32 + i] = (lo as i32 - 16) as f32 * d;
            out[b * 32 + 16 + i] = (hi as i32 - 16) as f32 * d;
        }
    }
    out
}

/// Dequantize an OXK-unsupported weight type to f32 (currently Q5_0; F16/F32 pass-through).
fn dequant_any(q: GgufQuantizationType, bytes: &[u8], n: usize) -> DiffusionResult<Vec<f32>> {
    match q {
        GgufQuantizationType::Q5_0 => Ok(dequant_q5_0(bytes, n)),
        GgufQuantizationType::F32 => {
            let mut v = vec![0.0_f32; n];
            for i in 0..n {
                v[i] = f32::from_le_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ]);
            }
            Ok(v)
        }
        GgufQuantizationType::F16 => Ok((0..n)
            .map(|i| f16_to_f32(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])))
            .collect()),
        other => Err(DiffusionGemmaError::UnsupportedQuant(format!(
            "dequant_any: unsupported quant {other:?}"
        ))),
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            (mant as f32) * 2f32.powi(-24)
        }
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -val } else { val }
}

impl DiffusionGemma {
    fn bytes<'a>(&'a self, w: &'a QW) -> &'a [u8] {
        match &w.owned {
            Some(b) => b,
            None => &self.mmap[w.off..w.off + w.len],
        }
    }
    fn ebytes<'a>(&'a self, w: &'a EW) -> &'a [u8] {
        match &w.owned {
            Some(b) => b,
            None => &self.mmap[w.off..w.off + w.len],
        }
    }

    /// Batched matmul `outputs[batch, rows] = W[rows, cols] @ inputs[batch, cols]` on OXK GEMM.
    fn gemm_qw(
        &self,
        w: &QW,
        rows: usize,
        cols: usize,
        inputs: &[f32],
        outputs: &mut [f32],
        batch: usize,
    ) -> DiffusionResult<()> {
        gemm_quantized_f32(w.q, self.bytes(w), rows, cols, inputs, outputs, batch)?;
        Ok(())
    }

    /// Single-vector matmul `output[rows] = W[rows, cols] @ input[cols]`.
    fn gemv_qw(
        &self,
        w: &QW,
        rows: usize,
        cols: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> DiffusionResult<()> {
        gemv_quantized_f32(w.q, self.bytes(w), rows, cols, input, output)?;
        Ok(())
    }

    /// Selected-experts matmul. `output[n_sel, rows]`; each expert reads `inputs[slot*stride..]`
    /// (or shared `inputs` when `stride == 0`).
    fn experts_ew(
        &self,
        w: &EW,
        sel: &[usize],
        rows: usize,
        cols: usize,
        inputs: &[f32],
        stride: usize,
        output: &mut [f32],
    ) -> DiffusionResult<()> {
        gemv_quantized_experts_f32(
            w.q,
            self.ebytes(w),
            N_EXPERT,
            sel,
            rows,
            cols,
            inputs,
            stride,
            output,
        )?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<DiffusionGemma, DiffusionGemmaError> {
        let mapped = load_mapped_gguf(path).map_err(|e| {
            DiffusionGemmaError::UnsupportedQuant(format!("gguf: {e:?}"))
        })?;
        let mmap = mapped.mmap();
        let infos = mapped.mapped_tensor_infos();
        let mut by_name: HashMap<String, GgufTensorInfo> = HashMap::new();
        for t in infos {
            by_name.insert(t.name.clone(), t);
        }

        let qw = |name: &str| -> DiffusionResult<QW> {
            let t = by_name.get(name).ok_or_else(|| {
                DiffusionGemmaError::UnsupportedQuant(format!("missing tensor {name}"))
            })?;
            let q = GgufQuantizationType::from_ggml_type(t.ggml_type);
            // 2D linear weight: dims = [cols(in), rows(out)]
            let cols = t.dimensions[0] as usize;
            let rows = t.dimensions[1] as usize;
            let len = bytes_for(q, rows, cols);
            let off = t.absolute_offset as usize;
            if quant_supported(q) {
                Ok(QW {
                    q,
                    off,
                    len,
                    rows,
                    cols,
                    owned: None,
                })
            } else {
                let owned = requant_to_q8_0(q, &mmap[off..off + len], rows * cols)?;
                Ok(QW {
                    q: GgufQuantizationType::Q8_0,
                    off,
                    len: owned.len(),
                    rows,
                    cols,
                    owned: Some(owned),
                })
            }
        };
        let ew = |name: &str| -> DiffusionResult<EW> {
            let t = by_name.get(name).ok_or_else(|| {
                DiffusionGemmaError::UnsupportedQuant(format!("missing tensor {name}"))
            })?;
            let q = GgufQuantizationType::from_ggml_type(t.ggml_type);
            // experts dims = [cols(in), rows(out), n_expert]
            let cols = t.dimensions[0] as usize;
            let rows = t.dimensions[1] as usize;
            let len = bytes_for(q, rows, cols) * N_EXPERT;
            let off = t.absolute_offset as usize;
            if quant_supported(q) {
                Ok(EW {
                    q,
                    off,
                    len,
                    rows,
                    cols,
                    owned: None,
                })
            } else {
                let owned = requant_to_q8_0(q, &mmap[off..off + len], N_EXPERT * rows * cols)?;
                Ok(EW {
                    q: GgufQuantizationType::Q8_0,
                    off,
                    len: owned.len(),
                    rows,
                    cols,
                    owned: Some(owned),
                })
            }
        };
        let f32v = |name: &str| -> DiffusionResult<Vec<f32>> {
            let t = by_name.get(name).ok_or_else(|| {
                DiffusionGemmaError::UnsupportedQuant(format!("missing tensor {name}"))
            })?;
            let n: usize = t.dimensions.iter().map(|&d| d as usize).product();
            let off = t.absolute_offset as usize;
            let q = GgufQuantizationType::from_ggml_type(t.ggml_type);
            match q {
                GgufQuantizationType::F32 => {
                    let mut v = vec![0.0_f32; n];
                    let raw = &mmap[off..off + n * 4];
                    for i in 0..n {
                        v[i] = f32::from_le_bytes([
                            raw[i * 4],
                            raw[i * 4 + 1],
                            raw[i * 4 + 2],
                            raw[i * 4 + 3],
                        ]);
                    }
                    Ok(v)
                }
                GgufQuantizationType::F16 => {
                    let mut v = vec![0.0_f32; n];
                    let raw = &mmap[off..off + n * 2];
                    for i in 0..n {
                        v[i] = f16_to_f32(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
                    }
                    Ok(v)
                }
                other => Err(DiffusionGemmaError::UnsupportedQuant(format!(
                    "f32v: unexpected quant {other:?} for {name}"
                ))),
            }
        };

        let mut layers = Vec::with_capacity(N_LAYER);
        for il in 0..N_LAYER {
            let p = |s: &str| format!("blk.{il}.{s}");
            let attn_v = if is_swa(il) {
                Some(qw(&p("attn_v.weight"))?)
            } else {
                None
            };
            // per-expert output scale ffn_down_exps.scale [N_EXPERT]; router scale ffn_gate_inp.scale
            let ds = f32v(&p("ffn_down_exps.scale")).unwrap_or_else(|_| vec![1.0; N_EXPERT]);
            let gis = f32v(&p("ffn_gate_inp.scale")).unwrap_or_else(|_| vec![1.0; N_EMBD]);
            let out_scale = f32v(&p("layer_output_scale.weight"))
                .ok()
                .and_then(|v| v.first().copied())
                .unwrap_or(1.0);
            layers.push(Layer {
                attn_norm: f32v(&p("attn_norm.weight"))?,
                attn_q: qw(&p("attn_q.weight"))?,
                attn_q_norm: f32v(&p("attn_q_norm.weight"))?,
                attn_k: qw(&p("attn_k.weight"))?,
                attn_k_norm: f32v(&p("attn_k_norm.weight"))?,
                attn_v,
                attn_output: qw(&p("attn_output.weight"))?,
                post_attention_norm: f32v(&p("post_attention_norm.weight"))?,
                ffn_norm: f32v(&p("ffn_norm.weight"))?,
                ffn_gate: qw(&p("ffn_gate.weight"))?,
                ffn_up: qw(&p("ffn_up.weight"))?,
                ffn_down: qw(&p("ffn_down.weight"))?,
                post_ffw_norm_1: f32v(&p("post_ffw_norm_1.weight"))?,
                pre_ffw_norm_2: f32v(&p("pre_ffw_norm_2.weight"))?,
                ffn_gate_inp: f32v(&p("ffn_gate_inp.weight"))?,
                ffn_gate_inp_s: gis,
                ffn_gate_up_exps: ew(&p("ffn_gate_up_exps.weight"))?,
                ffn_down_exps: ew(&p("ffn_down_exps.weight"))?,
                ffn_down_exps_s: ds,
                post_ffw_norm_2: f32v(&p("post_ffw_norm_2.weight"))?,
                post_ffw_norm: f32v(&p("post_ffw_norm.weight"))?,
                out_scale,
            });
        }

        Ok(DiffusionGemma {
            token_embd: qw("token_embd.weight")?,
            output_norm: f32v("output_norm.weight")?,
            self_cond_norm: f32v("self_cond_pre_norm.weight")?,
            self_cond_gate: qw("self_cond_gate.weight")?,
            self_cond_up: qw("self_cond_up.weight")?,
            self_cond_down: qw("self_cond_down.weight")?, // Q5_0 auto-dequantized
            rope_freqs: f32v("rope_freqs.weight").unwrap_or_else(|_| vec![1.0; 256]),
            mmap,
            layers,
        })
    }

    /// Embedding lookup for one token id into `out[..N_EMBD]`.
    fn embed(&self, token: u32, out: &mut [f32]) {
        crate::inference::lookup_quantized_embedding(
            N_EMBD,
            self.token_embd.q,
            self.bytes(&self.token_embd),
            (token as usize).min(N_VOCAB - 1),
            out,
        );
    }

    /// NEOX rope on the first `rot` dims of a head vector, with optional proportional factors.
    fn rope(vec: &mut [f32], pos: usize, rot: usize, base: f32, freqs: Option<&[f32]>) {
        let half = rot / 2;
        for i in 0..half {
            let mut theta = pos as f32 * base.powf(-2.0 * i as f32 / rot as f32);
            if let Some(f) = freqs {
                theta /= f[i];
            }
            let (s, c) = theta.sin_cos();
            let x0 = vec[i];
            let x1 = vec[i + half];
            vec[i] = x0 * c - x1 * s;
            vec[i + half] = x0 * s + x1 * c;
        }
    }

    /// Bidirectional forward over `tokens` at `positions`. `inpL` carries the prepared input
    /// embeddings (decoder: self-conditioned scale-less-normed; encoder: scaled). Returns the
    /// output-normed hidden states `[n_tok * N_EMBD]` (caller applies the tied head).
    fn forward_inner(
        &self,
        inpl: &mut [f32],
        positions: &[usize],
        prefix: usize,
    ) -> DiffusionResult<Vec<f32>> {
        let nt = positions.len();
        let ones = vec![1.0_f32; 512.max(N_EMBD)];
        let mut x = inpl.to_vec();
        let mut normed = vec![0.0_f32; nt * N_EMBD];

        for il in 0..N_LAYER {
            let l = &self.layers[il];
            let hd = head_dim(il);
            let kvh = n_head_kv(il);
            let qdim = N_HEAD * hd;
            let kvdim = kvh * hd;
            let group = N_HEAD / kvh;
            let rot = hd; // full rope over head_dim
            let freqs = if is_swa(il) {
                None
            } else {
                Some(&self.rope_freqs[..hd / 2])
            };

            // attn norm
            for i in 0..nt {
                rms_norm_f32(
                    &x[i * N_EMBD..(i + 1) * N_EMBD],
                    &l.attn_norm,
                    EPS,
                    &mut normed[i * N_EMBD..(i + 1) * N_EMBD],
                )?;
            }
            // Q/K(/V) projections (batched)
            let mut q = vec![0.0_f32; nt * qdim];
            let mut k = vec![0.0_f32; nt * kvdim];
            let mut v = vec![0.0_f32; nt * kvdim];
            self.gemm_qw(&l.attn_q, qdim, N_EMBD, &normed, &mut q, nt)?;
            self.gemm_qw(&l.attn_k, kvdim, N_EMBD, &normed, &mut k, nt)?;
            if let Some(wv) = &l.attn_v {
                self.gemm_qw(wv, kvdim, N_EMBD, &normed, &mut v, nt)?;
            } else {
                v.copy_from_slice(&k); // full layers: V = K (raw projection, before norms)
            }

            // per-head QK norm + rope; scale-less V norm (no rope)
            let mut tmp = vec![0.0_f32; hd];
            for i in 0..nt {
                let pos = positions[i];
                for h in 0..N_HEAD {
                    let qs = &mut q[i * qdim + h * hd..i * qdim + h * hd + hd];
                    rms_norm_f32(qs, &l.attn_q_norm, EPS, &mut tmp)?;
                    qs.copy_from_slice(&tmp);
                    Self::rope(qs, pos, rot, rope_base(il), freqs);
                }
                for h in 0..kvh {
                    let ks = &mut k[i * kvdim + h * hd..i * kvdim + h * hd + hd];
                    rms_norm_f32(ks, &l.attn_k_norm, EPS, &mut tmp)?;
                    ks.copy_from_slice(&tmp);
                    Self::rope(ks, pos, rot, rope_base(il), freqs);
                    let vs = &mut v[i * kvdim + h * hd..i * kvdim + h * hd + hd];
                    rms_norm_f32(vs, &ones[..hd], EPS, &mut tmp)?; // scale-less
                    vs.copy_from_slice(&tmp);
                }
            }

            // bidirectional attention (scale = 1.0), parallelized over query tokens.
            // prompt-prefix queries (i < prefix) are causal among the prefix; canvas queries
            // (i >= prefix) attend everything (bidirectional + full cross).
            let mut attn = vec![0.0_f32; nt * qdim];
            let attn_err: Mutex<Option<DiffusionGemmaError>> = Mutex::new(None);
            attn.par_chunks_mut(qdim).enumerate().for_each(|(i, arow)| {
                if matches!(attn_err.lock(), Ok(g) if g.is_some()) {
                    return;
                }
                let causal = i < prefix;
                let lim = if causal { i + 1 } else { nt };
                let mut scores = vec![0.0_f32; lim];
                let mut probs = vec![0.0_f32; lim];
                for h in 0..N_HEAD {
                    let kvhh = h / group;
                    let qv = &q[i * qdim + h * hd..i * qdim + h * hd + hd];
                    for j in 0..lim {
                        let kv = &k[j * kvdim + kvhh * hd..j * kvdim + kvhh * hd + hd];
                        let mut d = 0.0_f32;
                        for t in 0..hd {
                            d += qv[t] * kv[t];
                        }
                        scores[j] = d;
                    }
                    if let Err(e) = softmax_f32(&scores, &mut probs) {
                        if let Ok(mut guard) = attn_err.lock() {
                            *guard = Some(DiffusionGemmaError::Softmax(e));
                        }
                        return;
                    }
                    let out = &mut arow[h * hd..h * hd + hd];
                    for j in 0..lim {
                        let vv = &v[j * kvdim + kvhh * hd..j * kvdim + kvhh * hd + hd];
                        let p = probs[j];
                        for t in 0..hd {
                            out[t] += p * vv[t];
                        }
                    }
                }
            });
            if let Ok(Some(e)) = attn_err.into_inner() {
                return Err(e);
            }

            // output projection
            let mut attn_proj = vec![0.0_f32; nt * N_EMBD];
            self.gemm_qw(&l.attn_output, N_EMBD, qdim, &attn, &mut attn_proj, nt)?;

            // attn_out = post_attention_norm(attn_proj) + x
            let mut attn_out = vec![0.0_f32; nt * N_EMBD];
            for i in 0..nt {
                let r = i * N_EMBD..(i + 1) * N_EMBD;
                rms_norm_f32(
                    &attn_proj[r.clone()],
                    &l.post_attention_norm,
                    EPS,
                    &mut attn_out[r.clone()],
                )?;
                for t in 0..N_EMBD {
                    attn_out[i * N_EMBD + t] += x[i * N_EMBD + t];
                }
            }

            // ---- dual FFN: dense shared MLP + routed MoE, summed ----
            let mut ffn_comb = vec![0.0_f32; nt * N_EMBD];
            self.dense_ffn(l, &attn_out, &mut ffn_comb, nt)?;
            let mut moe = vec![0.0_f32; nt * N_EMBD];
            self.moe_ffn(l, &attn_out, &mut moe, nt)?;
            for t in 0..nt * N_EMBD {
                ffn_comb[t] += moe[t];
            }

            // cur = post_ffw_norm(ffn_comb); cur += attn_out; cur *= out_scale
            for i in 0..nt {
                let r = i * N_EMBD..(i + 1) * N_EMBD;
                let mut nrm = vec![0.0_f32; N_EMBD];
                rms_norm_f32(&ffn_comb[r.clone()], &l.post_ffw_norm, EPS, &mut nrm)?;
                for t in 0..N_EMBD {
                    x[i * N_EMBD + t] = (nrm[t] + attn_out[i * N_EMBD + t]) * l.out_scale;
                }
            }
        }

        // final norm
        let mut outv = vec![0.0_f32; nt * N_EMBD];
        for i in 0..nt {
            rms_norm_f32(
                &x[i * N_EMBD..(i + 1) * N_EMBD],
                &self.output_norm,
                EPS,
                &mut outv[i * N_EMBD..(i + 1) * N_EMBD],
            )?;
        }
        Ok(outv)
    }

    fn dense_ffn(
        &self,
        l: &Layer,
        src: &[f32],
        out: &mut [f32],
        nt: usize,
    ) -> DiffusionResult<()> {
        let mut nrm = vec![0.0_f32; nt * N_EMBD];
        for i in 0..nt {
            rms_norm_f32(
                &src[i * N_EMBD..(i + 1) * N_EMBD],
                &l.ffn_norm,
                EPS,
                &mut nrm[i * N_EMBD..(i + 1) * N_EMBD],
            )?;
        }
        let mut gate = vec![0.0_f32; nt * DENSE_FF];
        let mut up = vec![0.0_f32; nt * DENSE_FF];
        self.gemm_qw(&l.ffn_gate, DENSE_FF, N_EMBD, &nrm, &mut gate, nt)?;
        self.gemm_qw(&l.ffn_up, DENSE_FF, N_EMBD, &nrm, &mut up, nt)?;
        apply_geglu_inplace_f32(&mut gate, &up);
        let mut down = vec![0.0_f32; nt * N_EMBD];
        self.gemm_qw(&l.ffn_down, N_EMBD, DENSE_FF, &gate, &mut down, nt)?;
        // post_ffw_norm_1
        for i in 0..nt {
            rms_norm_f32(
                &down[i * N_EMBD..(i + 1) * N_EMBD],
                &l.post_ffw_norm_1,
                EPS,
                &mut out[i * N_EMBD..(i + 1) * N_EMBD],
            )?;
        }
        Ok(())
    }

    /// Routed MoE for the whole token batch, batched mul_mat_id-style: all `nt*N_USED`
    /// (token, expert) pairs flow through ONE gate_up experts GEMV and ONE down experts GEMV,
    /// giving a single level of rayon parallelism over the full output (no per-token nesting).
    fn moe_ffn(
        &self,
        l: &Layer,
        src: &[f32],
        out: &mut [f32],
        nt: usize,
    ) -> DiffusionResult<()> {
        let ones = vec![1.0_f32; N_EMBD];
        let inv = 1.0 / (N_EMBD as f32).sqrt();
        let ns = nt * N_USED;
        let gu_rows = 2 * EXPERT_FF;

        // Per-token (cheap, scalar): router selection, combine weights, and the per-(token,expert)
        // expert input (pre_ffw_norm_2(attn_out), repeated across the token's N_USED slots).
        let mut sel_flat = vec![0usize; ns];
        let mut wts = vec![0.0_f32; ns];
        let mut ein_rep = vec![0.0_f32; ns * N_EMBD];
        for i in 0..nt {
            let sr = &src[i * N_EMBD..(i + 1) * N_EMBD];
            let mut rin = vec![0.0_f32; N_EMBD];
            rms_norm_f32(sr, &ones, EPS, &mut rin)?;
            for t in 0..N_EMBD {
                rin[t] = rin[t] * inv * l.ffn_gate_inp_s[t];
            }
            let mut logits = vec![0.0_f32; N_EXPERT];
            gemv_f32(&l.ffn_gate_inp, N_EXPERT, N_EMBD, &rin, &mut logits)?;
            let mut probs = vec![0.0_f32; N_EXPERT];
            softmax_f32(&logits, &mut probs)?;
            let mut idx: Vec<usize> = (0..N_EXPERT).collect();
            idx.sort_by(|&a, &b| f32_cmp(probs[b], probs[a]));
            let wsum: f32 = idx[..N_USED].iter().map(|&e| probs[e]).sum();
            let mut ein = vec![0.0_f32; N_EMBD];
            rms_norm_f32(sr, &l.pre_ffw_norm_2, EPS, &mut ein)?;
            for s in 0..N_USED {
                let e = idx[s];
                sel_flat[i * N_USED + s] = e;
                wts[i * N_USED + s] = (probs[e] / wsum) * l.ffn_down_exps_s[e];
                ein_rep[(i * N_USED + s) * N_EMBD..(i * N_USED + s + 1) * N_EMBD]
                    .copy_from_slice(&ein);
            }
        }

        // ONE batched gate_up over all slots -> [ns, gu_rows]; swiglu -> h [ns, EXPERT_FF].
        let mut gu = vec![0.0_f32; ns * gu_rows];
        self.experts_ew(
            &l.ffn_gate_up_exps,
            &sel_flat,
            gu_rows,
            N_EMBD,
            &ein_rep,
            N_EMBD,
            &mut gu,
        )?;
        let mut h = vec![0.0_f32; ns * EXPERT_FF];
        h.par_chunks_mut(EXPERT_FF).enumerate().for_each(|(s, hs)| {
            let base = s * gu_rows;
            let mut g = gu[base..base + EXPERT_FF].to_vec();
            apply_geglu_inplace_f32(&mut g, &gu[base + EXPERT_FF..base + gu_rows]);
            hs.copy_from_slice(&g);
        });

        // ONE batched down over all slots -> [ns, N_EMBD].
        let mut dn = vec![0.0_f32; ns * N_EMBD];
        self.experts_ew(
            &l.ffn_down_exps,
            &sel_flat,
            N_EMBD,
            EXPERT_FF,
            &h,
            EXPERT_FF,
            &mut dn,
        )?;

        // Per-token combine: weighted expert sum, then post_ffw_norm_2.
        let moe_err: Mutex<Option<DiffusionGemmaError>> = Mutex::new(None);
        out.par_chunks_mut(N_EMBD).enumerate().for_each(|(i, or)| {
            if matches!(moe_err.lock(), Ok(g) if g.is_some()) {
                return;
            }
            for s in 0..N_USED {
                let slot = i * N_USED + s;
                let w = wts[slot];
                for t in 0..N_EMBD {
                    or[t] += w * dn[slot * N_EMBD + t];
                }
            }
            let mut nrm = vec![0.0_f32; N_EMBD];
            if let Err(e) = rms_norm_f32(or, &l.post_ffw_norm_2, EPS, &mut nrm) {
                if let Ok(mut guard) = moe_err.lock() {
                    *guard = Some(DiffusionGemmaError::RmsNorm(e));
                }
                return;
            }
            or.copy_from_slice(&nrm);
        });
        if let Ok(Some(e)) = moe_err.into_inner() {
            return Err(e);
        }
        Ok(())
    }

    /// Project output-normed hidden -> vocab logits via the tied token_embd head, with softcap.
    fn lm_head(&self, hidden: &[f32], logits: &mut [f32]) -> DiffusionResult<()> {
        self.gemv_qw(&self.token_embd, N_VOCAB, N_EMBD, hidden, logits)?;
        for v in logits.iter_mut() {
            *v = SOFTCAP * (*v / SOFTCAP).tanh();
        }
        Ok(())
    }

    /// Self-conditioning MLP: soft -> pre_norm -> gated FFN -> sc. `soft` is [N_EMBD] already
    /// scaled by sqrt(N_EMBD); returns the contribution to add to the scaled embedding.
    fn self_cond(&self, soft: &[f32], out: &mut [f32]) -> DiffusionResult<()> {
        let mut scn = vec![0.0_f32; N_EMBD];
        rms_norm_f32(soft, &self.self_cond_norm, EPS, &mut scn)?;
        let mut gate = vec![0.0_f32; DENSE_FF];
        let mut up = vec![0.0_f32; DENSE_FF];
        self.gemv_qw(&self.self_cond_gate, DENSE_FF, N_EMBD, &scn, &mut gate)?;
        self.gemv_qw(&self.self_cond_up, DENSE_FF, N_EMBD, &scn, &mut up)?;
        apply_geglu_inplace_f32(&mut gate, &up);
        // down (Q5_0 -> dequantized f32): [N_EMBD, DENSE_FF]
        self.gemv_qw(&self.self_cond_down, N_EMBD, DENSE_FF, &gate, out)?;
        Ok(())
    }

    /// Run the single-block block-diffusion denoise loop over a `CANVAS` of tokens conditioned
    /// on `prompt`. Returns timing + the final argmax canvas tokens + the per-step entropy trace.
    pub fn generate(
        &self,
        prompt: &[u32],
        steps: usize,
        seed: u64,
    ) -> DiffusionResult<GenStats> {
        const SC_K: usize = 256;
        let scale = (N_EMBD as f32).sqrt();
        let prefix = prompt.len();
        let nt = prefix + CANVAS;
        let positions: Vec<usize> = (0..nt).collect();
        let mut rng = Lcg::new(seed);

        // precompute scaled prompt embeddings (constant across steps)
        let mut emb_scaled = vec![0.0_f32; nt * N_EMBD];
        for i in 0..prefix {
            self.embed(prompt[i], &mut emb_scaled[i * N_EMBD..(i + 1) * N_EMBD]);
            for t in 0..N_EMBD {
                emb_scaled[i * N_EMBD + t] *= scale;
            }
        }

        // canvas init: random tokens
        let mut canvas: Vec<u32> = (0..CANVAS)
            .map(|_| (rng.next() % N_VOCAB as u64) as u32)
            .collect();
        let mut argmax_canvas = vec![u32::MAX; CANVAS];
        let mut prev_argmax = vec![u32::MAX; CANVAS];
        // self-cond top-k (id,prob) per canvas position; empty (prob 0) on step 1
        let mut sc_ids = vec![0u32; CANVAS * SC_K];
        let mut sc_probs = vec![0.0f32; CANVAS * SC_K];
        let mut have_sc = false;

        let mut entropy_trace: Vec<(usize, f32, usize)> = Vec::new();
        let t0 = std::time::Instant::now();
        let mut steps_run = 0usize;

        for step in (1..=steps).rev() {
            steps_run += 1;
            // build input embeddings for this step
            let mut inpl = emb_scaled.clone();
            for c in 0..CANVAS {
                let row = (prefix + c) * N_EMBD;
                // scaled embedding of the current canvas token
                let mut e = vec![0.0_f32; N_EMBD];
                self.embed(canvas[c], &mut e);
                for t in 0..N_EMBD {
                    e[t] *= scale;
                }
                // self-conditioning soft embedding from previous step
                let mut sc = vec![0.0_f32; N_EMBD];
                if have_sc {
                    let mut soft = vec![0.0_f32; N_EMBD];
                    let mut erow = vec![0.0_f32; N_EMBD];
                    for k in 0..SC_K {
                        let p = sc_probs[c * SC_K + k];
                        if p == 0.0 {
                            continue;
                        }
                        self.embed(sc_ids[c * SC_K + k], &mut erow);
                        for t in 0..N_EMBD {
                            soft[t] += p * erow[t];
                        }
                    }
                    for t in 0..N_EMBD {
                        soft[t] *= scale;
                    }
                    self.self_cond(&soft, &mut sc)?;
                }
                // inpL = scaleless_rms(emb_scaled + sc)
                let ones = vec![1.0_f32; N_EMBD];
                let mut summed = vec![0.0_f32; N_EMBD];
                for t in 0..N_EMBD {
                    summed[t] = e[t] + sc[t];
                }
                rms_norm_f32(&summed, &ones, EPS, &mut inpl[row..row + N_EMBD])?;
            }

            let outv = self.forward_masked(&inpl, &positions, prefix)?;

            // sample each canvas position (parallel over the canvas; lm_head + full-vocab
            // softmax/sort dominate the per-step cost). Randomness is a deterministic per
            // (seed, step, position) draw so the parallel map stays reproducible.
            let temp = 0.4 + 0.4 * (step as f32 / steps as f32);
            let mut entropy = vec![0.0_f32; CANVAS];
            let mut sampled = vec![0u32; CANVAS];
            // Batched output head: all canvas logits in one big parallel GEMM (the dominant
            // matmul), then a nest-free parallel sample over the canvas.
            let canvas_hidden = &outv[prefix * N_EMBD..(prefix + CANVAS) * N_EMBD];
            let mut all_logits = vec![0.0_f32; CANVAS * N_VOCAB];
            self.gemm_qw(
                &self.token_embd,
                N_VOCAB,
                N_EMBD,
                canvas_hidden,
                &mut all_logits,
                CANVAS,
            )?;
            all_logits.par_chunks_mut(N_VOCAB).for_each(|lg| {
                for v in lg.iter_mut() {
                    *v = SOFTCAP * (*v / SOFTCAP).tanh();
                }
            });
            let results: DiffusionResult<Vec<(f32, u32, u32, Vec<(u32, f32)>)>> = (0..CANVAS)
                .into_par_iter()
                .map(|c| {
                    let mut logits = all_logits[c * N_VOCAB..(c + 1) * N_VOCAB].to_vec();
                    let mut maxl = f32::NEG_INFINITY;
                    let mut amax = 0usize;
                    for v in 0..N_VOCAB {
                        let x = logits[v] / temp;
                        if x > maxl {
                            maxl = x;
                            amax = v;
                        }
                    }
                    let mut sum = 0.0f32;
                    for v in 0..N_VOCAB {
                        let p = (logits[v] / temp - maxl).exp();
                        logits[v] = p;
                        sum += p;
                    }
                    let mut ent = 0.0f32;
                    let r = det_unif(
                        seed ^ (step as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (c as u64),
                    ) * sum;
                    let mut cum = 0.0f32;
                    let mut tok = amax as u32;
                    let mut picked = false;
                    for v in 0..N_VOCAB {
                        let p = logits[v] / sum;
                        if p > 0.0 {
                            ent -= p * p.ln();
                        }
                        cum += logits[v];
                        if !picked && cum >= r {
                            tok = v as u32;
                            picked = true;
                        }
                    }
                    let mut order: Vec<usize> = (0..N_VOCAB).collect();
                    order.select_nth_unstable_by(SC_K, |&a, &b| f32_cmp(logits[b], logits[a]));
                    let sc: Vec<(u32, f32)> = order[..SC_K]
                        .iter()
                        .map(|&id| (id as u32, logits[id] / sum))
                        .collect();
                    Ok((ent, tok, amax as u32, sc))
                })
                .collect();
            let results = results?;
            for (c, (ent, tok, amax, sc)) in results.into_iter().enumerate() {
                entropy[c] = ent;
                sampled[c] = tok;
                argmax_canvas[c] = amax;
                for (k, (id, p)) in sc.into_iter().enumerate() {
                    sc_ids[c * SC_K + k] = id;
                    sc_probs[c * SC_K + k] = p;
                }
            }
            have_sc = true;

            // entropy-bound accept (ascending entropy prefix while cumsum <= 0.1)
            let mut ord: Vec<usize> = (0..CANVAS).collect();
            ord.sort_by(|&a, &b| f32_cmp(entropy[a], entropy[b]));
            let mut accept = vec![false; CANVAS];
            let mut pref = 0.0f32;
            let mut n_accept = 0;
            for &c in &ord {
                if pref <= 0.1 {
                    accept[c] = true;
                    pref += entropy[c];
                    n_accept += 1;
                } else {
                    break;
                }
            }
            let mean_ent: f32 = entropy.iter().sum::<f32>() / CANVAS as f32;
            entropy_trace.push((step, mean_ent, n_accept));

            let stable = argmax_canvas == prev_argmax;
            let confident = mean_ent < 0.005;
            if stable && confident {
                break;
            }
            prev_argmax.copy_from_slice(&argmax_canvas);
            // renoise non-accepted
            for c in 0..CANVAS {
                canvas[c] = if accept[c] {
                    sampled[c]
                } else {
                    (rng.next() % N_VOCAB as u64) as u32
                };
            }
        }

        let gen_secs = t0.elapsed().as_secs_f64();
        Ok(GenStats {
            steps_run,
            canvas_tokens: CANVAS,
            gen_secs,
            canvas_tok_s: CANVAS as f64 / gen_secs,
            entropy_trace,
            tokens: argmax_canvas,
        })
    }

    /// Forward with a causal prefix mask: query positions `< prefix` attend only `j <= i`
    /// (encoder/prompt prefix); canvas positions attend all (bidirectional + full cross).
    fn forward_masked(
        &self,
        inpl: &[f32],
        positions: &[usize],
        prefix: usize,
    ) -> DiffusionResult<Vec<f32>> {
        let mut buf = inpl.to_vec();
        self.forward_inner(&mut buf, positions, prefix)
    }
}

/// Deterministic uniform in [0,1) from a 64-bit key (splitmix64 finalizer).
fn det_unif(mut z: u64) -> f32 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

/// Cheap deterministic RNG (xorshift-ish LCG) to avoid an external dependency.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Timing + output of a single denoise block.
pub struct GenStats {
    pub steps_run: usize,
    pub canvas_tokens: usize,
    pub gen_secs: f64,
    pub canvas_tok_s: f64,
    /// (step, mean_entropy, n_accepted) per denoising step.
    pub entropy_trace: Vec<(usize, f32, usize)>,
    pub tokens: Vec<u32>,
}
