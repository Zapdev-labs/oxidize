use crate::flash_attention::flash_attention_decode_f32;
use crate::flash_attention::flash_attention_decode_heads_f32;
use crate::gguf::{GgufQuantizationType, MappedGgufFile};
use crate::kv_cache::{KvCache, KvCacheConfig};
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::quantization::{dequantize_scalar, quantized_size};
use crate::tensor::{
    DType, apply_rope_f32, apply_swiglu_f32, extract_bits, f16_le_to_f32, gemv_f32_transposed,
    gemv_quantized_f32_transposed, rms_norm_f32, rms_norm_gemv_f32_transposed,
};
use memmap2::Mmap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceConfig {
    pub vocab_size: usize,
    pub context_size: usize,
    pub layer_count: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub key_value_head_dim: usize,
    pub kv_cache_dtype: DType,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl InferenceConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_head_dim(&self) -> usize {
        if self.key_value_head_dim > 0 {
            self.key_value_head_dim
        } else {
            self.head_dim()
        }
    }
}

/// Pre-allocated scratch buffers reused across tokens and layers to eliminate
/// per-token `Vec<f32>` allocations in the hot decode path.
#[derive(Debug, Clone, PartialEq)]
pub struct Workspace {
    // Persistent activation across layers.
    pub x: Vec<f32>,
    // Hidden-size scratch (RMSNorm output, attention output, FFN output, etc.)
    pub hidden_a: Vec<f32>,
    pub hidden_b: Vec<f32>,
    // Intermediate-size scratch (gate, up, SwiGLU)
    pub intermediate_a: Vec<f32>,
    pub intermediate_b: Vec<f32>,
    pub intermediate_c: Vec<f32>,
    // Q/K/V projection scratch — sized to max Q/K/V length.
    pub q_full: Vec<f32>,
    pub k_vec: Vec<f32>,
    pub v_vec: Vec<f32>,
    // Attention result scratch.
    pub attn_result: Vec<f32>,
    // Per-head scratch (flash-attn output, RoPE, per-head norm).
    pub head_scratch: Vec<f32>,
    // KV cache copy fallback buffers.
    pub kv_keys_copy: Vec<f32>,
    pub kv_values_copy: Vec<f32>,
    // Final logits buffer.
    pub logits: Vec<f32>,
    // Mamba/SSM scratch.
    pub mamba_scratch: Vec<f32>,
    pub conv_out: Vec<f32>,
}

impl Workspace {
    pub fn for_config(config: &InferenceConfig) -> Self {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let max_kv_len = config.num_key_value_heads * config.kv_head_dim();
        let max_qkv = (h * 3).max(inter);
        let head_dim = config.head_dim().max(config.kv_head_dim());
        let kv_copy_size = config.context_size * max_kv_len;

        Self {
            x: vec![0.0_f32; h],
            hidden_a: vec![0.0_f32; h],
            hidden_b: vec![0.0_f32; h],
            intermediate_a: vec![0.0_f32; inter],
            intermediate_b: vec![0.0_f32; inter],
            intermediate_c: vec![0.0_f32; inter],
            q_full: vec![0.0_f32; max_qkv],
            k_vec: vec![0.0_f32; max_kv_len],
            v_vec: vec![0.0_f32; max_kv_len],
            attn_result: vec![0.0_f32; h],
            head_scratch: vec![0.0_f32; head_dim],
            kv_keys_copy: vec![0.0_f32; kv_copy_size],
            kv_values_copy: vec![0.0_f32; kv_copy_size],
            logits: vec![0.0_f32; config.vocab_size],
            mamba_scratch: vec![0.0_f32; h * 2],
            conv_out: vec![0.0_f32; max_qkv],
        }
    }
}

#[derive(Debug, Clone)]
pub enum WeightStorage {
    F32(Vec<f32>),
    Quantized(GgufQuantizationType, Vec<u8>),
    /// Zero-copy mmap-backed quantized weights: (qtype, mmap Arc, offset, size).
    /// Keeps weights on disk via page cache; dequantization happens on-the-fly.
    MmapQuantized(GgufQuantizationType, Arc<Mmap>, usize, usize),
}

impl PartialEq for WeightStorage {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (WeightStorage::F32(a), WeightStorage::F32(b)) => a == b,
            (WeightStorage::Quantized(qa, a), WeightStorage::Quantized(qb, b)) => {
                qa == qb && a == b
            }
            (
                WeightStorage::MmapQuantized(qa, _, oa, sa),
                WeightStorage::MmapQuantized(qb, _, ob, sb),
            ) => qa == qb && oa == ob && sa == sb,
            _ => false,
        }
    }
}

impl Default for WeightStorage {
    fn default() -> Self {
        WeightStorage::F32(Vec::new())
    }
}

impl WeightStorage {
    pub fn is_empty(&self) -> bool {
        match self {
            WeightStorage::F32(v) => v.is_empty(),
            WeightStorage::Quantized(_, v) => v.is_empty(),
            WeightStorage::MmapQuantized(_, _, _, size) => *size == 0,
        }
    }

    pub fn output_dim(&self, input_dim: usize) -> usize {
        match self {
            WeightStorage::F32(v) => v.len() / input_dim,
            WeightStorage::Quantized(qtype, v) => {
                let (block_width, block_size) = weight_block_info(*qtype);
                let bytes_per_row = (input_dim / block_width) * block_size;
                if bytes_per_row == 0 {
                    return 0;
                }
                v.len() / bytes_per_row
            }
            WeightStorage::MmapQuantized(qtype, _, _, size) => {
                let (block_width, block_size) = weight_block_info(*qtype);
                let bytes_per_row = (input_dim / block_width) * block_size;
                if bytes_per_row == 0 {
                    return 0;
                }
                size / bytes_per_row
            }
        }
    }
}

fn weight_block_info(qtype: GgufQuantizationType) -> (usize, usize) {
    match qtype {
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => (256, 144),
        GgufQuantizationType::Q6_K => (256, 210),
        GgufQuantizationType::Q8_0 => (32, 34),
        _ => (1, 4), // fallback to f32
    }
}

fn gemv_weight(
    storage: &WeightStorage,
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    // GGUF stores linear weights as [input_dim, output_dim] (transposed vs PyTorch convention)
    // so we use transposed GEMV: output[j] = sum_i W[i][j] * input[i]
    // Callers pass rows=output_dim, cols=input_dim; we swap for transposed GEMV
    match storage {
        WeightStorage::F32(data) => {
            gemv_f32_transposed(data, cols, rows, input, output).map_err(|e| format!("{:?}", e))
        }
        WeightStorage::Quantized(qtype, data) => {
            gemv_quantized_f32_transposed(*qtype, data, cols, rows, input, output)
                .map_err(|e| format!("{:?}", e))
        }
        WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
            let data = &mmap[*offset..*offset + *size];
            gemv_quantized_f32_transposed(*qtype, data, cols, rows, input, output)
                .map_err(|e| format!("{:?}", e))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
struct LayerWeights {
    attn_norm: Vec<f32>,
    attn_q: WeightStorage,
    attn_q_bias: Vec<f32>,
    attn_k: WeightStorage,
    attn_k_bias: Vec<f32>,
    attn_v: WeightStorage,
    attn_v_bias: Vec<f32>,
    attn_output: WeightStorage,
    attn_output_bias: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    ffn_gate: WeightStorage,
    ffn_up: WeightStorage,
    ffn_down: WeightStorage,
    ffn_down_bias: Vec<f32>,
    attn_qkv: WeightStorage,
    // SSM / Mamba tensors
    attn_gate: WeightStorage,
    ssm_a: Vec<f32>,
    ssm_alpha: Vec<f32>,
    ssm_beta: Vec<f32>,
    ssm_conv1d: Vec<f32>,
    ssm_dt_bias: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_out: WeightStorage,
    // Per-head norms for standard attention
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceModel {
    config: InferenceConfig,
    tok_embeddings: WeightStorage,
    tok_embeddings_cols: usize,
    norm_weight: Vec<f32>,
    output_weight: WeightStorage,
    layers: Vec<LayerWeights>,
    kv_cache: KvCache,
    // Mamba/SSM persistent state
    ssm_states: Vec<Vec<f32>>,            // [layer][state_dim]
    ssm_conv_buffers: Vec<Vec<Vec<f32>>>, // [layer][up to 3 previous projections][8192]
    workspace: Workspace,
}

pub(crate) fn lookup_quantized_embedding(
    h: usize,
    cols: usize,
    qtype: GgufQuantizationType,
    data: &[u8],
    token_idx: usize,
    x: &mut [f32],
) {
    let (block_width, block_size, bits, zero_point) = match qtype {
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => (256, 144, 4, 8.0),
        GgufQuantizationType::Q6_K => (256, 210, 6, 32.0),
        GgufQuantizationType::Q8_0 => (32, 34, 8, 0.0),
        _ => (256, 130, 4, 8.0),
    };
    let blocks_per_row = cols / block_width;
    let block_idx = token_idx / block_width;
    let pos_in_block = token_idx % block_width;
    for (i, value) in x.iter_mut().enumerate().take(h) {
        let row_start = i * blocks_per_row * block_size;
        let block_start = row_start + block_idx * block_size;
        let block = &data[block_start..block_start + block_size];
        let d = f16_le_to_f32([block[0], block[1]]);
        if qtype == GgufQuantizationType::Q8_0 {
            *value = (block[2 + pos_in_block] as i8) as f32 * d;
        } else {
            let bitstream = &block[2..];
            let q = extract_bits(bitstream, pos_in_block, bits) as f32;
            *value = (q - zero_point) * d;
        }
    }
}

impl InferenceModel {
    pub fn load_from_gguf(
        mapped: &MappedGgufFile,
        config: InferenceConfig,
        use_mmap: bool,
    ) -> Result<Self, String> {
        let mut tok_embeddings: Option<WeightStorage> = None;
        let mut tok_embeddings_cols: usize = config.hidden_size;
        let mut norm_weight: Option<Vec<f32>> = None;
        let mut output_weight: Option<WeightStorage> = None;
        let mut layers: Vec<LayerWeights> = vec![LayerWeights::default(); config.layer_count];
        let mmap_arc = if use_mmap { Some(mapped.mmap()) } else { None };

        for tensor in mapped.mapped_tensor_infos().iter() {
            let qtype = GgufQuantizationType::from_ggml_type(tensor.ggml_type);
            let value_count: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
            let qsize = quantized_size(qtype, value_count)
                .map_err(|e| format!("quantized_size: {:?}", e))?;
            let offset = tensor.absolute_offset as usize;
            let qdata = &mapped.bytes()[offset..offset + qsize];

            // Helper to decide whether to keep quantized or dequantize
            let should_keep_quantized = |name: &str| -> bool {
                // Keep weight matrices quantized; dequantize norms and small vectors
                name.ends_with(".weight") && !name.contains("norm") && !name.contains("bias")
            };

            let load_tensor = |name: &str,
                               qtype: GgufQuantizationType,
                               qdata: &[u8],
                               count: usize|
             -> Result<WeightStorage, String> {
                // Keep only formats with implemented on-the-fly GEMV kernels quantized.
                let is_supported_quant_gemv = matches!(
                    qtype,
                    GgufQuantizationType::Q8_0
                        | GgufQuantizationType::Q4_K_S
                        | GgufQuantizationType::Q4_K_M
                        | GgufQuantizationType::Q6_K
                );
                if should_keep_quantized(name) && is_supported_quant_gemv {
                    if let Some(ref arc) = mmap_arc {
                        Ok(WeightStorage::MmapQuantized(
                            qtype,
                            arc.clone(),
                            offset,
                            qsize,
                        ))
                    } else {
                        Ok(WeightStorage::Quantized(qtype, qdata.to_vec()))
                    }
                } else {
                    let mut f32_data = vec![0.0_f32; count];
                    dequantize_scalar(qtype, qdata, &mut f32_data)
                        .map_err(|e| format!("dequantize_scalar: {:?}", e))?;
                    Ok(WeightStorage::F32(f32_data))
                }
            };

            let load_bias = |qtype: GgufQuantizationType,
                             qdata: &[u8],
                             count: usize|
             -> Result<Vec<f32>, String> {
                let mut f32_data = vec![0.0_f32; count];
                dequantize_scalar(qtype, qdata, &mut f32_data)
                    .map_err(|e| format!("dequantize_scalar: {:?}", e))?;
                Ok(f32_data)
            };

            match tensor.name.as_str() {
                "tok_embeddings.weight" | "token_embd.weight" => {
                    tok_embeddings_cols = tensor
                        .dimensions
                        .get(1)
                        .copied()
                        .unwrap_or(config.hidden_size as u64)
                        as usize;
                    tok_embeddings = Some(load_tensor(&tensor.name, qtype, qdata, value_count)?);
                }
                "norm.weight" | "output_norm.weight" => {
                    let mut f32_data = vec![0.0_f32; value_count];
                    dequantize_scalar(qtype, qdata, &mut f32_data)
                        .map_err(|e| format!("dequantize_scalar: {:?}", e))?;
                    norm_weight = Some(f32_data);
                }
                "output.weight" => {
                    output_weight = Some(load_tensor(&tensor.name, qtype, qdata, value_count)?);
                }
                name if name.starts_with("blk.") => {
                    let parts: Vec<&str> = name.split('.').collect();
                    if parts.len() < 4 {
                        continue;
                    }
                    let layer_idx: usize = parts[1]
                        .parse()
                        .map_err(|_| format!("bad layer index in tensor name: {}", name))?;
                    if layer_idx >= config.layer_count {
                        continue;
                    }
                    let weight_name = parts[2];
                    let suffix = parts.get(3).copied();
                    match (weight_name, suffix) {
                        ("attn_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].attn_norm = f32_data;
                        }
                        ("attn_q", Some("weight")) => {
                            layers[layer_idx].attn_q = load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_q", Some("bias")) => {
                            layers[layer_idx].attn_q_bias = load_bias(qtype, qdata, value_count)?
                        }
                        ("attn_k", Some("weight")) => {
                            layers[layer_idx].attn_k = load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_k", Some("bias")) => {
                            layers[layer_idx].attn_k_bias = load_bias(qtype, qdata, value_count)?
                        }
                        ("attn_v", Some("weight")) => {
                            layers[layer_idx].attn_v = load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_v", Some("bias")) => {
                            layers[layer_idx].attn_v_bias = load_bias(qtype, qdata, value_count)?
                        }
                        ("attn_output", Some("weight")) => {
                            layers[layer_idx].attn_output =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_output", Some("bias")) => {
                            layers[layer_idx].attn_output_bias =
                                load_bias(qtype, qdata, value_count)?
                        }
                        ("attn_qkv", _) => {
                            layers[layer_idx].attn_qkv =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_gate", _) => {
                            layers[layer_idx].attn_gate =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_q_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].attn_q_norm = f32_data;
                        }
                        ("attn_k_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].attn_k_norm = f32_data;
                        }
                        ("ffn_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ffn_norm = f32_data;
                        }
                        ("post_attention_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].post_attention_norm = f32_data;
                        }
                        ("ffn_gate", _) => {
                            layers[layer_idx].ffn_gate =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_up", _) => {
                            layers[layer_idx].ffn_up = load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_down", Some("weight")) => {
                            layers[layer_idx].ffn_down =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_down", Some("bias")) => {
                            layers[layer_idx].ffn_down_bias = load_bias(qtype, qdata, value_count)?
                        }
                        ("ssm_a", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_a = f32_data;
                        }
                        ("ssm_alpha", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_alpha = f32_data;
                        }
                        ("ssm_beta", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_beta = f32_data;
                        }
                        ("ssm_conv1d", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_conv1d = f32_data;
                        }
                        ("ssm_dt", Some("bias")) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_dt_bias = f32_data;
                        }
                        ("ssm_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_norm = f32_data;
                        }
                        ("ssm_out", _) => {
                            layers[layer_idx].ssm_out =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        let tok_embeddings = tok_embeddings.ok_or("missing tok_embeddings.weight")?;
        let norm_weight = norm_weight.ok_or("missing norm.weight")?;
        let output_weight = output_weight.unwrap_or_else(|| tok_embeddings.clone());

        eprintln!(
            "InferenceConfig: vocab={}, context={}, layers={}, hidden={}, intermediate={}, heads={}, kv_heads={}, kv_head_dim={}, eps={}, theta={}",
            config.vocab_size,
            config.context_size,
            config.layer_count,
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.kv_head_dim(),
            config.rms_norm_eps,
            config.rope_theta
        );

        let kv_cache_config = KvCacheConfig {
            layer_count: config.layer_count,
            context_size: config.context_size,
            head_count: config.num_key_value_heads,
            head_dim: config.kv_head_dim(),
            dtype: config.kv_cache_dtype,
        };
        let kv_cache = KvCache::new(kv_cache_config).map_err(|e| format!("kv_cache: {:?}", e))?;

        // Initialize Mamba/SSM state
        let mut ssm_states: Vec<Vec<f32>> = Vec::with_capacity(config.layer_count);
        let mut ssm_conv_buffers: Vec<Vec<Vec<f32>>> = Vec::with_capacity(config.layer_count);
        for layer in layers.iter().take(config.layer_count) {
            let state_dim = layer.ssm_a.len().max(1);
            ssm_states.push(vec![0.0_f32; state_dim]);
            ssm_conv_buffers.push(Vec::new());
        }

        let workspace = Workspace::for_config(&config);

        Ok(Self {
            config,
            tok_embeddings,
            tok_embeddings_cols,
            norm_weight,
            output_weight,
            layers,
            kv_cache,
            ssm_states,
            ssm_conv_buffers,
            workspace,
        })
    }

    fn forward_single(
        &mut self,
        token: Token,
        pos: usize,
        need_logits: bool,
    ) -> Result<Option<Logits>, ModelError> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let k = cfg.num_key_value_heads;

        let ws = &mut self.workspace;

        // embedding lookup
        // token_embd.weight is stored as [hidden_size, vocab_size] row-major (transposed vs PyTorch)
        let x = &mut ws.x[..h];
        x.fill(0.0_f32);
        let token_idx = (token as usize).min(cfg.vocab_size.saturating_sub(1));
        match &self.tok_embeddings {
            WeightStorage::F32(data) => {
                for (i, value) in x.iter_mut().enumerate().take(h) {
                    *value = data[i * self.tok_embeddings_cols + token_idx];
                }
            }
            WeightStorage::Quantized(qtype, data) => {
                lookup_quantized_embedding(h, self.tok_embeddings_cols, *qtype, data, token_idx, x);
            }
            WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                let data = &mmap[*offset..*offset + *size];
                lookup_quantized_embedding(h, self.tok_embeddings_cols, *qtype, data, token_idx, x);
            }
        }

        for layer_idx in 0..cfg.layer_count {
            let layer = &self.layers[layer_idx];

            // Detect Mamba layers (have attn_qkv but no attn_q)
            let is_mamba = !layer.attn_qkv.is_empty() && layer.attn_q.is_empty();

            // Determine which norm weight to use for FFN
            let ffn_norm_weight: &[f32] = if !layer.post_attention_norm.is_empty() {
                &layer.post_attention_norm
            } else if !layer.ffn_norm.is_empty() {
                &layer.ffn_norm
            } else {
                &[]
            };

            if is_mamba {
                // ---- Mamba/SSM layer ----
                let mamba_out = {
                    let normed = &mut ws.hidden_a[..h];
                    normed.fill(0.0_f32);
                    rms_norm_f32(x, &layer.attn_norm, cfg.rms_norm_eps, normed)
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
                            // Current token (weight[0])
                            sum += layer.ssm_conv1d[c] * x_proj[c];
                            // Previous tokens (weights[1..3])
                            for b in 0..buffer.len() {
                                let weight_idx = (b + 1) * qkv_out_len + c;
                                let buf_idx = buffer.len() - 1 - b;
                                sum += layer.ssm_conv1d[weight_idx] * buffer[buf_idx][c];
                            }
                            conv_out[c] = sum;
                        }
                    } else {
                        conv_out.copy_from_slice(x_proj);
                    }

                    // Update conv buffer
                    self.ssm_conv_buffers[layer_idx].push(x_proj.to_vec());
                    if self.ssm_conv_buffers[layer_idx].len() > conv_kernel - 1 {
                        self.ssm_conv_buffers[layer_idx].remove(0);
                    }

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
                    x[i] += mamba_out[i];
                }
            } else if !layer.attn_q.is_empty() {
                // ---- Standard attention ----
                let attn_out = &mut ws.hidden_a[..h];
                attn_out.fill(0.0_f32);
                {
                    let normed = &mut ws.hidden_b[..h];
                    normed.fill(0.0_f32);
                    rms_norm_f32(x, &layer.attn_norm, cfg.rms_norm_eps, normed)
                        .map_err(|e| ModelError::InferenceFailed(format!("rms_norm: {:?}", e)))?;

                    // Compute dynamic dimensions
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
                    // Use fused RMSNorm+GEMV for attention Q projection when bias is absent.
                    if layer.attn_q_bias.is_empty() {
                        rms_norm_gemv_f32_transposed(
                            x,
                            &layer.attn_norm,
                            cfg.rms_norm_eps,
                            match &layer.attn_q {
                                WeightStorage::F32(data) => data.as_slice(),
                                _ => {
                                    gemv_weight(&layer.attn_q, q_len, h, normed, q_full).map_err(
                                        |e| ModelError::InferenceFailed(format!("attn_q: {:?}", e)),
                                    )?;
                                    // fallthrough handled below
                                    &[][..]
                                }
                            },
                            h,
                            q_len,
                            q_full,
                        )
                        .unwrap_or_else(|_| {
                            gemv_weight(&layer.attn_q, q_len, h, normed, q_full)
                                .expect("attn_q fallback");
                        });
                    } else {
                        gemv_weight(&layer.attn_q, q_len, h, normed, q_full)
                            .map_err(|e| ModelError::InferenceFailed(format!("attn_q: {:?}", e)))?;
                    }
                    if !layer.attn_q_bias.is_empty() {
                        for (i, q) in q_full.iter_mut().enumerate() {
                            *q += layer.attn_q_bias[i % layer.attn_q_bias.len()];
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

                    let k_vec = &mut ws.k_vec[..kv_len];
                    k_vec.fill(0.0_f32);
                    let v_vec = &mut ws.v_vec[..kv_len];
                    v_vec.fill(0.0_f32);
                    if !layer.attn_k.is_empty() {
                        gemv_weight(&layer.attn_k, kv_len, h, normed, k_vec)
                            .map_err(|e| ModelError::InferenceFailed(format!("attn_k: {:?}", e)))?;
                        if !layer.attn_k_bias.is_empty() {
                            for (i, k) in k_vec.iter_mut().enumerate() {
                                *k += layer.attn_k_bias[i % layer.attn_k_bias.len()];
                            }
                        }
                    }
                    if !layer.attn_v.is_empty() {
                        gemv_weight(&layer.attn_v, kv_len, h, normed, v_vec)
                            .map_err(|e| ModelError::InferenceFailed(format!("attn_v: {:?}", e)))?;
                        if !layer.attn_v_bias.is_empty() {
                            for (i, v) in v_vec.iter_mut().enumerate() {
                                *v += layer.attn_v_bias[i % layer.attn_v_bias.len()];
                            }
                        }
                    }

                    // Compute head dimensions based on ACTUAL Q length after splitting
                    let q_len_used = q.len();
                    let q_head_dim = if n > 0 && q_len_used.is_multiple_of(n) {
                        q_len_used / n
                    } else {
                        q_len_used
                    };
                    let q_heads = q_len_used.checked_div(q_head_dim).unwrap_or(1);

                    let kv_head_dim = if k > 0 && kv_len.is_multiple_of(k) {
                        kv_len / k
                    } else if kv_len > 0 {
                        kv_len
                    } else {
                        q_head_dim
                    };
                    let kv_heads = kv_len.checked_div(kv_head_dim).unwrap_or(1);

                    // Apply per-head Q/K norm if available
                    if !layer.attn_q_norm.is_empty() && q_head_dim == layer.attn_q_norm.len() {
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
                            .map_err(|e| ModelError::InferenceFailed(format!("q_norm: {:?}", e)))?;
                            q[start..end].copy_from_slice(normed_head);
                        }
                    }
                    if !layer.attn_k_norm.is_empty() && kv_head_dim == layer.attn_k_norm.len() {
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
                            .map_err(|e| ModelError::InferenceFailed(format!("k_norm: {:?}", e)))?;
                            k_vec[start..end].copy_from_slice(normed_head);
                        }
                    }

                    // apply RoPE to Q
                    for head in 0..q_heads {
                        let off = head * q_head_dim;
                        if off + q_head_dim > q.len() {
                            break;
                        }
                        let rotated = &mut ws.head_scratch[..q_head_dim];
                        rotated.fill(0.0_f32);
                        apply_rope_f32(
                            &q[off..off + q_head_dim],
                            pos,
                            q_head_dim,
                            cfg.rope_theta,
                            rotated,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("rope q: {:?}", e)))?;
                        q[off..off + q_head_dim].copy_from_slice(rotated);
                    }

                    // apply RoPE to K
                    for head in 0..kv_heads {
                        let off = head * kv_head_dim;
                        if off + kv_head_dim > k_vec.len() {
                            break;
                        }
                        let rotated = &mut ws.head_scratch[..kv_head_dim];
                        rotated.fill(0.0_f32);
                        apply_rope_f32(
                            &k_vec[off..off + kv_head_dim],
                            pos,
                            kv_head_dim,
                            cfg.rope_theta,
                            rotated,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("rope k: {:?}", e)))?;
                        k_vec[off..off + kv_head_dim].copy_from_slice(rotated);
                    }

                    // store in KV cache
                    self.kv_cache
                        .set(layer_idx, pos, k_vec, v_vec)
                        .map_err(|e| ModelError::InferenceFailed(format!("kv set: {:?}", e)))?;

                    // Borrow the F32 KV prefix when the logical prefix is still
                    // contiguous in storage; otherwise copy into workspace buffers.
                    let seq_len = pos + 1;
                    let borrowed_key_cache = self
                        .kv_cache
                        .f32_layer_key_prefix(layer_idx, seq_len)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("kv borrow keys: {:?}", e))
                        })?;
                    let borrowed_value_cache = self
                        .kv_cache
                        .f32_layer_value_prefix(layer_idx, seq_len)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("kv borrow values: {:?}", e))
                        })?;

                    let key_cache: &[f32];
                    let value_cache: &[f32];
                    if let (Some(keys), Some(values)) = (borrowed_key_cache, borrowed_value_cache) {
                        key_cache = keys;
                        value_cache = values;
                    } else {
                        let key_copy = &mut ws.kv_keys_copy[..seq_len * kv_len];
                        key_copy.fill(0.0_f32);
                        let value_copy = &mut ws.kv_values_copy[..seq_len * kv_len];
                        value_copy.fill(0.0_f32);
                        self.kv_cache
                            .copy_layer_keys(layer_idx, seq_len, key_copy)
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("kv copy keys: {:?}", e))
                            })?;
                        self.kv_cache
                            .copy_layer_values(layer_idx, seq_len, value_copy)
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("kv copy values: {:?}", e))
                            })?;
                        key_cache = key_copy;
                        value_cache = value_copy;
                    }

                    // compute attention using parallel flash attention decode over heads
                    let attn_result = &mut ws.attn_result[..q_len_used];
                    attn_result.fill(0.0_f32);

                    // For MLA-style where q_head_dim > kv_head_dim, we need to truncate Q heads.
                    // Build a temporary truncated Q buffer if needed.
                    let q_for_flash: Vec<f32> = if q_head_dim > kv_head_dim {
                        let mut q_truncated = vec![0.0_f32; q_heads * kv_head_dim];
                        for head in 0..q_heads {
                            let src_start = head * q_head_dim;
                            let dst_start = head * kv_head_dim;
                            q_truncated[dst_start..dst_start + kv_head_dim]
                                .copy_from_slice(&q[src_start..src_start + kv_head_dim]);
                        }
                        q_truncated
                    } else {
                        q.to_vec()
                    };
                    flash_attention_decode_heads_f32(
                        &q_for_flash,
                        key_cache,
                        value_cache,
                        seq_len,
                        kv_head_dim,
                        kv_len,
                        q_heads,
                        kv_heads,
                        attn_result,
                    )
                    .map_err(|e| {
                        ModelError::InferenceFailed(format!("flash attention heads: {:?}", e))
                    })?;

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
                                *out += layer.attn_output_bias[i % layer.attn_output_bias.len()];
                            }
                        }
                    }
                }

                for i in 0..h {
                    x[i] += attn_out[i];
                }
            }

            // ---- FFN (shared between Mamba and standard layers) ----
            let has_ffn = !layer.ffn_gate.is_empty()
                && !layer.ffn_up.is_empty()
                && !layer.ffn_down.is_empty()
                && !ffn_norm_weight.is_empty();
            if has_ffn {
                let ffn_out = &mut ws.hidden_a[..h];
                ffn_out.fill(0.0_f32);
                {
                    let normed = &mut ws.hidden_b[..h];
                    normed.fill(0.0_f32);
                    rms_norm_f32(x, ffn_norm_weight, cfg.rms_norm_eps, normed)
                        .map_err(|e| ModelError::InferenceFailed(format!("ffn_norm: {:?}", e)))?;

                    let gate = &mut ws.intermediate_a[..cfg.intermediate_size];
                    gate.fill(0.0_f32);
                    let up = &mut ws.intermediate_b[..cfg.intermediate_size];
                    up.fill(0.0_f32);
                    gemv_weight(&layer.ffn_gate, cfg.intermediate_size, h, normed, gate)
                        .map_err(|e| ModelError::InferenceFailed(format!("ffn_gate: {:?}", e)))?;
                    gemv_weight(&layer.ffn_up, cfg.intermediate_size, h, normed, up)
                        .map_err(|e| ModelError::InferenceFailed(format!("ffn_up: {:?}", e)))?;

                    // Inline SwiGLU into the gate buffer to avoid an intermediate allocation.
                    for (g, u) in gate.iter_mut().zip(up.iter()) {
                        let sigmoid = 1.0_f32 / (1.0 + (-*g).exp());
                        *g = *g * sigmoid * *u;
                    }

                    gemv_weight(&layer.ffn_down, h, cfg.intermediate_size, gate, ffn_out)
                        .map_err(|e| ModelError::InferenceFailed(format!("ffn_down: {:?}", e)))?;
                    if !layer.ffn_down_bias.is_empty() {
                        for (i, out) in ffn_out.iter_mut().enumerate() {
                            *out += layer.ffn_down_bias[i % layer.ffn_down_bias.len()];
                        }
                    }
                }

                for i in 0..h {
                    x[i] += ffn_out[i];
                }
            }
        }

        if !need_logits {
            return Ok(None);
        }

        let normed = &mut ws.hidden_a[..h];
        normed.fill(0.0_f32);
        rms_norm_f32(x, &self.norm_weight, cfg.rms_norm_eps, normed)
            .map_err(|e| ModelError::InferenceFailed(format!("final_norm: {:?}", e)))?;

        let logits = &mut ws.logits[..cfg.vocab_size];
        logits.fill(0.0_f32);
        gemv_weight(&self.output_weight, cfg.vocab_size, h, normed, logits)
            .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))?;

        // Return a clone of the logits slice. This is the only allocation in the
        // hot path for the final token of each forward call.
        Ok(Some(logits.to_vec()))
    }
}

impl Model for InferenceModel {
    fn rewind_to(&mut self, consumed_tokens: usize) {
        if consumed_tokens == 0 {
            let _ = self.kv_cache.rewind_to(0);
            return;
        }
        let _ = self.kv_cache.rewind_to(consumed_tokens.saturating_sub(1));
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
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            let need_logits = i + 1 == tokens.len();
            if let Some(final_logits) = self.forward_single(token, pos, need_logits)? {
                logits = final_logits;
            }
        }
        session.record_tokens(tokens.len());
        Ok(logits)
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

    fn tiny_inference_model() -> InferenceModel {
        let config = InferenceConfig {
            vocab_size: 3,
            context_size: 8,
            layer_count: 0,
            hidden_size: 2,
            intermediate_size: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            key_value_head_dim: 2,
            kv_cache_dtype: DType::F32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let kv_cache_config = KvCacheConfig {
            layer_count: config.layer_count,
            context_size: config.context_size,
            head_count: config.num_key_value_heads,
            head_dim: config.kv_head_dim(),
            dtype: DType::F32,
        };

        InferenceModel {
            config: config.clone(),
            tok_embeddings: WeightStorage::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            tok_embeddings_cols: 3,
            norm_weight: vec![1.0, 1.0],
            output_weight: WeightStorage::F32(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            layers: Vec::new(),
            kv_cache: KvCache::new(kv_cache_config).expect("tiny kv cache should be valid"),
            ssm_states: Vec::new(),
            ssm_conv_buffers: Vec::new(),
            workspace: Workspace::for_config(&config),
        }
    }

    #[test]
    fn workspace_buffer_capacities_cover_model_dimensions() {
        let config = InferenceConfig {
            vocab_size: 100,
            context_size: 16,
            layer_count: 2,
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            key_value_head_dim: 16,
            kv_cache_dtype: DType::F32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let ws = Workspace::for_config(&config);

        assert_eq!(ws.x.len(), config.hidden_size);
        assert_eq!(ws.hidden_a.len(), config.hidden_size);
        assert_eq!(ws.hidden_b.len(), config.hidden_size);
        assert_eq!(ws.intermediate_a.len(), config.intermediate_size);
        assert_eq!(ws.intermediate_b.len(), config.intermediate_size);
        assert_eq!(ws.intermediate_c.len(), config.intermediate_size);

        let max_qkv = (config.hidden_size * 3).max(config.intermediate_size);
        assert_eq!(ws.q_full.len(), max_qkv);

        let max_kv_len = config.num_key_value_heads * config.kv_head_dim();
        assert_eq!(ws.k_vec.len(), max_kv_len);
        assert_eq!(ws.v_vec.len(), max_kv_len);
        assert_eq!(ws.attn_result.len(), config.hidden_size);

        let head_dim = config.head_dim().max(config.kv_head_dim());
        assert_eq!(ws.head_scratch.len(), head_dim);

        let kv_copy_size = config.context_size * max_kv_len;
        assert_eq!(ws.kv_keys_copy.len(), kv_copy_size);
        assert_eq!(ws.kv_values_copy.len(), kv_copy_size);
        assert_eq!(ws.logits.len(), config.vocab_size);
        assert_eq!(ws.mamba_scratch.len(), config.hidden_size * 2);
        assert_eq!(ws.conv_out.len(), max_qkv);
    }

    #[test]
    fn forward_single_can_skip_final_logits() {
        let mut model = tiny_inference_model();

        let skipped_logits = model
            .forward_single(0, 0, false)
            .expect("non-final token should run successfully");
        assert_eq!(skipped_logits, None);

        let final_logits = model
            .forward_single(1, 1, true)
            .expect("final token should run successfully")
            .expect("final token should produce logits");
        assert_eq!(final_logits.len(), model.vocab_size());
    }

    #[test]
    fn forward_returns_only_final_token_logits() {
        let mut prefill_model = tiny_inference_model();
        let mut prefill_session = Session::new();
        let prefill_logits = prefill_model
            .forward(&[0, 1], &mut prefill_session)
            .expect("prefill should produce final logits");

        let mut single_model = tiny_inference_model();
        let mut single_session = Session::new();
        let single_logits = single_model
            .forward(&[1], &mut single_session)
            .expect("single final token should produce logits");

        assert_eq!(prefill_logits, single_logits);
        assert_eq!(prefill_session.consumed_tokens(), 2);
        assert_eq!(single_session.consumed_tokens(), 1);
    }
}
