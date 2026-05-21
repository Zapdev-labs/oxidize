use crate::gguf::{GgufQuantizationType, MappedGgufFile};
use crate::kv_cache::{KvCache, KvCacheConfig};
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::quantization::{dequantize_scalar, quantized_size};
use crate::flash_attention::flash_attention_decode_f32;
use crate::tensor::{
    DType, apply_rope_f32, apply_swiglu_f32, extract_bits, f16_le_to_f32, gemv_f32_transposed,
    gemv_quantized_f32_transposed, rms_norm_f32, scaled_dot_product_attention_f32,
};

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

#[derive(Debug, Clone, PartialEq)]
pub enum WeightStorage {
    F32(Vec<f32>),
    Quantized(GgufQuantizationType, Vec<u8>),
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
        }
    }

    pub fn output_dim(&self, input_dim: usize) -> usize {
        match self {
            WeightStorage::F32(v) => v.len() / input_dim,
            WeightStorage::Quantized(qtype, v) => {
                let (block_width, block_size) = match qtype {
                    GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => (256, 130),
                    GgufQuantizationType::Q6_K => (256, 194),
                    GgufQuantizationType::Q8_0 => (32, 34),
                    _ => (1, 4), // fallback to f32
                };
                let bytes_per_row = (input_dim / block_width) * block_size;
                if bytes_per_row == 0 {
                    return 0;
                }
                v.len() / bytes_per_row
            }
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
}

impl InferenceModel {
    pub fn load_from_gguf(
        mapped: &MappedGgufFile,
        config: InferenceConfig,
    ) -> Result<Self, String> {
        let mut tok_embeddings: Option<WeightStorage> = None;
        let mut tok_embeddings_cols: usize = config.hidden_size;
        let mut norm_weight: Option<Vec<f32>> = None;
        let mut output_weight: Option<WeightStorage> = None;
        let mut layers: Vec<LayerWeights> = vec![LayerWeights::default(); config.layer_count];

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
                // Only keep non-k-quant types quantized (Q8_0, Q4_0, Q5_0, Q5_1 have working GEMV).
                // K-quants (Q2_K..Q8_K) are dequantized to F32 at load time because
                // their on-the-fly GEMV is not yet fully optimized/correct.
                let is_supported_quant_gemv = matches!(
                    qtype,
                    GgufQuantizationType::Q8_0
                        | GgufQuantizationType::Q4_0
                        | GgufQuantizationType::Q4_1
                        | GgufQuantizationType::Q5_0
                        | GgufQuantizationType::Q5_1
                );
                if should_keep_quantized(name) && is_supported_quant_gemv {
                    Ok(WeightStorage::Quantized(qtype, qdata.to_vec()))
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
                            layers[layer_idx].ffn_down_bias =
                                load_bias(qtype, qdata, value_count)?
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
            dtype: DType::F32,
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
        })
    }

    fn forward_single(&mut self, token: Token, pos: usize) -> Result<Logits, ModelError> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let k = cfg.num_key_value_heads;

        // embedding lookup
        // token_embd.weight is stored as [hidden_size, vocab_size] row-major (transposed vs PyTorch)
        let mut x = vec![0.0_f32; h];
        let token_idx = (token as usize).min(cfg.vocab_size.saturating_sub(1));
        match &self.tok_embeddings {
            WeightStorage::F32(data) => {
                for (i, value) in x.iter_mut().enumerate().take(h) {
                    *value = data[i * self.tok_embeddings_cols + token_idx];
                }
            }
            WeightStorage::Quantized(qtype, data) => {
                let cols = self.tok_embeddings_cols;
                let (block_width, block_size, bits, zero_point) = match qtype {
                    GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
                        (256, 130, 4, 8.0)
                    }
                    GgufQuantizationType::Q6_K => (256, 194, 6, 32.0),
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
                    if *qtype == GgufQuantizationType::Q8_0 {
                        *value = (block[2 + pos_in_block] as i8) as f32 * d;
                    } else {
                        let bitstream = &block[2..];
                        let q = extract_bits(bitstream, pos_in_block, bits) as f32;
                        *value = (q - zero_point) * d;
                    }
                }
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
                    let mut normed = vec![0.0_f32; h];
                    rms_norm_f32(&x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed)
                        .map_err(|e| ModelError::InferenceFailed(format!("mamba_norm: {:?}", e)))?;

                    // Gate branch
                    let mut gate = Vec::new();
                    if !layer.attn_gate.is_empty() {
                        let gate_len = layer.attn_gate.output_dim(h);
                        if gate_len > 0 {
                            gate = vec![0.0_f32; gate_len];
                            gemv_weight(&layer.attn_gate, gate_len, h, &normed, &mut gate)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("attn_gate: {:?}", e))
                                })?;
                        }
                    }

                    // SSM branch projection: [h] -> [8192]
                    let qkv_out_len = layer.attn_qkv.output_dim(h);
                    let mut x_proj = vec![0.0_f32; qkv_out_len];
                    gemv_weight(&layer.attn_qkv, qkv_out_len, h, &normed, &mut x_proj)
                        .map_err(|e| ModelError::InferenceFailed(format!("attn_qkv: {:?}", e)))?;

                    // Causal conv1d over qkv_out_len channels
                    let conv_kernel = 4_usize;
                    let mut conv_out = vec![0.0_f32; qkv_out_len];
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
                        conv_out.copy_from_slice(&x_proj);
                    }

                    // Update conv buffer
                    self.ssm_conv_buffers[layer_idx].push(x_proj);
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
                    let z_gate = if qkv_out_len > half {
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
                    let gate_to_use = if !gate.is_empty() && gate.len() == mamba_out.len() {
                        // Use attn_gate if available
                        let mut silu_gate = gate;
                        for val in silu_gate.iter_mut() {
                            *val = *val * (1.0_f32 / (1.0_f32 + (-*val).exp()));
                        }
                        silu_gate
                    } else if z_gate.len() == mamba_out.len() {
                        // Use second half of qkv projection
                        z_gate
                    } else {
                        vec![1.0_f32; mamba_out.len()]
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
                let mut attn_out = vec![0.0_f32; h];
                {
                    let mut normed = vec![0.0_f32; h];
                    rms_norm_f32(&x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed)
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

                    let mut q_full = vec![0.0_f32; q_len];
                    gemv_weight(&layer.attn_q, q_len, h, &normed, &mut q_full)
                        .map_err(|e| ModelError::InferenceFailed(format!("attn_q: {:?}", e)))?;
                    if !layer.attn_q_bias.is_empty() {
                        for i in 0..q_full.len() {
                            q_full[i] += layer.attn_q_bias[i % layer.attn_q_bias.len()];
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
                    let mut q = q_full[..q_len_actual].to_vec();

                    let mut k_vec = vec![0.0_f32; kv_len];
                    let mut v_vec = vec![0.0_f32; kv_len];
                    if !layer.attn_k.is_empty() {
                        gemv_weight(&layer.attn_k, kv_len, h, &normed, &mut k_vec)
                            .map_err(|e| ModelError::InferenceFailed(format!("attn_k: {:?}", e)))?;
                        if !layer.attn_k_bias.is_empty() {
                            for i in 0..k_vec.len() {
                                k_vec[i] += layer.attn_k_bias[i % layer.attn_k_bias.len()];
                            }
                        }
                    }
                    if !layer.attn_v.is_empty() {
                        gemv_weight(&layer.attn_v, kv_len, h, &normed, &mut v_vec)
                            .map_err(|e| ModelError::InferenceFailed(format!("attn_v: {:?}", e)))?;
                        if !layer.attn_v_bias.is_empty() {
                            for i in 0..v_vec.len() {
                                v_vec[i] += layer.attn_v_bias[i % layer.attn_v_bias.len()];
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
                            if end <= q.len() {
                                let mut normed_head = vec![0.0_f32; q_head_dim];
                                rms_norm_f32(
                                    &q[start..end],
                                    &layer.attn_q_norm,
                                    cfg.rms_norm_eps,
                                    &mut normed_head,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("q_norm: {:?}", e))
                                })?;
                                q[start..end].copy_from_slice(&normed_head);
                            }
                        }
                    }
                    if !layer.attn_k_norm.is_empty() && kv_head_dim == layer.attn_k_norm.len() {
                        for head in 0..kv_heads {
                            let start = head * kv_head_dim;
                            let end = start + kv_head_dim;
                            if end <= k_vec.len() {
                                let mut normed_head = vec![0.0_f32; kv_head_dim];
                                rms_norm_f32(
                                    &k_vec[start..end],
                                    &layer.attn_k_norm,
                                    cfg.rms_norm_eps,
                                    &mut normed_head,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("k_norm: {:?}", e))
                                })?;
                                k_vec[start..end].copy_from_slice(&normed_head);
                            }
                        }
                    }

                    // apply RoPE to Q
                    for head in 0..q_heads {
                        let off = head * q_head_dim;
                        if off + q_head_dim > q.len() {
                            break;
                        }
                        let mut rotated = vec![0.0_f32; q_head_dim];
                        apply_rope_f32(
                            &q[off..off + q_head_dim],
                            pos,
                            q_head_dim,
                            cfg.rope_theta,
                            &mut rotated,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("rope q: {:?}", e)))?;
                        q[off..off + q_head_dim].copy_from_slice(&rotated);
                    }

                    // apply RoPE to K
                    for head in 0..kv_heads {
                        let off = head * kv_head_dim;
                        if off + kv_head_dim > k_vec.len() {
                            break;
                        }
                        let mut rotated = vec![0.0_f32; kv_head_dim];
                        apply_rope_f32(
                            &k_vec[off..off + kv_head_dim],
                            pos,
                            kv_head_dim,
                            cfg.rope_theta,
                            &mut rotated,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("rope k: {:?}", e)))?;
                        k_vec[off..off + kv_head_dim].copy_from_slice(&rotated);
                    }

                    // store in KV cache
                    self.kv_cache
                        .set(layer_idx, pos, &k_vec, &v_vec)
                        .map_err(|e| ModelError::InferenceFailed(format!("kv set: {:?}", e)))?;

                    // retrieve full key/value sequence up to pos in one copy
                    let seq_len = pos + 1;
                    let mut key_cache = vec![0.0_f32; seq_len * kv_len];
                    let mut value_cache = vec![0.0_f32; seq_len * kv_len];
                    self.kv_cache
                        .copy_layer_keys(layer_idx, seq_len, &mut key_cache)
                        .map_err(|e| ModelError::InferenceFailed(format!("kv copy keys: {:?}", e)))?;
                    self.kv_cache
                        .copy_layer_values(layer_idx, seq_len, &mut value_cache)
                        .map_err(|e| ModelError::InferenceFailed(format!("kv copy values: {:?}", e)))?;

                    // compute attention per head using flash attention decode
                    let mut attn_result = vec![0.0_f32; q_len_used];
                    let actual_kv_group_size = q_heads
                        .checked_div(kv_heads)
                        .filter(|group_size| *group_size > 0)
                        .unwrap_or(1);
                    for head in 0..q_heads {
                        let kv_head = head / actual_kv_group_size;
                        let q_head_start = head * q_head_dim;
                        let q_head_end = q_head_start + q_head_dim;
                        if q_head_end > q.len() {
                            break;
                        }
                        let q_head = &q[q_head_start..q_head_end];

                        // In MLA-style architectures, Q may have larger head_dim than KV.
                        let q_head_for_attn = if q_head_dim > kv_head_dim {
                            &q_head[..kv_head_dim]
                        } else {
                            q_head
                        };

                        let mut out_head = vec![0.0_f32; kv_head_dim];
                        flash_attention_decode_f32(
                            q_head_for_attn,
                            &key_cache,
                            &value_cache,
                            seq_len,
                            kv_head_dim,
                            kv_len,
                            kv_head,
                            &mut out_head,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("flash attention: {:?}", e)))?;

                        let write_start = head * kv_head_dim;
                        if write_start + out_head.len() <= attn_result.len() {
                            attn_result[write_start..write_start + out_head.len()]
                                .copy_from_slice(&out_head);
                        }
                    }

                    // Reconcile attention result size with attn_output expected input
                    let attn_input = if attn_output_input_len > 0
                        && attn_result.len() != attn_output_input_len
                    {
                        if attn_result.len() >= attn_output_input_len {
                            attn_result[..attn_output_input_len].to_vec()
                        } else {
                            let mut padded = vec![0.0_f32; attn_output_input_len];
                            padded[..attn_result.len()].copy_from_slice(&attn_result);
                            padded
                        }
                    } else {
                        attn_result
                    };

                    if !layer.attn_output.is_empty() && attn_output_input_len > 0 {
                        gemv_weight(
                            &layer.attn_output,
                            h,
                            attn_output_input_len,
                            &attn_input,
                            &mut attn_out,
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("attn_output: {:?}", e))
                        })?;
                        if !layer.attn_output_bias.is_empty() {
                            for i in 0..attn_out.len() {
                                attn_out[i] += layer.attn_output_bias[i % layer.attn_output_bias.len()];
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
                let mut ffn_out = vec![0.0_f32; h];
                {
                    let mut normed = vec![0.0_f32; h];
                    rms_norm_f32(&x, ffn_norm_weight, cfg.rms_norm_eps, &mut normed)
                        .map_err(|e| ModelError::InferenceFailed(format!("ffn_norm: {:?}", e)))?;

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
                    apply_swiglu_f32(&gate, &up, &mut swiglu)
                        .map_err(|e| ModelError::InferenceFailed(format!("swiglu: {:?}", e)))?;

                    gemv_weight(
                        &layer.ffn_down,
                        h,
                        cfg.intermediate_size,
                        &swiglu,
                        &mut ffn_out,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("ffn_down: {:?}", e)))?;
                    if !layer.ffn_down_bias.is_empty() {
                        for i in 0..ffn_out.len() {
                            ffn_out[i] += layer.ffn_down_bias[i % layer.ffn_down_bias.len()];
                        }
                    }
                }

                for i in 0..h {
                    x[i] += ffn_out[i];
                }
            }
        }

        let mut normed = vec![0.0_f32; h];
        rms_norm_f32(&x, &self.norm_weight, cfg.rms_norm_eps, &mut normed)
            .map_err(|e| ModelError::InferenceFailed(format!("final_norm: {:?}", e)))?;

        let mut logits = vec![0.0_f32; cfg.vocab_size];
        gemv_weight(&self.output_weight, cfg.vocab_size, h, &normed, &mut logits)
            .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))?;

        Ok(logits)
    }
}

impl Model for InferenceModel {
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
            logits = self.forward_single(token, pos)?;
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
