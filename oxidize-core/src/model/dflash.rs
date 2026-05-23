use crate::flash_attention::flash_attention_decode_heads_f32;
use crate::gguf::{GgufQuantizationType, MappedGgufFile};
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::quantization::{dequantize_scalar, quantized_size};
use crate::safetensors::MappedSafeTensorsFile;
use crate::tensor::{
    DType, apply_rope_f32, f16_le_to_f32, gemm_f32, gemm_quantized_f32, gemv_f32_transposed,
    gemv_quantized_f32, rms_norm_f32,
};

/// DFlash configuration matching the HuggingFace config.json.
#[derive(Debug, Clone, PartialEq)]
pub struct DFlashConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_target_layers: usize,
    pub block_size: usize,
    pub target_layer_ids: Vec<usize>,
    pub mask_token_id: u32,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Default for DFlashConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 8,
            num_target_layers: 40,
            block_size: 16,
            target_layer_ids: vec![1, 10, 19, 28, 37],
            mask_token_id: 248070,
            vocab_size: 248320,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }
}

impl DFlashConfig {
    /// Config for Qwen3.6-35B-A3B-DFlash.
    pub fn qwen3_6_35b_a3b_dflash() -> Self {
        Self::default()
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_head_dim(&self) -> usize {
        self.head_dim()
    }

    pub fn target_hidden_width(&self) -> usize {
        self.hidden_size * self.num_target_layers
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlashFeature {
    Loading,
    TargetHiddenFusion,
    KvCache,
    SpeculativeAlgorithm,
}

pub fn implemented_dflash_features() -> &'static [DFlashFeature] {
    &[
        DFlashFeature::Loading,
        DFlashFeature::TargetHiddenFusion,
        DFlashFeature::KvCache,
        DFlashFeature::SpeculativeAlgorithm,
    ]
}

/// Raw quantized weight kept in its native GGUF row-major layout
/// (`out_dim` rows of `in_dim` columns). Dequantization happens on-the-fly
/// inside the GEMV kernel, so the matrix is read at ~0.5 bytes/weight (Q4_K)
/// instead of 4 bytes/weight, which is the dominant cost during decode.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantWeight {
    pub bytes: Vec<u8>,
    pub qtype: GgufQuantizationType,
    pub out_dim: usize,
    pub in_dim: usize,
}

/// A projection weight, stored either as on-the-fly-dequantized quantized
/// bytes (fast, memory-bandwidth-friendly decode path) or as a plain f32
/// matrix in the transposed layout expected by `gemv_f32_transposed`.
#[derive(Debug, Clone, PartialEq)]
pub struct F32Weight {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub quant: Option<QuantWeight>,
}

impl F32Weight {
    pub fn from_slice(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            quant: None,
        }
    }

    /// Build a weight backed by raw quantized GGUF bytes. `out_dim`/`in_dim`
    /// are the native GGUF dimensions (rows = output features, cols = input
    /// features).
    pub fn from_quantized(
        bytes: Vec<u8>,
        qtype: GgufQuantizationType,
        out_dim: usize,
        in_dim: usize,
    ) -> Self {
        Self {
            data: Vec::new(),
            rows: in_dim,
            cols: out_dim,
            quant: Some(QuantWeight {
                bytes,
                qtype,
                out_dim,
                in_dim,
            }),
        }
    }

    /// True when this weight holds either f32 data or quantized bytes.
    pub fn is_loaded(&self) -> bool {
        !self.data.is_empty() || self.quant.is_some()
    }

    pub fn gemv(&self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        if let Some(q) = &self.quant {
            gemv_quantized_f32(q.qtype, &q.bytes, q.out_dim, q.in_dim, input, output)
                .map_err(|e| format!("{:?}", e))
        } else {
            gemv_f32_transposed(&self.data, self.cols, self.rows, input, output)
                .map_err(|e| format!("{:?}", e))
        }
    }

    /// Batched matmul: `inputs` is row-major `[batch, in_dim]`, `outputs` is
    /// row-major `[batch, out_dim]`. Falls back to [`Self::gemv`] for batch=1.
    pub fn gemm(&self, inputs: &[f32], outputs: &mut [f32], batch: usize) -> Result<(), String> {
        if batch <= 1 {
            return self.gemv(inputs, outputs);
        }
        if let Some(q) = &self.quant {
            gemm_quantized_f32(
                q.qtype, &q.bytes, q.out_dim, q.in_dim, inputs, outputs, batch,
            )
            .map_err(|e| format!("{:?}", e))
        } else {
            // Non-quant weight is stored as [in_dim × out_dim] row-major
            // (`self.cols = in_dim`, `self.rows = out_dim`). gemm_f32 expects
            // C[batch × out_dim] = A[batch × in_dim] · B[in_dim × out_dim].
            gemm_f32(inputs, batch, self.cols, &self.data, self.rows, outputs)
                .map_err(|e| format!("{:?}", e))
        }
    }
}

/// Whether the fused on-the-fly quantized GEMV kernel supports `qtype` for an
/// input dimension of `in_dim`. The kernels iterate full quantization blocks,
/// so `in_dim` must be block-aligned.
fn quantized_gemv_supported(qtype: GgufQuantizationType, in_dim: usize) -> bool {
    match qtype {
        GgufQuantizationType::Q4_K_S
        | GgufQuantizationType::Q4_K_M
        | GgufQuantizationType::Q2_K
        | GgufQuantizationType::Q6_K => in_dim.is_multiple_of(256),
        GgufQuantizationType::Q8_0 => in_dim.is_multiple_of(32),
        _ => false,
    }
}

/// DFlash attention layer: q_proj on noise, k/v_proj on concatenated target+noise.
#[derive(Debug, Clone, PartialEq)]
pub struct DFlashAttentionLayer {
    pub q_proj: F32Weight,
    pub k_proj: F32Weight,
    pub v_proj: F32Weight,
    pub o_proj: F32Weight,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
}

/// DFlash decoder layer: norm → attention → MLP.
#[derive(Debug, Clone, PartialEq)]
pub struct DFlashDecoderLayer {
    pub input_layernorm: Vec<f32>,
    pub attention: DFlashAttentionLayer,
    pub post_attention_layernorm: Vec<f32>,
    pub mlp_gate: F32Weight,
    pub mlp_up: F32Weight,
    pub mlp_down: F32Weight,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DFlashKvLayerCache {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub seq_len: usize,
}

impl DFlashKvLayerCache {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            seq_len: 0,
        }
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_len = 0;
    }

    fn reserve_tokens(&mut self, additional_tokens: usize, kv_len: usize) {
        let additional = additional_tokens.saturating_mul(kv_len);
        self.keys.reserve(additional);
        self.values.reserve(additional);
    }
}

impl Default for DFlashKvLayerCache {
    fn default() -> Self {
        Self::new()
    }
}

/// DFlash draft model.
///
/// Forward pass:
/// 1. noise_embedding → target_hidden fusion via fc layer
/// 2. hidden_norm
/// 3. layer loop (custom attention + MLP)
/// 4. final norm
/// 5. output projection
#[derive(Debug, Clone, PartialEq)]
pub struct DFlashDraftModel {
    pub config: DFlashConfig,
    /// FC fusion layer: projects concatenated target_hidden_states to hidden_size.
    pub fc: F32Weight,
    pub fc_bias: Vec<f32>,
    /// Hidden norm after fusion.
    pub hidden_norm: Vec<f32>,
    /// Decoder layers.
    pub layers: Vec<DFlashDecoderLayer>,
    /// Final norm.
    pub norm: Vec<f32>,
    /// Output projection (lm_head).
    pub output: F32Weight,
    /// Token embeddings (shared with output or separate).
    pub tok_embeddings: F32Weight,
    /// Contiguous KV cache for each layer, laid out as [seq_len][kv_heads * head_dim].
    pub kv_cache: Vec<DFlashKvLayerCache>,
    pub target_hidden_cache: Vec<Vec<f32>>,
    /// Position offset for RoPE.
    pub position_offset: usize,
}

impl DFlashDraftModel {
    pub fn new(config: DFlashConfig) -> Self {
        let layers = vec![];
        Self {
            config: config.clone(),
            fc: F32Weight::from_slice(Vec::new(), 0, 0),
            fc_bias: Vec::new(),
            hidden_norm: Vec::new(),
            layers,
            norm: Vec::new(),
            output: F32Weight::from_slice(Vec::new(), 0, 0),
            tok_embeddings: F32Weight::from_slice(Vec::new(), 0, 0),
            kv_cache: vec![DFlashKvLayerCache::new(); config.num_hidden_layers],
            target_hidden_cache: Vec::new(),
            position_offset: 0,
        }
    }

    pub fn cache_target_hidden(&mut self, hidden: Vec<f32>) -> Result<(), String> {
        if hidden.len() != self.config.target_hidden_width() {
            return Err(format!(
                "target hidden width mismatch: expected {}, actual {}",
                self.config.target_hidden_width(),
                hidden.len()
            ));
        }
        self.target_hidden_cache.push(hidden);
        Ok(())
    }

    pub fn clear_speculative_caches(&mut self) {
        for layer_cache in &mut self.kv_cache {
            layer_cache.clear();
        }
        self.target_hidden_cache.clear();
        self.position_offset = 0;
    }

    /// Load DFlash draft model from a mapped SafeTensors file.
    pub fn load_from_safetensors(
        mapped: &MappedSafeTensorsFile,
        config: DFlashConfig,
    ) -> Result<Self, String> {
        let mut model = Self::new(config.clone());

        let load_f32 = |name: &str| -> Result<Option<Vec<f32>>, String> {
            let info = match mapped.tensors().iter().find(|t| t.name == name) {
                Some(i) => i,
                None => return Ok(None),
            };
            let data = mapped
                .tensor_data(name)
                .ok_or_else(|| format!("tensor data missing: {}", name))?;
            let count = info.shape.iter().product::<usize>();
            let mut f32_data = vec![0.0_f32; count];
            match info.dtype {
                DType::F32 => {
                    if data.len() < count * 4 {
                        return Err(format!("tensor {} has insufficient bytes", name));
                    }
                    for (i, value) in f32_data.iter_mut().enumerate().take(count) {
                        let bytes = &data[i * 4..(i + 1) * 4];
                        *value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    }
                }
                DType::F16 => {
                    if data.len() < count * 2 {
                        return Err(format!("tensor {} has insufficient bytes", name));
                    }
                    for (i, value) in f32_data.iter_mut().enumerate().take(count) {
                        *value = f16_le_to_f32([data[i * 2], data[i * 2 + 1]]);
                    }
                }
                dtype => {
                    return Err(format!(
                        "tensor {} uses unsupported dtype {:?}",
                        name, dtype
                    ));
                }
            }
            Ok(Some(f32_data))
        };

        // Load FC fusion layer.
        if let Some(data) = load_f32("model.fc.weight")? {
            let rows = config.hidden_size;
            let cols = data.len() / rows;
            model.fc = F32Weight::from_slice(data, rows, cols);
        }
        if let Some(data) = load_f32("model.fc.bias")? {
            model.fc_bias = data;
        }

        // Load hidden norm.
        if let Some(data) = load_f32("model.hidden_norm.weight")? {
            model.hidden_norm = data;
        }

        // Load final norm.
        if let Some(data) = load_f32("model.norm.weight")? {
            model.norm = data;
        }

        // Load output / lm_head.
        if let Some(data) = load_f32("lm_head.weight")? {
            let rows = config.vocab_size;
            let cols = data.len() / rows;
            model.output = F32Weight::from_slice(data, rows, cols);
        }

        // Load token embeddings.
        if let Some(data) = load_f32("model.embed_tokens.weight")? {
            let rows = config.vocab_size;
            let cols = data.len() / rows;
            model.tok_embeddings = F32Weight::from_slice(data, rows, cols);
        }

        // Load layers.
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", layer_idx);

            let input_layernorm = load_f32(&format!("{}.input_layernorm.weight", prefix))?
                .unwrap_or_else(|| vec![1.0_f32; config.hidden_size]);
            let post_attention_layernorm =
                load_f32(&format!("{}.post_attention_layernorm.weight", prefix))?
                    .unwrap_or_else(|| vec![1.0_f32; config.hidden_size]);

            let q_data =
                load_f32(&format!("{}.self_attn.q_proj.weight", prefix))?.unwrap_or_default();
            let q_proj = if q_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.hidden_size;
                let cols = q_data.len() / rows;
                F32Weight::from_slice(q_data, rows, cols)
            };

            let k_data =
                load_f32(&format!("{}.self_attn.k_proj.weight", prefix))?.unwrap_or_default();
            let k_proj = if k_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.num_key_value_heads * config.kv_head_dim();
                let cols = k_data.len() / rows;
                F32Weight::from_slice(k_data, rows, cols)
            };

            let v_data =
                load_f32(&format!("{}.self_attn.v_proj.weight", prefix))?.unwrap_or_default();
            let v_proj = if v_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.num_key_value_heads * config.kv_head_dim();
                let cols = v_data.len() / rows;
                F32Weight::from_slice(v_data, rows, cols)
            };

            let o_data =
                load_f32(&format!("{}.self_attn.o_proj.weight", prefix))?.unwrap_or_default();
            let o_proj = if o_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.hidden_size;
                let cols = o_data.len() / rows;
                F32Weight::from_slice(o_data, rows, cols)
            };

            let q_norm = load_f32(&format!("{}.self_attn.q_norm.weight", prefix))?
                .unwrap_or_else(|| vec![1.0_f32; config.head_dim()]);
            let k_norm = load_f32(&format!("{}.self_attn.k_norm.weight", prefix))?
                .unwrap_or_else(|| vec![1.0_f32; config.head_dim()]);

            let gate_data =
                load_f32(&format!("{}.mlp.gate_proj.weight", prefix))?.unwrap_or_default();
            let mlp_gate = if gate_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.intermediate_size;
                let cols = gate_data.len() / rows;
                F32Weight::from_slice(gate_data, rows, cols)
            };

            let up_data = load_f32(&format!("{}.mlp.up_proj.weight", prefix))?.unwrap_or_default();
            let mlp_up = if up_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.intermediate_size;
                let cols = up_data.len() / rows;
                F32Weight::from_slice(up_data, rows, cols)
            };

            let down_data =
                load_f32(&format!("{}.mlp.down_proj.weight", prefix))?.unwrap_or_default();
            let mlp_down = if down_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.hidden_size;
                let cols = down_data.len() / rows;
                F32Weight::from_slice(down_data, rows, cols)
            };

            let attention = DFlashAttentionLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_weight: q_norm,
                k_norm_weight: k_norm,
            };

            let layer = DFlashDecoderLayer {
                input_layernorm,
                attention,
                post_attention_layernorm,
                mlp_gate,
                mlp_up,
                mlp_down,
            };
            model.layers.push(layer);
        }

        Ok(model)
    }

    /// Load DFlash draft model from a mapped GGUF file (llama.cpp format).
    pub fn load_from_gguf(mapped: &MappedGgufFile, config: DFlashConfig) -> Result<Self, String> {
        let mut model = Self::new(config.clone());

        let tensor_infos = mapped.mapped_tensor_infos();

        // GGUF stores weight matrices in row-major with dims [output, input].
        // gemv_f32_transposed expects column-major (transposed) layout with
        // rows = input_len, cols = output_len, so we need to transpose.
        fn transpose_f32(data: &[f32], gguf_rows: usize, gguf_cols: usize) -> Vec<f32> {
            let mut result = vec![0.0f32; data.len()];
            for r in 0..gguf_rows {
                for c in 0..gguf_cols {
                    result[c * gguf_rows + r] = data[r * gguf_cols + c];
                }
            }
            result
        }

        type LoadF32Result = Result<Option<(Vec<f32>, Vec<u64>)>, String>;
        let load_f32_with_dims = |name: &str| -> LoadF32Result {
            let info = match tensor_infos.iter().find(|t| t.name == name) {
                Some(i) => i,
                None => return Ok(None),
            };
            let qtype = GgufQuantizationType::from_ggml_type(info.ggml_type);
            let value_count: usize = info.dimensions.iter().map(|&d| d as usize).product();
            let qsize = quantized_size(qtype, value_count)
                .map_err(|e| format!("quantized_size for {}: {:?}", name, e))?;
            let offset = info.absolute_offset as usize;
            let end = offset.checked_add(qsize).ok_or_else(|| {
                format!("tensor {name}: offset overflow (offset={offset}, size={qsize})")
            })?;
            let bytes = mapped.bytes();
            if end > bytes.len() {
                return Err(format!(
                    "tensor {name}: data out of bounds (offset={offset}, size={qsize}, file_len={})",
                    bytes.len()
                ));
            }
            let qdata = &bytes[offset..end];
            let mut f32_data = vec![0.0_f32; value_count];
            dequantize_scalar(qtype, qdata, &mut f32_data)
                .map_err(|e| format!("dequantize_scalar for {}: {:?}", name, e))?;
            Ok(Some((f32_data, info.dimensions.clone())))
        };

        // Load a projection weight, preferring the on-the-fly quantized GEMV
        // path (8x less memory traffic during decode) and falling back to a
        // dequantized f32 transpose for unsupported quant types / shapes.
        let bytes_all = mapped.bytes();
        let load_proj = |name: &str| -> Result<F32Weight, String> {
            let info = match tensor_infos.iter().find(|t| t.name == name) {
                Some(i) => i,
                None => return Ok(F32Weight::from_slice(Vec::new(), 0, 0)),
            };
            if info.dimensions.len() != 2 {
                // Fall back to the f32 path for non-2D tensors.
                return load_f32_with_dims(name).map(|opt| match opt {
                    Some((data, dims)) => match dims.len() {
                        0 => F32Weight::from_slice(Vec::new(), 0, 0),
                        1 => {
                            let n = dims[0] as usize;
                            F32Weight::from_slice(data, n, 1)
                        }
                        _ => {
                            let r = dims[0] as usize;
                            let c = dims[1] as usize;
                            F32Weight::from_slice(transpose_f32(&data, r, c), c, r)
                        }
                    },
                    None => F32Weight::from_slice(Vec::new(), 0, 0),
                });
            }
            let qtype = GgufQuantizationType::from_ggml_type(info.ggml_type);
            // GGUF stores the innermost dim (dims[0]) as the input/contiguous
            // axis and dims[1] as the output rows, so the native row-major
            // bytes are exactly `out_dim` rows of `in_dim` quantized elements —
            // precisely the layout the fused GEMV kernel consumes.
            let in_dim = info.dimensions[0] as usize;
            let out_dim = info.dimensions[1] as usize;

            if quantized_gemv_supported(qtype, in_dim) {
                let value_count = out_dim * in_dim;
                let qsize = quantized_size(qtype, value_count)
                    .map_err(|e| format!("quantized_size for {}: {:?}", name, e))?;
                let offset = info.absolute_offset as usize;
                let end = offset.checked_add(qsize).ok_or_else(|| {
                    format!("tensor {name}: offset overflow (offset={offset}, size={qsize})")
                })?;
                if end > bytes_all.len() {
                    return Err(format!(
                        "tensor {name}: data out of bounds (offset={offset}, size={qsize}, file_len={})",
                        bytes_all.len()
                    ));
                }
                return Ok(F32Weight::from_quantized(
                    bytes_all[offset..end].to_vec(),
                    qtype,
                    out_dim,
                    in_dim,
                ));
            }

            // Fallback: dequantize to f32 and store in transposed layout
            // (matches the original f32 load: transpose [dims0,dims1] then
            // store rows = dims1, cols = dims0).
            match load_f32_with_dims(name)? {
                Some((data, _)) => Ok(F32Weight::from_slice(
                    transpose_f32(&data, in_dim, out_dim),
                    out_dim,
                    in_dim,
                )),
                None => Ok(F32Weight::from_slice(Vec::new(), 0, 0)),
            }
        };

        // Load FC fusion layer (merges target model hidden states with draft hidden).
        let fc = load_proj("dflash_fc.weight")?;
        if fc.is_loaded() {
            model.fc = fc;
        }

        // Load hidden norm.
        if let Some((data, _)) = load_f32_with_dims("dflash_hidden_norm.weight")? {
            model.hidden_norm = data;
        }

        // Load final norm (output_norm in GGUF).
        if let Some((data, _)) = load_f32_with_dims("output_norm.weight")? {
            model.norm = data;
        }

        // Load layers using llama.cpp blk.N naming.
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("blk.{}", layer_idx);

            let input_layernorm = load_f32_with_dims(&format!("{}.attn_norm.weight", prefix))?
                .map(|(d, _)| d)
                .unwrap_or_else(|| vec![1.0_f32; config.hidden_size]);
            let post_attention_layernorm =
                load_f32_with_dims(&format!("{}.post_attention_norm.weight", prefix))?
                    .map(|(d, _)| d)
                    .unwrap_or_else(|| vec![1.0_f32; config.hidden_size]);

            let q_proj = load_proj(&format!("{}.attn_q.weight", prefix))?;
            let k_proj = load_proj(&format!("{}.attn_k.weight", prefix))?;
            let v_proj = load_proj(&format!("{}.attn_v.weight", prefix))?;
            let o_proj = load_proj(&format!("{}.attn_output.weight", prefix))?;

            let q_norm = load_f32_with_dims(&format!("{}.attn_q_norm.weight", prefix))?
                .map(|(d, _)| d)
                .unwrap_or_else(|| vec![1.0_f32; config.head_dim()]);
            let k_norm = load_f32_with_dims(&format!("{}.attn_k_norm.weight", prefix))?
                .map(|(d, _)| d)
                .unwrap_or_else(|| vec![1.0_f32; config.head_dim()]);

            let mlp_gate = load_proj(&format!("{}.ffn_gate.weight", prefix))?;
            let mlp_up = load_proj(&format!("{}.ffn_up.weight", prefix))?;
            let mlp_down = load_proj(&format!("{}.ffn_down.weight", prefix))?;

            let attention = DFlashAttentionLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_weight: q_norm,
                k_norm_weight: k_norm,
            };

            let layer = DFlashDecoderLayer {
                input_layernorm,
                attention,
                post_attention_layernorm,
                mlp_gate,
                mlp_up,
                mlp_down,
            };
            model.layers.push(layer);
        }

        Ok(model)
    }

    /// Forward pass for a single token with target_hidden fusion.
    ///
    /// `token` - the input token id.
    /// `target_hidden` - hidden states from target model layers (concatenated).
    pub fn forward_token(
        &mut self,
        token: u32,
        target_hidden: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let h = self.config.hidden_size;
        let mut hidden = vec![0.0_f32; h];

        // Token embedding lookup.
        if !self.tok_embeddings.data.is_empty() {
            let idx = (token as usize).min(self.config.vocab_size - 1);
            let emb = &self.tok_embeddings.data[idx * h..(idx + 1) * h];
            hidden.copy_from_slice(emb);
        }

        // If target_hidden provided, fuse via FC layer.
        if let Some(th) = target_hidden
            && self.fc.is_loaded()
        {
            let mut fused = vec![0.0_f32; h];
            self.fc.gemv(th, &mut fused)?;
            if !self.fc_bias.is_empty() {
                for (i, fused_i) in fused.iter_mut().enumerate().take(h) {
                    *fused_i += self.fc_bias[i];
                }
            }
            // Add residual: noise_embedding + fc(target_hidden)
            for (hidden_i, fused_i) in hidden.iter_mut().zip(fused.iter()).take(h) {
                *hidden_i += *fused_i;
            }
        }

        // Hidden norm.
        if !self.hidden_norm.is_empty() {
            let mut normed_hidden = hidden.clone();
            rms_norm_f32(
                &hidden,
                &self.hidden_norm,
                self.config.rms_norm_eps,
                &mut normed_hidden,
            )
            .map_err(|e| format!("rms_norm: {:?}", e))?;
            hidden = normed_hidden;
        }

        // Layer loop.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Determine actual head_dim from q_norm weight shape (may differ from hidden_size/num_heads).
            let head_dim = if !layer.attention.q_norm_weight.is_empty() {
                layer.attention.q_norm_weight.len()
            } else {
                self.config.head_dim()
            };
            let num_heads = self.config.num_attention_heads;
            let num_kv_heads = self.config.num_key_value_heads;
            let q_size = num_heads * head_dim;
            let kv_len = num_kv_heads * head_dim;

            let mut attn_out = vec![0.0_f32; q_size];
            let mut mlp_out = vec![0.0_f32; h];

            // Attention branch.
            {
                let mut normed = hidden.clone();
                rms_norm_f32(
                    &hidden,
                    &layer.input_layernorm,
                    self.config.rms_norm_eps,
                    &mut normed,
                )
                .map_err(|e| format!("rms_norm: {:?}", e))?;

                // Q projection (on noise/normed hidden only).
                let mut q = vec![0.0_f32; q_size];
                if layer.attention.q_proj.is_loaded() {
                    layer.attention.q_proj.gemv(&normed, &mut q)?;
                }

                // K/V projection on concatenated target_hidden + normed hidden.
                let kv_input = if let Some(th) = target_hidden {
                    let mut concat = th.to_vec();
                    concat.extend_from_slice(&normed);
                    concat
                } else {
                    normed.clone()
                };

                let mut k = vec![0.0_f32; kv_len];
                let mut v = vec![0.0_f32; kv_len];
                if layer.attention.k_proj.is_loaded() {
                    layer.attention.k_proj.gemv(&kv_input, &mut k)?;
                }
                if layer.attention.v_proj.is_loaded() {
                    layer.attention.v_proj.gemv(&kv_input, &mut v)?;
                }

                let pos = self.position_offset;
                let mut head_scratch = vec![0.0_f32; head_dim];

                if !layer.attention.q_norm_weight.is_empty()
                    && layer.attention.q_norm_weight.len() == head_dim
                {
                    for h_idx in 0..num_heads {
                        let start = h_idx * head_dim;
                        let end = start + head_dim;
                        head_scratch.fill(0.0_f32);
                        rms_norm_f32(
                            &q[start..end],
                            &layer.attention.q_norm_weight,
                            self.config.rms_norm_eps,
                            &mut head_scratch,
                        )
                        .map_err(|e| format!("q_norm: {:?}", e))?;
                        q[start..end].copy_from_slice(&head_scratch);
                    }
                }
                if !layer.attention.k_norm_weight.is_empty()
                    && layer.attention.k_norm_weight.len() == head_dim
                {
                    for kv_h in 0..num_kv_heads {
                        let start = kv_h * head_dim;
                        let end = start + head_dim;
                        head_scratch.fill(0.0_f32);
                        rms_norm_f32(
                            &k[start..end],
                            &layer.attention.k_norm_weight,
                            self.config.rms_norm_eps,
                            &mut head_scratch,
                        )
                        .map_err(|e| format!("k_norm: {:?}", e))?;
                        k[start..end].copy_from_slice(&head_scratch);
                    }
                }

                for h_idx in 0..num_heads {
                    let start = h_idx * head_dim;
                    head_scratch.fill(0.0_f32);
                    apply_rope_f32(
                        &q[start..start + head_dim],
                        pos,
                        head_dim,
                        self.config.rope_theta,
                        &mut head_scratch,
                    )
                    .map_err(|e| format!("rope q: {:?}", e))?;
                    q[start..start + head_dim].copy_from_slice(&head_scratch);
                }
                for kv_h in 0..num_kv_heads {
                    let start = kv_h * head_dim;
                    head_scratch.fill(0.0_f32);
                    apply_rope_f32(
                        &k[start..start + head_dim],
                        pos,
                        head_dim,
                        self.config.rope_theta,
                        &mut head_scratch,
                    )
                    .map_err(|e| format!("rope k: {:?}", e))?;
                    k[start..start + head_dim].copy_from_slice(&head_scratch);
                }

                let layer_cache = &mut self.kv_cache[layer_idx];
                layer_cache.keys.extend_from_slice(&k);
                layer_cache.values.extend_from_slice(&v);
                layer_cache.seq_len += 1;

                flash_attention_decode_heads_f32(
                    &q,
                    &layer_cache.keys,
                    &layer_cache.values,
                    layer_cache.seq_len,
                    head_dim,
                    kv_len,
                    num_heads,
                    num_kv_heads,
                    &mut attn_out,
                )
                .map_err(|e| format!("flash attention decode: {:?}", e))?;

                // O projection: attn_out[q_size] -> hidden[h]
                let mut o_result = vec![0.0_f32; h];
                if layer.attention.o_proj.is_loaded() {
                    layer.attention.o_proj.gemv(&attn_out, &mut o_result)?;
                } else {
                    o_result.copy_from_slice(&attn_out);
                }
                attn_out = o_result;
            }

            // Residual connection after attention.
            for i in 0..h {
                hidden[i] += attn_out[i];
            }

            // MLP branch.
            {
                let mut normed = hidden.clone();
                rms_norm_f32(
                    &hidden,
                    &layer.post_attention_layernorm,
                    self.config.rms_norm_eps,
                    &mut normed,
                )
                .map_err(|e| format!("rms_norm: {:?}", e))?;

                let mut gate = vec![0.0_f32; self.config.intermediate_size];
                let mut up = vec![0.0_f32; self.config.intermediate_size];
                if layer.mlp_gate.is_loaded() {
                    layer.mlp_gate.gemv(&normed, &mut gate)?;
                }
                if layer.mlp_up.is_loaded() {
                    layer.mlp_up.gemv(&normed, &mut up)?;
                }

                // SwiGLU: gate * sigmoid(gate) * up
                for i in 0..self.config.intermediate_size {
                    let g = gate[i];
                    gate[i] = g * (1.0 / (1.0 + (-g).exp())) * up[i];
                }

                if layer.mlp_down.is_loaded() {
                    layer.mlp_down.gemv(&gate, &mut mlp_out)?;
                }
            }

            // Residual connection after MLP.
            for i in 0..h {
                hidden[i] += mlp_out[i];
            }
        }

        // Final norm.
        let mut normed_final = hidden.clone();
        rms_norm_f32(
            &hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut normed_final,
        )
        .map_err(|e| format!("rms_norm: {:?}", e))?;
        hidden = normed_final;

        // Advance the absolute position so the next call sees the correct
        // RoPE base. Without this, speculative-decode loops that repeatedly
        // call `forward_token` would apply position 0 to every token.
        self.position_offset += 1;

        Ok(hidden)
    }

    /// Batched prefill: process `tokens` in one pass with batched projections.
    ///
    /// Returns the hidden state of the last token (sufficient for the next-token
    /// logits). All linear layers are batched matmuls; attention is still
    /// sequential per token because each `q_t` must see the cache populated
    /// through position `t`. Target-hidden fusion is not supported here — use
    /// `forward_token` for speculative-decoding paths.
    pub fn forward_batch(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("empty token batch".into());
        }
        let b = tokens.len();
        if b == 1 {
            // `forward_token` advances `position_offset` itself.
            return self.forward_token(tokens[0], None);
        }

        let h = self.config.hidden_size;

        // Embedding lookup: hidden[b * h] row-major.
        let mut hidden = vec![0.0_f32; b * h];
        if !self.tok_embeddings.data.is_empty() {
            for (t, &token) in tokens.iter().enumerate() {
                let idx = (token as usize).min(self.config.vocab_size - 1);
                let emb = &self.tok_embeddings.data[idx * h..(idx + 1) * h];
                hidden[t * h..(t + 1) * h].copy_from_slice(emb);
            }
        }

        // Hidden norm per token.
        if !self.hidden_norm.is_empty() {
            let src = hidden.clone();
            for t in 0..b {
                rms_norm_f32(
                    &src[t * h..(t + 1) * h],
                    &self.hidden_norm,
                    self.config.rms_norm_eps,
                    &mut hidden[t * h..(t + 1) * h],
                )
                .map_err(|e| format!("rms_norm: {:?}", e))?;
            }
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let head_dim = if !layer.attention.q_norm_weight.is_empty() {
                layer.attention.q_norm_weight.len()
            } else {
                self.config.head_dim()
            };
            let num_heads = self.config.num_attention_heads;
            let num_kv_heads = self.config.num_key_value_heads;
            let q_size = num_heads * head_dim;
            let kv_len = num_kv_heads * head_dim;

            // Attention branch: pre-attn norm per token.
            let mut normed = vec![0.0_f32; b * h];
            for t in 0..b {
                rms_norm_f32(
                    &hidden[t * h..(t + 1) * h],
                    &layer.input_layernorm,
                    self.config.rms_norm_eps,
                    &mut normed[t * h..(t + 1) * h],
                )
                .map_err(|e| format!("rms_norm: {:?}", e))?;
            }

            let mut q_all = vec![0.0_f32; b * q_size];
            if layer.attention.q_proj.is_loaded() {
                layer.attention.q_proj.gemm(&normed, &mut q_all, b)?;
            }

            let mut k_all = vec![0.0_f32; b * kv_len];
            let mut v_all = vec![0.0_f32; b * kv_len];
            if layer.attention.k_proj.is_loaded() {
                layer.attention.k_proj.gemm(&normed, &mut k_all, b)?;
            }
            if layer.attention.v_proj.is_loaded() {
                layer.attention.v_proj.gemm(&normed, &mut v_all, b)?;
            }

            let mut head_scratch = vec![0.0_f32; head_dim];

            if !layer.attention.q_norm_weight.is_empty()
                && layer.attention.q_norm_weight.len() == head_dim
            {
                for t in 0..b {
                    let q = &mut q_all[t * q_size..(t + 1) * q_size];
                    for h_idx in 0..num_heads {
                        let start = h_idx * head_dim;
                        let end = start + head_dim;
                        head_scratch.fill(0.0);
                        rms_norm_f32(
                            &q[start..end],
                            &layer.attention.q_norm_weight,
                            self.config.rms_norm_eps,
                            &mut head_scratch,
                        )
                        .map_err(|e| format!("q_norm: {:?}", e))?;
                        q[start..end].copy_from_slice(&head_scratch);
                    }
                }
            }
            if !layer.attention.k_norm_weight.is_empty()
                && layer.attention.k_norm_weight.len() == head_dim
            {
                for t in 0..b {
                    let k = &mut k_all[t * kv_len..(t + 1) * kv_len];
                    for kv_h in 0..num_kv_heads {
                        let start = kv_h * head_dim;
                        let end = start + head_dim;
                        head_scratch.fill(0.0);
                        rms_norm_f32(
                            &k[start..end],
                            &layer.attention.k_norm_weight,
                            self.config.rms_norm_eps,
                            &mut head_scratch,
                        )
                        .map_err(|e| format!("k_norm: {:?}", e))?;
                        k[start..end].copy_from_slice(&head_scratch);
                    }
                }
            }

            // RoPE per (token, head). Position = position_offset + t.
            for t in 0..b {
                let pos = self.position_offset + t;
                let q = &mut q_all[t * q_size..(t + 1) * q_size];
                let k = &mut k_all[t * kv_len..(t + 1) * kv_len];
                for h_idx in 0..num_heads {
                    let start = h_idx * head_dim;
                    head_scratch.fill(0.0);
                    apply_rope_f32(
                        &q[start..start + head_dim],
                        pos,
                        head_dim,
                        self.config.rope_theta,
                        &mut head_scratch,
                    )
                    .map_err(|e| format!("rope q: {:?}", e))?;
                    q[start..start + head_dim].copy_from_slice(&head_scratch);
                }
                for kv_h in 0..num_kv_heads {
                    let start = kv_h * head_dim;
                    head_scratch.fill(0.0);
                    apply_rope_f32(
                        &k[start..start + head_dim],
                        pos,
                        head_dim,
                        self.config.rope_theta,
                        &mut head_scratch,
                    )
                    .map_err(|e| format!("rope k: {:?}", e))?;
                    k[start..start + head_dim].copy_from_slice(&head_scratch);
                }
            }

            // Append KV and run decode-attention per token (causal correctness).
            let mut attn_pre_o = vec![0.0_f32; b * q_size];
            {
                let layer_cache = &mut self.kv_cache[layer_idx];
                for t in 0..b {
                    let k = &k_all[t * kv_len..(t + 1) * kv_len];
                    let v = &v_all[t * kv_len..(t + 1) * kv_len];
                    layer_cache.keys.extend_from_slice(k);
                    layer_cache.values.extend_from_slice(v);
                    layer_cache.seq_len += 1;

                    let q = &q_all[t * q_size..(t + 1) * q_size];
                    let out = &mut attn_pre_o[t * q_size..(t + 1) * q_size];
                    flash_attention_decode_heads_f32(
                        q,
                        &layer_cache.keys,
                        &layer_cache.values,
                        layer_cache.seq_len,
                        head_dim,
                        kv_len,
                        num_heads,
                        num_kv_heads,
                        out,
                    )
                    .map_err(|e| format!("flash attention decode: {:?}", e))?;
                }
            }

            // O projection batched.
            let mut attn_out_all = vec![0.0_f32; b * h];
            if layer.attention.o_proj.is_loaded() {
                layer
                    .attention
                    .o_proj
                    .gemm(&attn_pre_o, &mut attn_out_all, b)?;
            } else if q_size == h {
                attn_out_all.copy_from_slice(&attn_pre_o);
            }

            // Residual after attention.
            for i in 0..b * h {
                hidden[i] += attn_out_all[i];
            }

            // MLP branch.
            let mut normed_mlp = vec![0.0_f32; b * h];
            for t in 0..b {
                rms_norm_f32(
                    &hidden[t * h..(t + 1) * h],
                    &layer.post_attention_layernorm,
                    self.config.rms_norm_eps,
                    &mut normed_mlp[t * h..(t + 1) * h],
                )
                .map_err(|e| format!("rms_norm: {:?}", e))?;
            }

            let inter = self.config.intermediate_size;
            let mut gate = vec![0.0_f32; b * inter];
            let mut up = vec![0.0_f32; b * inter];
            if layer.mlp_gate.is_loaded() {
                layer.mlp_gate.gemm(&normed_mlp, &mut gate, b)?;
            }
            if layer.mlp_up.is_loaded() {
                layer.mlp_up.gemm(&normed_mlp, &mut up, b)?;
            }
            for i in 0..b * inter {
                let g = gate[i];
                gate[i] = g * (1.0 / (1.0 + (-g).exp())) * up[i];
            }
            let mut mlp_out_all = vec![0.0_f32; b * h];
            if layer.mlp_down.is_loaded() {
                layer.mlp_down.gemm(&gate, &mut mlp_out_all, b)?;
            }

            for i in 0..b * h {
                hidden[i] += mlp_out_all[i];
            }
        }

        // Final norm per token.
        let mut out_hidden = vec![0.0_f32; b * h];
        for t in 0..b {
            rms_norm_f32(
                &hidden[t * h..(t + 1) * h],
                &self.norm,
                self.config.rms_norm_eps,
                &mut out_hidden[t * h..(t + 1) * h],
            )
            .map_err(|e| format!("rms_norm: {:?}", e))?;
        }

        self.position_offset += b;
        Ok(out_hidden[(b - 1) * h..b * h].to_vec())
    }

    /// Compute logits from hidden state.
    pub fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        if !self.output.is_loaded() {
            return Ok(Vec::new());
        }
        let mut logits = vec![0.0_f32; self.config.vocab_size];
        self.output.gemv(hidden, &mut logits)?;
        Ok(logits)
    }

    /// Reset KV cache and position offset.
    pub fn reset_cache(&mut self) {
        self.kv_cache = vec![DFlashKvLayerCache::new(); self.config.num_hidden_layers];
        self.position_offset = 0;
    }

    pub fn reserve_cache_tokens(&mut self, tokens: usize) {
        let kv_len = self.config.num_key_value_heads * self.config.head_dim();
        for layer_cache in &mut self.kv_cache {
            layer_cache.reserve_tokens(tokens, kv_len);
        }
    }
}

impl Model for DFlashDraftModel {
    fn forward(&mut self, tokens: &[Token], session: &mut Session) -> Result<Logits, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::EmptyInput);
        }

        // Prefer batched prefill: every linear is computed with a single
        // weight scan amortized over all tokens. Falls back to forward_token
        // for batch=1 (decode).
        let hidden = self
            .forward_batch(tokens)
            .map_err(ModelError::InferenceFailed)?;

        let logits = self.logits(&hidden).map_err(ModelError::InferenceFailed)?;
        session.record_tokens(tokens.len());
        Ok(logits)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn context_size(&self) -> usize {
        4096 // Default context size for DFlash draft
    }

    fn layer_count(&self) -> usize {
        self.config.num_hidden_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dflash_config_defaults() {
        let cfg = DFlashConfig::default();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 8);
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.mask_token_id, 248070);
    }

    #[test]
    fn dflash_model_creation() {
        let cfg = DFlashConfig::default();
        let model = DFlashDraftModel::new(cfg);
        assert_eq!(model.layers.len(), 0);
        assert_eq!(model.config.vocab_size, 248320);
    }
}
