use crate::gguf::{GgufQuantizationType, MappedGgufFile};
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::quantization::{dequantize_scalar, quantized_size};
use crate::safetensors::MappedSafeTensorsFile;
use crate::tensor::{DType, apply_rope_f32, f16_le_to_f32, gemv_f32_transposed, rms_norm_f32};

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
}

/// Simple F32 weight matrix wrapper for clarity.
#[derive(Debug, Clone, PartialEq)]
pub struct F32Weight {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl F32Weight {
    pub fn from_slice(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        Self { data, rows, cols }
    }

    pub fn gemv(&self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        gemv_f32_transposed(&self.data, self.cols, self.rows, input, output)
            .map_err(|e| format!("{:?}", e))
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
    /// KV cache for each layer: (keys, values) per token.
    pub kv_cache: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
    /// Position offset for RoPE.
    pub position_offset: usize,
}

impl DFlashDraftModel {
    pub fn new(config: DFlashConfig) -> Self {
        let h = config.hidden_size;
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
            kv_cache: vec![Vec::new(); config.num_hidden_layers],
            position_offset: 0,
        }
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
            let post_attention_layernorm = load_f32(&format!("{}.post_attention_layernorm.weight", prefix))?
                .unwrap_or_else(|| vec![1.0_f32; config.hidden_size]);

            let q_data = load_f32(&format!("{}.self_attn.q_proj.weight", prefix))?
                .unwrap_or_default();
            let q_proj = if q_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.hidden_size;
                let cols = q_data.len() / rows;
                F32Weight::from_slice(q_data, rows, cols)
            };

            let k_data = load_f32(&format!("{}.self_attn.k_proj.weight", prefix))?
                .unwrap_or_default();
            let k_proj = if k_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.num_key_value_heads * config.kv_head_dim();
                let cols = k_data.len() / rows;
                F32Weight::from_slice(k_data, rows, cols)
            };

            let v_data = load_f32(&format!("{}.self_attn.v_proj.weight", prefix))?
                .unwrap_or_default();
            let v_proj = if v_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.num_key_value_heads * config.kv_head_dim();
                let cols = v_data.len() / rows;
                F32Weight::from_slice(v_data, rows, cols)
            };

            let o_data = load_f32(&format!("{}.self_attn.o_proj.weight", prefix))?
                .unwrap_or_default();
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

            let gate_data = load_f32(&format!("{}.mlp.gate_proj.weight", prefix))?
                .unwrap_or_default();
            let mlp_gate = if gate_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.intermediate_size;
                let cols = gate_data.len() / rows;
                F32Weight::from_slice(gate_data, rows, cols)
            };

            let up_data = load_f32(&format!("{}.mlp.up_proj.weight", prefix))?
                .unwrap_or_default();
            let mlp_up = if up_data.is_empty() {
                F32Weight::from_slice(Vec::new(), 0, 0)
            } else {
                let rows = config.intermediate_size;
                let cols = up_data.len() / rows;
                F32Weight::from_slice(up_data, rows, cols)
            };

            let down_data = load_f32(&format!("{}.mlp.down_proj.weight", prefix))?
                .unwrap_or_default();
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
    pub fn load_from_gguf(
        mapped: &MappedGgufFile,
        config: DFlashConfig,
    ) -> Result<Self, String> {
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

        let load_f32_with_dims = |name: &str| -> Result<Option<(Vec<f32>, Vec<u64>)>, String> {
            let info = match tensor_infos.iter().find(|t| t.name == name) {
                Some(i) => i,
                None => return Ok(None),
            };
            let qtype = GgufQuantizationType::from_ggml_type(info.ggml_type);
            let value_count: usize = info.dimensions.iter().map(|&d| d as usize).product();
            let qsize = quantized_size(qtype, value_count)
                .map_err(|e| format!("quantized_size for {}: {:?}", name, e))?;
            let offset = info.absolute_offset as usize;
            let qdata = &mapped.bytes()[offset..offset + qsize];
            let mut f32_data = vec![0.0_f32; value_count];
            dequantize_scalar(qtype, qdata, &mut f32_data)
                .map_err(|e| format!("dequantize_scalar for {}: {:?}", name, e))?;
            Ok(Some((f32_data, info.dimensions.clone())))
        };

        // Load FC fusion layer (merges target model hidden states with draft hidden).
        if let Some((data, dims)) = load_f32_with_dims("dflash_fc.weight")? {
            let gguf_rows = dims[0] as usize;  // hidden output
            let gguf_cols = dims[1] as usize;  // target features input
            let transposed = transpose_f32(&data, gguf_rows, gguf_cols);
            model.fc = F32Weight::from_slice(transposed, gguf_rows, gguf_cols);
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
            let post_attention_layernorm = load_f32_with_dims(&format!("{}.post_attention_norm.weight", prefix))?
                .map(|(d, _)| d)
                .unwrap_or_else(|| vec![1.0_f32; config.hidden_size]);

            let q_proj = match load_f32_with_dims(&format!("{}.attn_q.weight", prefix))? {
                Some((data, dims)) => {
                    let gguf_rows = dims[0] as usize;
                    let gguf_cols = dims[1] as usize;
                    let transposed = transpose_f32(&data, gguf_rows, gguf_cols);
                    F32Weight::from_slice(transposed, gguf_cols, gguf_rows)
                }
                None => F32Weight::from_slice(Vec::new(), 0, 0),
            };

            let k_proj = match load_f32_with_dims(&format!("{}.attn_k.weight", prefix))? {
                Some((data, dims)) => {
                    let gguf_rows = dims[0] as usize;
                    let gguf_cols = dims[1] as usize;
                    let transposed = transpose_f32(&data, gguf_rows, gguf_cols);
                    F32Weight::from_slice(transposed, gguf_cols, gguf_rows)
                }
                None => F32Weight::from_slice(Vec::new(), 0, 0),
            };

            let v_proj = match load_f32_with_dims(&format!("{}.attn_v.weight", prefix))? {
                Some((data, dims)) => {
                    let gguf_rows = dims[0] as usize;
                    let gguf_cols = dims[1] as usize;
                    let transposed = transpose_f32(&data, gguf_rows, gguf_cols);
                    F32Weight::from_slice(transposed, gguf_cols, gguf_rows)
                }
                None => F32Weight::from_slice(Vec::new(), 0, 0),
            };

            let o_proj = match load_f32_with_dims(&format!("{}.attn_output.weight", prefix))? {
                Some((data, dims)) => {
                    let gguf_rows = dims[0] as usize;
                    let gguf_cols = dims[1] as usize;
                    let transposed = transpose_f32(&data, gguf_rows, gguf_cols);
                    F32Weight::from_slice(transposed, gguf_cols, gguf_rows)
                }
                None => F32Weight::from_slice(Vec::new(), 0, 0),
            };

            let q_norm = load_f32_with_dims(&format!("{}.attn_q_norm.weight", prefix))?
                .map(|(d, _)| d)
                .unwrap_or_else(|| vec![1.0_f32; config.head_dim()]);
            let k_norm = load_f32_with_dims(&format!("{}.attn_k_norm.weight", prefix))?
                .map(|(d, _)| d)
                .unwrap_or_else(|| vec![1.0_f32; config.head_dim()]);

            let mlp_gate = match load_f32_with_dims(&format!("{}.ffn_gate.weight", prefix))? {
                Some((data, dims)) => {
                    let gguf_rows = dims[0] as usize;
                    let gguf_cols = dims[1] as usize;
                    let transposed = transpose_f32(&data, gguf_rows, gguf_cols);
                    F32Weight::from_slice(transposed, gguf_cols, gguf_rows)
                }
                None => F32Weight::from_slice(Vec::new(), 0, 0),
            };

            let mlp_up = match load_f32_with_dims(&format!("{}.ffn_up.weight", prefix))? {
                Some((data, dims)) => {
                    let gguf_rows = dims[0] as usize;
                    let gguf_cols = dims[1] as usize;
                    let transposed = transpose_f32(&data, gguf_rows, gguf_cols);
                    F32Weight::from_slice(transposed, gguf_cols, gguf_rows)
                }
                None => F32Weight::from_slice(Vec::new(), 0, 0),
            };

            let mlp_down = match load_f32_with_dims(&format!("{}.ffn_down.weight", prefix))? {
                Some((data, dims)) => {
                    let gguf_rows = dims[0] as usize;
                    let gguf_cols = dims[1] as usize;
                    let transposed = transpose_f32(&data, gguf_rows, gguf_cols);
                    F32Weight::from_slice(transposed, gguf_cols, gguf_rows)
                }
                None => F32Weight::from_slice(Vec::new(), 0, 0),
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
        if let Some(th) = target_hidden {
            if !self.fc.data.is_empty() {
                let mut fused = vec![0.0_f32; h];
                self.fc.gemv(th, &mut fused)?;
                if !self.fc_bias.is_empty() {
                    for i in 0..h {
                        fused[i] += self.fc_bias[i];
                    }
                }
                // Add residual: noise_embedding + fc(target_hidden)
                for i in 0..h {
                    hidden[i] += fused[i];
                }
            }
        }

        // Hidden norm.
        if !self.hidden_norm.is_empty() {
            let mut normed_hidden = hidden.clone();
            rms_norm_f32(&hidden, &self.hidden_norm, self.config.rms_norm_eps, &mut normed_hidden).map_err(|e| format!("rms_norm: {:?}", e))?;
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
            let queries_per_kv = num_heads / num_kv_heads;

            let mut attn_out = vec![0.0_f32; q_size];
            let mut mlp_out = vec![0.0_f32; h];

            // Attention branch.
            {
                let mut normed = hidden.clone();
                rms_norm_f32(&hidden, &layer.input_layernorm, self.config.rms_norm_eps, &mut normed).map_err(|e| format!("rms_norm: {:?}", e))?;

                // Q projection (on noise/normed hidden only).
                let mut q = vec![0.0_f32; q_size];
                if !layer.attention.q_proj.data.is_empty() {
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
                if !layer.attention.k_proj.data.is_empty() {
                    layer.attention.k_proj.gemv(&kv_input, &mut k)?;
                }
                if !layer.attention.v_proj.data.is_empty() {
                    layer.attention.v_proj.gemv(&kv_input, &mut v)?;
                }

                let pos = self.position_offset + self.kv_cache[layer_idx].len();
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

                self.kv_cache[layer_idx].push((k.clone(), v.clone()));

                // Simple attention: compute scores with all past tokens.
                let seq_len = self.kv_cache[layer_idx].len();

                for h_idx in 0..num_heads {
                    let kv_h = h_idx / queries_per_kv;
                    let q_offset = h_idx * head_dim;
                    let mut attn_weights = vec![0.0_f32; seq_len];

                    for t in 0..seq_len {
                        let (ref k_cache, _) = self.kv_cache[layer_idx][t];
                        let k_offset = kv_h * head_dim;
                        let mut score = 0.0_f32;
                        for d in 0..head_dim {
                            score += q[q_offset + d] * k_cache[k_offset + d];
                        }
                        attn_weights[t] = score / (head_dim as f32).sqrt();
                    }

                    // Softmax.
                    let max_score = attn_weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0_f32;
                    for t in 0..seq_len {
                        attn_weights[t] = (attn_weights[t] - max_score).exp();
                        sum_exp += attn_weights[t];
                    }
                    for t in 0..seq_len {
                        attn_weights[t] /= sum_exp;
                    }

                    // Weighted sum of values.
                    let mut out_offset = h_idx * head_dim;
                    for d in 0..head_dim {
                        let mut val = 0.0_f32;
                        for t in 0..seq_len {
                            let (_, ref v_cache) = self.kv_cache[layer_idx][t];
                            let v_offset = kv_h * head_dim;
                            val += attn_weights[t] * v_cache[v_offset + d];
                        }
                        attn_out[out_offset + d] = val;
                    }
                }

                // O projection: attn_out[q_size] -> hidden[h]
                let mut o_result = vec![0.0_f32; h];
                if !layer.attention.o_proj.data.is_empty() {
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
                rms_norm_f32(&hidden, &layer.post_attention_layernorm, self.config.rms_norm_eps, &mut normed).map_err(|e| format!("rms_norm: {:?}", e))?;

                let mut gate = vec![0.0_f32; self.config.intermediate_size];
                let mut up = vec![0.0_f32; self.config.intermediate_size];
                if !layer.mlp_gate.data.is_empty() {
                    layer.mlp_gate.gemv(&normed, &mut gate)?;
                }
                if !layer.mlp_up.data.is_empty() {
                    layer.mlp_up.gemv(&normed, &mut up)?;
                }

                // SwiGLU: gate * sigmoid(gate) * up
                for i in 0..self.config.intermediate_size {
                    let g = gate[i];
                    gate[i] = g * (1.0 / (1.0 + (-g).exp())) * up[i];
                }

                if !layer.mlp_down.data.is_empty() {
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
        rms_norm_f32(&hidden, &self.norm, self.config.rms_norm_eps, &mut normed_final).map_err(|e| format!("rms_norm: {:?}", e))?;
        hidden = normed_final;

        Ok(hidden)
    }

    /// Compute logits from hidden state.
    pub fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        let mut logits = vec![0.0_f32; self.config.vocab_size];
        if !self.output.data.is_empty() {
            self.output.gemv(hidden, &mut logits)?;
        }
        Ok(logits)
    }

    /// Reset KV cache and position offset.
    pub fn reset_cache(&mut self) {
        self.kv_cache = vec![Vec::new(); self.config.num_hidden_layers];
        self.position_offset = 0;
    }
}

impl Model for DFlashDraftModel {
    fn forward(&mut self, tokens: &[Token], session: &mut Session) -> Result<Logits, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::EmptyInput);
        }

        let mut hidden = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            // For simple draft-only forward, no target_hidden fusion.
            // In speculative decoding, forward_token with target_hidden will be called directly.
            hidden = self
                .forward_token(token, None)
                .map_err(|e| ModelError::InferenceFailed(e))?;
            self.position_offset += 1;
        }

        let logits = self
            .logits(&hidden)
            .map_err(|e| ModelError::InferenceFailed(e))?;
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
