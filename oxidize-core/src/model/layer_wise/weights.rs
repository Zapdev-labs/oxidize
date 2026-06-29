use super::*;

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct LayerWeights {
    pub(super) attn_norm: Vec<f32>,
    pub(super) attn_q: WeightStorage,
    pub(super) attn_q_bias: Vec<f32>,
    pub(super) attn_k: WeightStorage,
    pub(super) attn_k_bias: Vec<f32>,
    pub(super) attn_v: WeightStorage,
    pub(super) attn_v_bias: Vec<f32>,
    pub(super) attn_output: WeightStorage,
    pub(super) attn_output_bias: Vec<f32>,
    pub(super) ffn_norm: Vec<f32>,
    pub(super) post_attention_norm: Vec<f32>,
    pub(super) ffn_gate: WeightStorage,
    pub(super) ffn_up: WeightStorage,
    pub(super) ffn_down: WeightStorage,
    pub(super) ffn_down_bias: Vec<f32>,
    pub(super) ffn_gate_exps: WeightStorage,
    pub(super) ffn_up_exps: WeightStorage,
    pub(super) ffn_down_exps: WeightStorage,
    pub(super) ffn_gate_inp: WeightStorage,
    pub(super) ffn_exp_probs_b: Vec<f32>,
    pub(super) ffn_gate_shexp: WeightStorage,
    pub(super) ffn_gate_inp_shexp: WeightStorage,
    pub(super) ffn_up_shexp: WeightStorage,
    pub(super) ffn_down_shexp: WeightStorage,
    pub(super) attn_qkv: WeightStorage,
    pub(super) attn_gate: WeightStorage,
    pub(super) ssm_a: Vec<f32>,
    pub(super) ssm_alpha: WeightStorage,
    pub(super) ssm_beta: WeightStorage,
    pub(super) ssm_conv1d: Vec<f32>,
    pub(super) ssm_dt_bias: Vec<f32>,
    pub(super) ssm_norm: Vec<f32>,
    pub(super) ssm_out: WeightStorage,
    pub(super) attn_q_norm: Vec<f32>,
    pub(super) attn_k_norm: Vec<f32>,
}

impl LayerWiseModel {
    pub(super) fn load_quant_weight(
        &self,
        qtype: GgufQuantizationType,
        mmap_index: usize,
        offset: usize,
        size: usize,
        qdata: &[u8],
        prefer_mmap: bool,
    ) -> WeightStorage {
        if prefer_mmap {
            let info = GgufTensorInfo {
                name: String::new(),
                dimensions: Vec::new(),
                ggml_type: 0,
                relative_offset: 0,
                absolute_offset: offset as u64,
                mmap_index,
            };
            WeightStorage::MmapQuantized(qtype, self.mmap.tensor_mmap(&info), offset, size)
        } else {
            WeightStorage::Quantized(qtype, qdata.to_vec())
        }
    }

    fn tensor_ref_bytes(&self, tensor_ref: &GgufTensorRef) -> &[u8] {
        let info = GgufTensorInfo {
            name: String::new(),
            dimensions: Vec::new(),
            ggml_type: 0,
            relative_offset: 0,
            absolute_offset: tensor_ref.offset as u64,
            mmap_index: tensor_ref.mmap_index,
        };
        self.mmap.tensor_bytes(&info, tensor_ref.size)
    }

    pub(super) fn load_layer_weights(&self, layer_idx: usize) -> Result<LayerWeights, String> {
        let refs = &self.layer_tensors[layer_idx];
        let mut layer = LayerWeights::default();
        let is_supported_quant_gemv = |qtype: GgufQuantizationType| {
            matches!(
                qtype,
                GgufQuantizationType::Q8_0
                    | GgufQuantizationType::Q4_K_S
                    | GgufQuantizationType::Q4_K_M
                    | GgufQuantizationType::Q6_K
            )
        };
        let prefer_mmap = |key: &str, size: usize| {
            size > 16 * 1024 * 1024
                || key.starts_with("ffn_gate_exps")
                || key.starts_with("ffn_up_exps")
                || key.starts_with("ffn_down_exps")
                || key.starts_with("ffn_gate_inp")
                || key.starts_with("ffn_gate_shexp")
                || key.starts_with("ffn_gate_inp_shexp")
                || key.starts_with("ffn_up_shexp")
                || key.starts_with("ffn_down_shexp")
        };

        for (key, tensor_ref) in refs.iter() {
            let qdata = self.tensor_ref_bytes(tensor_ref);
            let count = tensor_ref.value_count;
            let qtype = tensor_ref.qtype;
            let weight_key = key.as_str();
            let mmap_this = prefer_mmap(weight_key, tensor_ref.size);
            let is_ssm_vec = matches!(weight_key, "ssm_a.weight" | "ssm_conv1d.weight");

            if weight_key.ends_with(".weight")
                && !weight_key.contains("norm")
                && !weight_key.contains("bias")
                && !is_ssm_vec
                && is_supported_quant_gemv(qtype)
            {
                let ws = self.load_quant_weight(
                    qtype,
                    tensor_ref.mmap_index,
                    tensor_ref.offset,
                    tensor_ref.size,
                    qdata,
                    mmap_this,
                );
                match weight_key {
                    "attn_q.weight" => layer.attn_q = ws,
                    "attn_k.weight" => layer.attn_k = ws,
                    "attn_v.weight" => layer.attn_v = ws,
                    "attn_output.weight" => layer.attn_output = ws,
                    "ffn_gate.weight" => layer.ffn_gate = ws,
                    "ffn_up.weight" => layer.ffn_up = ws,
                    "ffn_down.weight" => layer.ffn_down = ws,
                    "ffn_gate_exps.weight" => layer.ffn_gate_exps = ws,
                    "ffn_up_exps.weight" => layer.ffn_up_exps = ws,
                    "ffn_down_exps.weight" => layer.ffn_down_exps = ws,
                    "ffn_gate_inp.weight" => layer.ffn_gate_inp = ws,
                    "ffn_gate_inp_shexp.weight" => layer.ffn_gate_inp_shexp = ws,
                    "ffn_gate_shexp.weight" => layer.ffn_gate_shexp = ws,
                    "ffn_up_shexp.weight" => layer.ffn_up_shexp = ws,
                    "ffn_down_shexp.weight" => layer.ffn_down_shexp = ws,
                    "attn_qkv.weight" => layer.attn_qkv = ws,
                    "attn_gate.weight" => layer.attn_gate = ws,
                    "ssm_alpha.weight" => layer.ssm_alpha = ws,
                    "ssm_beta.weight" => layer.ssm_beta = ws,
                    "ssm_out.weight" => layer.ssm_out = ws,
                    _ => {}
                }
            } else {
                let mut v = vec![0.0_f32; count];
                dequantize_scalar(qtype, qdata, &mut v)
                    .map_err(|e| format!("dequantize: {:?}", e))?;
                match weight_key {
                    "attn_norm.weight" => layer.attn_norm = v,
                    "attn_q.weight" => layer.attn_q = WeightStorage::F32(v),
                    "attn_q.bias" => layer.attn_q_bias = v,
                    "attn_k.weight" => layer.attn_k = WeightStorage::F32(v),
                    "attn_k.bias" => layer.attn_k_bias = v,
                    "attn_v.weight" => layer.attn_v = WeightStorage::F32(v),
                    "attn_v.bias" => layer.attn_v_bias = v,
                    "attn_output.weight" => layer.attn_output = WeightStorage::F32(v),
                    "attn_output.bias" => layer.attn_output_bias = v,
                    "ffn_norm.weight" => layer.ffn_norm = v,
                    "post_attention_norm.weight" => layer.post_attention_norm = v,
                    "ffn_gate.weight" => layer.ffn_gate = WeightStorage::F32(v),
                    "ffn_up.weight" => layer.ffn_up = WeightStorage::F32(v),
                    "ffn_down.weight" => layer.ffn_down = WeightStorage::F32(v),
                    "ffn_down.bias" => layer.ffn_down_bias = v,
                    "ffn_gate_exps.weight" => layer.ffn_gate_exps = WeightStorage::F32(v),
                    "ffn_up_exps.weight" => layer.ffn_up_exps = WeightStorage::F32(v),
                    "ffn_down_exps.weight" => layer.ffn_down_exps = WeightStorage::F32(v),
                    "ffn_gate_inp.weight" => layer.ffn_gate_inp = WeightStorage::F32(v),
                    "ffn_gate_inp_shexp.weight" => layer.ffn_gate_inp_shexp = WeightStorage::F32(v),
                    "ffn_gate_shexp.weight" => layer.ffn_gate_shexp = WeightStorage::F32(v),
                    "ffn_up_shexp.weight" => layer.ffn_up_shexp = WeightStorage::F32(v),
                    "ffn_down_shexp.weight" => layer.ffn_down_shexp = WeightStorage::F32(v),
                    "ffn_exp_probs_b.bias" => layer.ffn_exp_probs_b = v,
                    "attn_qkv.weight" => layer.attn_qkv = WeightStorage::F32(v),
                    "attn_gate.weight" => layer.attn_gate = WeightStorage::F32(v),
                    "ssm_a.weight" => layer.ssm_a = v,
                    "ssm_alpha.weight" => layer.ssm_alpha = WeightStorage::F32(v),
                    "ssm_beta.weight" => layer.ssm_beta = WeightStorage::F32(v),
                    "ssm_conv1d.weight" => layer.ssm_conv1d = v,
                    "ssm_dt.bias" => layer.ssm_dt_bias = v,
                    "ssm_norm.weight" => layer.ssm_norm = v,
                    "ssm_out.weight" => layer.ssm_out = WeightStorage::F32(v),
                    "attn_q_norm.weight" => layer.attn_q_norm = v,
                    "attn_k_norm.weight" => layer.attn_k_norm = v,
                    _ => {}
                }
            }
        }
        Ok(layer)
    }

    pub(super) fn ensure_layer_loaded(&mut self, layer_idx: usize) -> Result<(), String> {
        if self.cache.entries[layer_idx].is_none() {
            let layer = self.load_layer_weights(layer_idx)?;
            self.cache.put(layer_idx, layer);
        }
        self.cache.generation += 1;
        self.cache.access_count[layer_idx] = self.cache.generation;
        Ok(())
    }

    pub(super) fn layer_ref(&self, layer_idx: usize) -> &LayerWeights {
        self.cache.entries[layer_idx]
            .as_ref()
            .expect("layer must be loaded before layer_ref")
    }

    pub(super) fn get_or_load_layer(&mut self, layer_idx: usize) -> Result<LayerWeights, String> {
        if let Some(cached) = self.cache.get(layer_idx) {
            return Ok(cached);
        }
        self.load_layer_weights(layer_idx)
    }

    pub(super) fn return_layer(&mut self, layer_idx: usize, weights: LayerWeights) {
        self.cache.put(layer_idx, weights);
    }
}
