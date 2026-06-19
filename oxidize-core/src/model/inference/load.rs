use super::*;

impl InferenceModel {
    pub fn load_from_gguf(
        mapped: &MappedGgufFile,
        mut config: InferenceConfig,
        use_mmap: bool,
    ) -> Result<Self, String> {
        // Architecture-aware configuration
        config.architecture = ModelArchitecture::from_gguf(mapped);
        if config.alibi_num_heads == 0 {
            config.alibi_num_heads = config.num_attention_heads;
        }
        let mut tok_embeddings: Option<WeightStorage> = None;
        let mut tok_embeddings_cols: usize = config.hidden_size;
        let mut norm_weight: Option<Vec<f32>> = None;
        let mut output_weight: Option<WeightStorage> = None;
        let mut layers: Vec<LayerWeights> = vec![LayerWeights::default(); config.layer_count];
        let mut mtp: Option<MtpWeights> =
            (config.nextn_predict_layers > 0).then(MtpWeights::default);
        let use_mmap_flag = use_mmap;

        let tensor_list = mapped.mapped_tensor_infos();
        for tensor in tensor_list.iter() {
            let qtype = GgufQuantizationType::from_ggml_type(tensor.ggml_type);
            let value_count: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
            let qsize = quantized_size(qtype, value_count)
                .map_err(|e| format!("quantized_size: {:?}", e))?;
            let offset = tensor.absolute_offset as usize;
            let qdata = mapped.tensor_bytes(tensor, qsize);

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
                        | GgufQuantizationType::IQ1_S
                        | GgufQuantizationType::IQ1_M
                        | GgufQuantizationType::NVFP4
                );
                if should_keep_quantized(name) && is_supported_quant_gemv {
                    if use_mmap_flag {
                        Ok(WeightStorage::MmapQuantized(
                            qtype,
                            mapped.tensor_mmap(tensor),
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
                // LFM2 has no separate output_norm; token_embd_norm is the final norm.
                "norm.weight" | "output_norm.weight" | "token_embd_norm.weight" => {
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
                        if let Some(mtp) = mtp.as_mut()
                            && layer_idx == config.layer_count
                        {
                            if parts.get(2) == Some(&"nextn") {
                                let nextn_name = parts.get(3).copied().unwrap_or("");
                                let nextn_suffix = parts.get(4).copied();
                                match (nextn_name, nextn_suffix) {
                                    ("eh_proj", Some("weight")) => {
                                        mtp.eh_proj = load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("enorm", Some("weight")) | ("enorm", None) => {
                                        mtp.enorm = load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("hnorm", Some("weight")) | ("hnorm", None) => {
                                        mtp.hnorm = load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("embed_tokens", Some("weight")) => {
                                        mtp.embed_tokens =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("shared_head_norm", Some("weight"))
                                    | ("shared_head_norm", None) => {
                                        mtp.shared_head_norm =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("shared_head_head", Some("weight"))
                                    | ("shared_head", Some("weight")) => {
                                        mtp.shared_head_head =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    _ => {}
                                }
                            } else {
                                let weight_name = parts[2];
                                let suffix = parts.get(3).copied();
                                match (weight_name, suffix) {
                                    ("attn_norm", _) => {
                                        mtp.layer.attn_norm = load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_q", Some("weight")) => {
                                        mtp.layer.attn_q =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("attn_q", Some("bias")) => {
                                        mtp.layer.attn_q_bias =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_k", Some("weight")) => {
                                        mtp.layer.attn_k =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("attn_k", Some("bias")) => {
                                        mtp.layer.attn_k_bias =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_v", Some("weight")) => {
                                        mtp.layer.attn_v =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("attn_v", Some("bias")) => {
                                        mtp.layer.attn_v_bias =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_output", Some("weight")) => {
                                        mtp.layer.attn_output =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("attn_output", Some("bias")) => {
                                        mtp.layer.attn_output_bias =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_q_norm", _) => {
                                        mtp.layer.attn_q_norm =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_k_norm", _) => {
                                        mtp.layer.attn_k_norm =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("ffn_norm", _) | ("post_attention_norm", _) => {
                                        mtp.layer.post_attention_norm =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("ffn_gate", _) => {
                                        mtp.layer.ffn_gate =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("ffn_up", _) => {
                                        mtp.layer.ffn_up =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("ffn_down", Some("weight")) => {
                                        mtp.layer.ffn_down =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("ffn_down", Some("bias")) => {
                                        mtp.layer.ffn_down_bias =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    _ => {}
                                }
                            }
                        }
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
                        ("post_ffw_norm", _) | ("post_ffn_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].post_ffn_norm = f32_data;
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
                        // MoE expert weights
                        ("ffn_gate_exps", _) => {
                            layers[layer_idx].ffn_gate_exps =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_up_exps", _) => {
                            layers[layer_idx].ffn_up_exps =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_down_exps", _) => {
                            layers[layer_idx].ffn_down_exps =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_gate_inp", _) => {
                            layers[layer_idx].ffn_gate_inp =
                                load_tensor(name, qtype, qdata, value_count)?
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
                        // LFM2 short-convolution operator
                        ("shortconv", Some("in_proj")) => {
                            layers[layer_idx].shortconv_in_proj =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("shortconv", Some("out_proj")) => {
                            layers[layer_idx].shortconv_out_proj =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("shortconv", Some("conv")) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].shortconv_conv = f32_data;
                        }
                        // LFM2MoE per-layer expert routing bias
                        ("exp_probs_b", _) => {
                            layers[layer_idx].ffn_exp_probs_b =
                                load_bias(qtype, qdata, value_count)?
                        }
                        ("attn_q_a", Some("weight")) => {
                            layers[layer_idx].mla_q_a =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_q_a_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].mla_q_a_norm = f32_data;
                        }
                        ("attn_q_b", Some("weight")) => {
                            layers[layer_idx].mla_q_b =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_kv_a_mqa", Some("weight")) => {
                            layers[layer_idx].mla_kv_a_mqa =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_kv_a_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].mla_kv_a_norm = f32_data;
                        }
                        ("attn_k_b", Some("weight")) => {
                            layers[layer_idx].mla_k_b =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_v_b", Some("weight")) => {
                            layers[layer_idx].mla_v_b =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_gate_shexp", _) => {
                            layers[layer_idx].ffn_gate_shexp =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_gate_inp_shexp", _) => {
                            layers[layer_idx].ffn_gate_inp_shexp =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_up_shexp", _) => {
                            layers[layer_idx].ffn_up_shexp =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_down_shexp", _) => {
                            layers[layer_idx].ffn_down_shexp =
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
        let mtp = mtp.and_then(|weights| {
            if weights.is_usable(&config) {
                Some(weights)
            } else {
                eprintln!(
                    "MTP metadata advertises {} nextn layer(s), but required blk.{}.nextn/decoder tensors were incomplete; disabling native MTP",
                    config.nextn_predict_layers, config.layer_count
                );
                None
            }
        });

        eprintln!(
            "InferenceConfig: vocab={}, context={}, layers={}, mtp_nextn={}, hidden={}, intermediate={}, heads={}, kv_heads={}, kv_head_dim={}, eps={}, theta={}",
            config.vocab_size,
            config.context_size,
            config.layer_count,
            config.nextn_predict_layers,
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.kv_head_dim(),
            config.rms_norm_eps,
            config.rope_theta
        );

        // Build a map from absolute layer index to KV cache layer index.
        // Only attention layers (those with a non-empty attn_q projection) write to
        // the KV cache.  Shortconv and Mamba layers skip the KV cache entirely, so
        // sizing the cache to all `layer_count` layers wastes memory proportional to
        // the number of non-attention layers (e.g. 18 out of 24 for LFM2MoE, saving
        // ~8.8 GB for a 128k-context model).
        let mut kv_layer_map: Vec<Option<usize>> = Vec::with_capacity(layers.len());
        let mut attn_layer_count: usize = 0;
        for layer in layers.iter().take(config.layer_count) {
            let is_attn = !layer.attn_q.is_empty() || !layer.mla_kv_a_mqa.is_empty();
            if is_attn {
                kv_layer_map.push(Some(attn_layer_count));
                attn_layer_count += 1;
            } else {
                kv_layer_map.push(None);
            }
        }

        let kv_head_count = if config.architecture.uses_mla() {
            config.num_attention_heads
        } else {
            config.num_key_value_heads
        };
        let kv_cache_config = KvCacheConfig {
            layer_count: attn_layer_count,
            context_size: config.context_size,
            head_count: 1,
            head_dim: kv_cache_token_size_for_layers(&config, &layers).max(kv_head_count),
            dtype: config.kv_cache_dtype,
            quantization: config.kv_quantization,
        };
        let kv_cache = KvCache::new(kv_cache_config).map_err(|e| format!("kv_cache: {:?}", e))?;

        // Initialize Mamba/SSM state
        let mut ssm_states: Vec<Vec<f32>> = Vec::with_capacity(config.layer_count);
        let mut ssm_conv_buffers: Vec<ConvHistoryRing> = Vec::with_capacity(config.layer_count);
        let shortconv_hist = config.shortconv_l_cache.saturating_sub(1).max(3);
        for layer in layers.iter().take(config.layer_count) {
            let state_dim = layer.ssm_a.len().max(1);
            ssm_states.push(vec![0.0_f32; state_dim]);
            let hist_dim = if !layer.shortconv_in_proj.is_empty() {
                config.hidden_size
            } else {
                layer.attn_qkv.output_dim(config.hidden_size).max(1)
            };
            let cap = if !layer.shortconv_in_proj.is_empty() {
                shortconv_hist
            } else {
                3
            };
            ssm_conv_buffers.push(ConvHistoryRing::new(cap, hist_dim));
        }

        let workspace = Workspace::for_config(&config);
        let last_output_hidden = vec![0.0_f32; config.hidden_size];

        Ok(Self {
            config,
            tok_embeddings,
            tok_embeddings_cols,
            norm_weight,
            output_weight,
            layers,
            mtp,
            kv_cache,
            kv_layer_map,
            ssm_states,
            ssm_conv_buffers,
            workspace,
            last_output_hidden,
        })
    }
}
