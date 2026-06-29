use super::*;

impl LayerWiseModel {
    pub fn load_from_gguf(
        mapped: &MappedGgufFile,
        config: InferenceConfig,
        layer_cache_size: usize,
    ) -> Result<Self, String> {
        let mut tok_embeddings: Option<WeightStorage> = None;
        let mut tok_embeddings_cols: usize = config.hidden_size;
        let mut norm_weight: Option<Vec<f32>> = None;
        let mut output_weight: Option<WeightStorage> = None;
        let mut layer_tensors: Vec<HashMap<String, GgufTensorRef>> =
            vec![HashMap::new(); config.layer_count];
        // Byte ranges of dense (non-routed-expert) mmap-resident weights: the
        // candidate set for partial NUMA replication. Routed expert tensors
        // (`*_exps`) are excluded — they are the bulk of MoE models and only
        // ~2% of them is read per token; shared experts (`*_shexp`) are dense.
        let mut dense_ranges: Vec<(usize, usize)> = Vec::new();

        let is_supported_quant_gemv = |qtype: GgufQuantizationType| {
            matches!(
                qtype,
                GgufQuantizationType::Q8_0
                    | GgufQuantizationType::Q4_K_S
                    | GgufQuantizationType::Q4_K_M
                    | GgufQuantizationType::Q6_K
            )
        };

        for tensor in mapped.mapped_tensor_infos().iter() {
            let qtype = GgufQuantizationType::from_ggml_type(tensor.ggml_type);
            let value_count: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
            let qsize = quantized_size(qtype, value_count)
                .map_err(|e| format!("quantized_size: {:?}", e))?;
            let offset = tensor.absolute_offset as usize;

            let Some(tensor_name) = normalize_gguf_tensor_name(&tensor.name) else {
                continue;
            };
            match tensor_name.as_str() {
                "tok_embeddings.weight" | "token_embd.weight" => {
                    tok_embeddings_cols = tensor
                        .dimensions
                        .get(1)
                        .copied()
                        .unwrap_or(config.hidden_size as u64)
                        as usize;
                    if is_supported_quant_gemv(qtype) {
                        dense_ranges.push((offset, qsize));
                        tok_embeddings = Some(WeightStorage::MmapQuantized(
                            qtype,
                            mapped.tensor_mmap(tensor),
                            offset,
                            qsize,
                        ));
                    } else {
                        let qdata = mapped.tensor_bytes(tensor, qsize);
                        let mut f32_data = vec![0.0_f32; value_count];
                        dequantize_scalar(qtype, qdata, &mut f32_data)
                            .map_err(|e| format!("dequantize: {:?}", e))?;
                        tok_embeddings = Some(WeightStorage::F32(f32_data));
                    }
                }
                "norm.weight" | "output_norm.weight" => {
                    let qdata = mapped.tensor_bytes(tensor, qsize);
                    let mut f32_data = vec![0.0_f32; value_count];
                    dequantize_scalar(qtype, qdata, &mut f32_data)
                        .map_err(|e| format!("dequantize: {:?}", e))?;
                    norm_weight = Some(f32_data);
                }
                "output.weight" => {
                    if is_supported_quant_gemv(qtype) {
                        dense_ranges.push((offset, qsize));
                        output_weight = Some(WeightStorage::MmapQuantized(
                            qtype,
                            mapped.tensor_mmap(tensor),
                            offset,
                            qsize,
                        ));
                    } else {
                        let qdata = mapped.tensor_bytes(tensor, qsize);
                        let mut f32_data = vec![0.0_f32; value_count];
                        dequantize_scalar(qtype, qdata, &mut f32_data)
                            .map_err(|e| format!("dequantize: {:?}", e))?;
                        output_weight = Some(WeightStorage::F32(f32_data));
                    }
                }
                name if name.starts_with("blk.") => {
                    let parts: Vec<&str> = name.split('.').collect();
                    // Suffix-less vectors like `blk.N.ssm_a` are 3 parts.
                    if parts.len() < 3 {
                        continue;
                    }
                    let layer_idx: usize = parts[1]
                        .parse()
                        .map_err(|_| format!("bad layer index: {}", name))?;
                    if layer_idx >= config.layer_count {
                        continue;
                    }
                    let mut key = parts[2..].join(".");
                    // llama.cpp-style qwen35 GGUFs emit the GDN decay vector as
                    // a bare `ssm_a` (no `.weight` suffix); canonicalize so the
                    // slot loader's `ssm_a.weight` match finds it.
                    if key == "ssm_a" {
                        key = "ssm_a.weight".to_owned();
                    }
                    if !key.contains("_exps") {
                        dense_ranges.push((offset, qsize));
                    }
                    layer_tensors[layer_idx].insert(
                        key,
                        GgufTensorRef {
                            qtype,
                            mmap_index: tensor.mmap_index,
                            offset,
                            size: qsize,
                            value_count,
                        },
                    );
                }
                _ => {}
            }
        }

        let tok_embeddings = tok_embeddings.ok_or("missing tok_embeddings.weight")?;
        let norm_weight = norm_weight.ok_or("missing norm.weight")?;
        let output_weight = output_weight.unwrap_or_else(|| tok_embeddings.clone());

        let kv_cache_config = crate::kv_cache::KvCacheConfig {
            layer_count: config.layer_count,
            context_size: config.context_size,
            head_count: config.num_key_value_heads,
            head_dim: config.kv_head_dim(),
            dtype: config.kv_cache_dtype,
            quantization: config.kv_quantization,
        };
        let kv_cache = KvCache::new(kv_cache_config).map_err(|e| format!("kv_cache: {:?}", e))?;

        let mut ssm_states = Vec::with_capacity(config.layer_count);
        let mut ssm_conv_buffers = Vec::with_capacity(config.layer_count);
        let layer_count = config.layer_count;
        for _ in 0..layer_count {
            ssm_states.push(vec![0.0_f32; 1]);
            ssm_conv_buffers.push(ConvHistoryRing::new(4, 0));
        }

        let effective_cache = if layer_cache_size == 0 {
            layer_count
        } else {
            layer_cache_size
        };
        if effective_cache < layer_count {
            eprintln!(
                "layer-wise: layer_cache={effective_cache} < {layer_count} layers — expect low TPS from cache thrashing; use --layer-cache 0 or --layer-cache {layer_count} when RAM allows"
            );
        }

        let numa_mode = std::env::var("OXIDIZE_NUMA_REPLICATE").unwrap_or_default();
        if numa_mode == "1" || numa_mode == "dense" {
            let t0 = std::time::Instant::now();
            // Whole-model replication needs one full copy per node; cap it at
            // a fraction of the smallest node so the copy cannot OOM the box.
            // Past the cap (e.g. a 208 GB MoE GGUF on 92/224 GB nodes), fall
            // back to replicating only the dense tensors — a few GB that
            // carry roughly half the per-token weight reads.
            let full_budget = crate::numa::min_node_total_bytes() / 2;
            let full_fits = mapped.total_bytes_len() <= full_budget;
            let replicated = if numa_mode == "1" && full_fits {
                if crate::numa::replicate(mapped.bytes()) {
                    mapped.bytes().len()
                } else {
                    0
                }
            } else {
                crate::numa::replicate_ranges(mapped.bytes(), &dense_ranges)
            };
            if replicated > 0 {
                eprintln!(
                    "layer-wise: NUMA-replicated {:.1} GiB of {} weights per node in {:.1}s",
                    replicated as f64 / (1u64 << 30) as f64,
                    if numa_mode == "1" && full_fits {
                        "all"
                    } else {
                        "dense"
                    },
                    t0.elapsed().as_secs_f32()
                );
            } else {
                eprintln!("layer-wise: NUMA replication unavailable; using shared mapping");
            }
        }
        Ok(Self {
            config,
            mmap: Arc::new(mapped.clone()),
            layer_tensors,
            tok_embeddings,
            tok_embeddings_cols,
            norm_weight,
            output_weight,
            kv_cache,
            ssm_states,
            ssm_conv_buffers,
            ssm_pos: 0,
            ssm_checkpoints: Vec::new(),
            cache: LayerCache::new(effective_cache, layer_count),
        })
    }

    /// Preload layer weights into the LRU cache so decode does not reload tensors each token.
    pub fn warm_layer_cache(&mut self) -> Result<(), String> {
        let layer_count = self.config.layer_count;
        let capacity = self.cache.capacity;
        let warm_count = capacity.min(layer_count);
        if warm_count == 0 {
            return Ok(());
        }
        eprintln!("layer-wise: warming {warm_count}/{layer_count} layer slots...");
        let started = std::time::Instant::now();
        for layer_idx in 0..warm_count {
            if self.cache.entries[layer_idx].is_none() {
                let layer = self.load_layer_weights(layer_idx)?;
                self.cache.put(layer_idx, layer);
            }
        }
        eprintln!(
            "layer-wise: cache warm done in {:.1}s",
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }
}
