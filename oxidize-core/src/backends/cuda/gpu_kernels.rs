#[allow(unused_imports)]
use super::*;

/// Allocate (or reallocate) the GPU activation buffers for a given model size.
///
/// Safe to call multiple times; existing buffers are replaced only if the
/// dimensions change.
#[cfg(feature = "cuda")]
pub fn gpu_init_activation_buffers(
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<(), String> {
    with_gpu(|gpu| {
        if let Some(ref ab) = gpu.activation {
            if ab.hidden_size == hidden_size && ab.intermediate_size == intermediate_size {
                return Ok(());
            }
        }
        let hidden = cust::memory::DeviceBuffer::<f32>::zeroed(hidden_size).map_err(stringify)?;
        let normed = cust::memory::DeviceBuffer::<f32>::zeroed(hidden_size).map_err(stringify)?;
        let ffn_gate =
            cust::memory::DeviceBuffer::<f32>::zeroed(intermediate_size).map_err(stringify)?;
        let ffn_up =
            cust::memory::DeviceBuffer::<f32>::zeroed(intermediate_size).map_err(stringify)?;
        let ffn_down_in =
            cust::memory::DeviceBuffer::<f32>::zeroed(intermediate_size).map_err(stringify)?;
        let bpr_hidden = hidden_size / 256;
        let bpr_inter = intermediate_size / 256;
        let xq8k = cust::memory::DeviceBuffer::<u8>::zeroed(
            bpr_hidden * super::BLOCK_Q8_K_BYTES,
        )
        .map_err(stringify)?;
        let xq8k_ffn = cust::memory::DeviceBuffer::<u8>::zeroed(
            bpr_inter * super::BLOCK_Q8_K_BYTES,
        )
        .map_err(stringify)?;
        gpu.activation = Some(GpuActivationBuffer {
            hidden,
            normed,
            ffn_gate,
            ffn_up,
            ffn_down_in,
            xq8k,
            xq8k_ffn,
            hidden_size,
            intermediate_size,
        });
        Ok(())
    })
}

/// Upload a CPU f32 hidden-state slice into `activation.hidden` on the GPU.
///
/// The activation buffers must have been initialised via
/// [`gpu_init_activation_buffers`] before calling this function.
#[cfg(feature = "cuda")]
pub fn gpu_upload_hidden(hidden: &[f32]) -> Result<(), String> {
    with_gpu(|gpu| {
        let ab = gpu
            .activation
            .as_mut()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        if hidden.len() != ab.hidden_size {
            return Err(format!(
                "gpu_upload_hidden: slice len {} != hidden_size {}",
                hidden.len(),
                ab.hidden_size
            ));
        }
        ab.hidden.copy_from(hidden).map_err(stringify)
    })
}

/// Run RMS-norm on the GPU: reads `activation.hidden`, writes `activation.normed`.
///
/// `weight` is a per-element scale vector of length `hidden_size`; it is
/// cached in `resident_f32` (mmap-stable pointer identity, same as ordinary
/// weight matrices).
#[cfg(feature = "cuda")]
pub fn gpu_rms_norm(weight: &[f32], eps: f32) -> Result<(), String> {
    with_gpu(|gpu| {
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_size = ab.hidden_size;
        if weight.len() != hidden_size {
            return Err(format!(
                "gpu_rms_norm: weight len {} != hidden_size {}",
                weight.len(),
                hidden_size
            ));
        }

        // Cache norm weight in resident_f32 (same pointer-stable identity pattern).
        let key = f32_cache_key(weight);
        if !gpu.resident_f32.contains_key(&key) {
            let buf = cust::memory::DeviceBuffer::from_slice(weight).map_err(stringify)?;
            gpu.resident_f32.insert(key, buf);
        }

        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_ptr = ab.hidden.as_device_ptr();
        let normed_ptr = ab.normed.as_device_ptr();
        let weight_ptr = gpu.resident_f32.get(&key).unwrap().as_device_ptr();

        // One warp-reduction block per token; blockDim capped at 512 for the
        // parallel-reduction within the block (must be power-of-two).
        let block_size = hidden_size.next_power_of_two().min(512) as u32;
        let grid_size = 1_u32;
        let n = hidden_size as u32;

        let function = gpu
            .module
            .get_function(RMS_NORM_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        let shmem_bytes = block_size * 4; // rms_norm kernel uses block_size floats of dynamic shared mem
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, shmem_bytes, stream>>>(
                    hidden_ptr,
                    weight_ptr,
                    normed_ptr,
                    n,
                    eps
                )
            )
            .map_err(stringify)?;
        }
        Ok(())
    })
}

/// GPU residual add: `activation.hidden[i] += delta[i]` for all i.
///
/// `delta` must be a device buffer of exactly `hidden_size` f32 elements.
#[cfg(feature = "cuda")]
pub fn gpu_residual_add(delta: &cust::memory::DeviceBuffer<f32>) -> Result<(), String> {
    with_gpu(|gpu| {
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_size = ab.hidden_size;
        if delta.len() != hidden_size {
            return Err(format!(
                "gpu_residual_add: delta len {} != hidden_size {}",
                delta.len(),
                hidden_size
            ));
        }

        let hidden_ptr = ab.hidden.as_device_ptr();
        let delta_ptr = delta.as_device_ptr();
        let n = hidden_size as u32;

        let block_size = 256_u32;
        let grid_size = n.div_ceil(block_size);

        let function = gpu
            .module
            .get_function(RESIDUAL_ADD_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    hidden_ptr,
                    delta_ptr,
                    n
                )
            )
            .map_err(stringify)?;
        }
        Ok(())
    })
}

/// GPU SiLU-mul: `activation.ffn_down_in[i] = silu(ffn_gate[i]) * ffn_up[i]`.
///
/// Reads `activation.ffn_gate` and `activation.ffn_up`, writes
/// `activation.ffn_down_in`.  `intermediate_size` must equal the value used
/// when the buffers were allocated.
#[cfg(feature = "cuda")]
pub fn gpu_silu_mul(intermediate_size: usize) -> Result<(), String> {
    with_gpu(|gpu| {
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        if intermediate_size != ab.intermediate_size {
            return Err(format!(
                "gpu_silu_mul: intermediate_size {} != buffer size {}",
                intermediate_size, ab.intermediate_size
            ));
        }

        let gate_ptr = ab.ffn_gate.as_device_ptr();
        let up_ptr = ab.ffn_up.as_device_ptr();
        let out_ptr = ab.ffn_down_in.as_device_ptr();
        let n = intermediate_size as u32;

        let block_size = 256_u32;
        let grid_size = n.div_ceil(block_size);

        let function = gpu
            .module
            .get_function(SILU_MUL_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    gate_ptr,
                    up_ptr,
                    out_ptr,
                    n
                )
            )
            .map_err(stringify)?;
        }
        Ok(())
    })
}

/// Synchronise the GPU stream so that all preceding asynchronous kernel launches
/// have completed before the caller reads any device buffers.
#[cfg(feature = "cuda")]
pub fn gpu_sync_stream() -> Result<(), String> {
    with_gpu(|gpu| gpu.stream.synchronize().map_err(stringify))
}

/// True when GPU activation buffers are allocated (gpu_native decode path).
#[cfg(feature = "cuda")]
pub fn gpu_activation_ready() -> bool {
    with_gpu(|gpu| Ok(gpu.activation.is_some())).unwrap_or(false)
}

/// Write one token's embedding row directly into `activation.hidden` on the GPU.
///
/// Avoids CPU quantized dequant + host-to-device upload on every decode token.
#[cfg(feature = "cuda")]
pub fn gpu_embed_token(
    storage: &crate::inference::WeightStorage,
    hidden_size: usize,
    vocab_size: usize,
    token: crate::model::Token,
    scale: f32,
) -> Result<(), String> {
    use crate::gguf::GgufQuantizationType;
    use crate::inference::WeightStorage;

    if hidden_size == 0 || !hidden_size.is_multiple_of(256) {
        return Err(format!(
            "gpu_embed_token: hidden_size {hidden_size} must be a non-zero multiple of 256"
        ));
    }
    let token_idx = (token as usize).min(vocab_size.saturating_sub(1));
    let blocks_per_row = hidden_size / 256;

    with_gpu(|gpu| {
        let hidden_ptr = {
            let ab = gpu
                .activation
                .as_mut()
                .ok_or_else(|| "activation buffers not initialised".to_string())?;
            if ab.hidden_size != hidden_size {
                return Err(format!(
                    "gpu_embed_token: activation hidden_size {} != {hidden_size}",
                    ab.hidden_size
                ));
            }
            ab.hidden.as_device_ptr()
        };

        match storage {
            WeightStorage::F32(data) => {
                if data.len() < (token_idx + 1) * hidden_size {
                    return Err("gpu_embed_token: F32 embedding table too small".to_string());
                }
                let key = f32_cache_key(data);
                if !gpu.resident_f32.contains_key(&key) {
                    let buf = cust::memory::DeviceBuffer::from_slice(data).map_err(stringify)?;
                    gpu.resident_f32.insert(key, buf);
                }
                let table_ptr = gpu.resident_f32[&key].as_device_ptr();
                let block_size = 256_u32;
                let grid = hidden_size.div_ceil(256) as u32;
                let fn_embed = gpu
                    .module
                    .get_function(EMBED_F32_ROW_KERNEL_NAME)
                    .map_err(stringify)?;
                let stream = &gpu.stream;
                unsafe {
                    cust::launch!(fn_embed<<<grid, block_size, 0, stream>>>(
                        table_ptr,
                        token_idx as u32,
                        hidden_size as u32,
                        scale,
                        hidden_ptr
                    ))
                    .map_err(stringify)?;
                }
            }
            WeightStorage::Quantized(qtype, data) => match qtype {
                GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
                    let row_bytes = blocks_per_row * 144;
                    if data.len() < (token_idx + 1) * row_bytes {
                        return Err("gpu_embed_token: Q4K embedding table too small".to_string());
                    }
                    let key = bytes_cache_key(data);
                    gpu.ensure_resident_quant(key, data)?;
                    let table_ptr = gpu.resident_quant[&key].as_device_ptr();
                    let block_size = 256_u32;
                    let grid = blocks_per_row.div_ceil(256) as u32;
                    let fn_embed = gpu
                        .module
                        .get_function(EMBED_Q4K_F32_ROW_KERNEL_NAME)
                        .map_err(stringify)?;
                    let stream = &gpu.stream;
                    unsafe {
                        cust::launch!(fn_embed<<<grid, block_size, 0, stream>>>(
                            table_ptr,
                            token_idx as u32,
                            blocks_per_row as u32,
                            scale,
                            hidden_ptr
                        ))
                        .map_err(stringify)?;
                    }
                }
                _ => {
                    return Err(format!(
                        "gpu_embed_token: unsupported embedding quant type {:?}",
                        qtype
                    ));
                }
            },
            WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                let data = &mmap[*offset..*offset + *size];
                match qtype {
                    GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
                        let row_bytes = blocks_per_row * 144;
                        if data.len() < (token_idx + 1) * row_bytes {
                            return Err(
                                "gpu_embed_token: Q4K embedding table too small".to_string()
                            );
                        }
                        let key = bytes_cache_key(data);
                        gpu.ensure_resident_quant(key, data)?;
                        let table_ptr = gpu.resident_quant[&key].as_device_ptr();
                        let block_size = 256_u32;
                        let grid = blocks_per_row.div_ceil(256) as u32;
                        let fn_embed = gpu
                            .module
                            .get_function(EMBED_Q4K_F32_ROW_KERNEL_NAME)
                            .map_err(stringify)?;
                        let stream = &gpu.stream;
                        unsafe {
                            cust::launch!(fn_embed<<<grid, block_size, 0, stream>>>(
                                table_ptr,
                                token_idx as u32,
                                blocks_per_row as u32,
                                scale,
                                hidden_ptr
                            ))
                            .map_err(stringify)?;
                        }
                    }
                    _ => {
                        return Err(format!(
                            "gpu_embed_token: unsupported embedding quant type {:?}",
                            qtype
                        ));
                    }
                }
            }
        }
        Ok(())
    })
}
