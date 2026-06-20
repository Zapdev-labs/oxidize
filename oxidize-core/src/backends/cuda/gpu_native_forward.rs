#[allow(unused_imports)]
use super::*;

/// GPU-native attention block: RMS-norm the hidden state, run Q/K/V projections,
/// then download Q/K/V to CPU for the attention computation.
///
/// The three GEMV results stay on the GPU until the combined D2H sync at the
/// end, keeping the GPU busy and reducing the CPU↔GPU round-trips from 3 to 1
/// per attention block.
///
/// Only supports Q4_K_S / Q4_K_M quantised weight matrices (which covers the
/// vast majority of GGUF files in the wild).  Returns `Err` if called without
/// having first called [`gpu_init_activation_buffers`].
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_attn_rms_and_qkv_q4k(
    attn_norm: &[f32],
    eps: f32,
    wq: &[u8],
    q_len: usize,
    hidden_size: usize,
    wk: &[u8],
    kv_len: usize,
    wv: &[u8],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> Result<(), String> {
    if !hidden_size.is_multiple_of(256) {
        return Err(format!(
            "gpu_attn_rms_and_qkv: hidden_size {hidden_size} not multiple of 256"
        ));
    }
    let blocks_per_row = hidden_size / 256;
    let bpr_u32 = blocks_per_row as u32;
    let q_u32 = q_len as u32;
    let kv_u32 = kv_len as u32;

    with_gpu(|gpu| {
        // --- RMS-norm: hidden → normed ---
        let norm_key = f32_cache_key(attn_norm);
        if !gpu.resident_f32.contains_key(&norm_key) {
            let buf = cust::memory::DeviceBuffer::from_slice(attn_norm).map_err(stringify)?;
            gpu.resident_f32.insert(norm_key, buf);
        }

        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_ptr = ab.hidden.as_device_ptr();
        let normed_ptr = ab.normed.as_device_ptr();
        let hidden_size_u32 = ab.hidden_size as u32;
        let weight_ptr = gpu.resident_f32[&norm_key].as_device_ptr();

        let block_dim = ab.hidden_size.next_power_of_two().min(512) as u32;
        let shmem_attn = block_dim * 4;
        let function_norm = gpu
            .module
            .get_function(RMS_NORM_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(function_norm<<<1, block_dim, shmem_attn, stream>>>(
                hidden_ptr, weight_ptr, normed_ptr, hidden_size_u32, eps
            ))
            .map_err(stringify)?;
        }

        // Detect Q4K vs Q6K from block byte-size (Q4K=144, Q6K=210 bytes per 256-value block).
        let quant_kern_name = |w: &[u8], rows: usize| -> &'static str {
            let bsz = if rows > 0 && blocks_per_row > 0 {
                w.len() / (rows * blocks_per_row)
            } else {
                144
            };
            if bsz >= 200 {
                GEMV_Q6K_F32IN_KERNEL_NAME
            } else {
                GEMV_Q4K_F32IN_KERNEL_NAME
            }
        };
        let qname_q = quant_kern_name(wq, q_len);
        let qname_k = quant_kern_name(wk, kv_len);
        let qname_v = quant_kern_name(wv, kv_len);

        // --- Upload weight matrices (Q4K or Q6K) ---
        let q_key = bytes_cache_key(wq);
        gpu.ensure_resident_quant(q_key, wq)?;
        let k_key = bytes_cache_key(wk);
        gpu.ensure_resident_quant(k_key, wk)?;
        let v_key = bytes_cache_key(wv);
        gpu.ensure_resident_quant(v_key, wv)?;

        // --- Allocate temporary GPU output buffers ---
        let d_q = gpu.get_f32_buffer(q_len)?;
        let d_k = gpu.get_f32_buffer(kv_len)?;
        let d_v = gpu.get_f32_buffer(kv_len)?;

        let block_size = 256_u32;
        let stream = &gpu.stream;

        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let normed_ptr = ab.normed.as_device_ptr();

        let q_grid = q_u32.saturating_mul(32).div_ceil(block_size);
        let k_grid = kv_u32.saturating_mul(32).div_ceil(block_size);

        let wq_ptr = gpu.resident_quant[&q_key].as_device_ptr();
        let wk_ptr = gpu.resident_quant[&k_key].as_device_ptr();
        let wv_ptr = gpu.resident_quant[&v_key].as_device_ptr();

        let fn_q = gpu.module.get_function(qname_q).map_err(stringify)?;
        let fn_k = gpu.module.get_function(qname_k).map_err(stringify)?;
        let fn_v = gpu.module.get_function(qname_v).map_err(stringify)?;

        unsafe {
            cust::launch!(fn_q<<<q_grid, block_size, 0, stream>>>(
                wq_ptr, normed_ptr, d_q.as_device_ptr(), q_u32, bpr_u32
            ))
            .map_err(stringify)?;
            cust::launch!(fn_k<<<k_grid, block_size, 0, stream>>>(
                wk_ptr, normed_ptr, d_k.as_device_ptr(), kv_u32, bpr_u32
            ))
            .map_err(stringify)?;
            cust::launch!(fn_v<<<k_grid, block_size, 0, stream>>>(
                wv_ptr, normed_ptr, d_v.as_device_ptr(), kv_u32, bpr_u32
            ))
            .map_err(stringify)?;
        }

        // --- Single combined D2H sync: download Q, K, V ---
        gpu.stream.synchronize().map_err(stringify)?;
        d_q.copy_to(q_out).map_err(stringify)?;
        d_k.copy_to(k_out).map_err(stringify)?;
        d_v.copy_to(v_out).map_err(stringify)?;

        gpu.return_f32_buffer(d_q);
        gpu.return_f32_buffer(d_k);
        gpu.return_f32_buffer(d_v);
        Ok(())
    })
}

/// GPU-native O-projection + attention residual add.
///
/// Uploads the CPU attention output to the GPU, runs the wo Q4K GEMV, then
/// does an in-place residual add on the GPU hidden state.  The hidden state
/// remains on the GPU after this call; no D2H copy is performed.
#[cfg(feature = "cuda")]
pub fn gpu_wo_residual_q4k(
    attn_out: &[f32],
    wo: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    if !cols.is_multiple_of(256) {
        return Err(format!(
            "gpu_wo_residual_q4k: cols {cols} not multiple of 256"
        ));
    }
    let bpr = (cols / 256) as u32;
    let rows_u32 = rows as u32;

    with_gpu(|gpu| {
        // Upload attention output to a temporary GPU buffer.
        let mut d_attn = gpu.get_f32_buffer(attn_out.len())?;
        d_attn.copy_from(attn_out).map_err(stringify)?;

        let wo_key = bytes_cache_key(wo);
        gpu.ensure_resident_quant(wo_key, wo)?;

        // Reuse `activation.normed` as the wo-projection output buffer — it is
        // no longer needed after the attention QKV step.
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let normed_ptr = ab.normed.as_device_ptr();
        let hidden_ptr = ab.hidden.as_device_ptr();
        let hidden_n = ab.hidden_size as u32;

        let block_size = 256_u32;
        let wo_grid = rows_u32.saturating_mul(32).div_ceil(block_size);
        let res_grid = hidden_n.div_ceil(block_size);
        let wo_ptr = gpu.resident_quant[&wo_key].as_device_ptr();

        let bpr_usize = (cols / 256) as usize;
        let wo_kern_name = if bpr_usize > 0 && rows > 0 && wo.len() / (rows * bpr_usize) >= 200 {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else {
            GEMV_Q4K_F32IN_KERNEL_NAME
        };
        let fn_wo = gpu.module.get_function(wo_kern_name).map_err(stringify)?;
        let fn_res = gpu
            .module
            .get_function(RESIDUAL_ADD_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;

        unsafe {
            cust::launch!(fn_wo<<<wo_grid, block_size, 0, stream>>>(
                wo_ptr, d_attn.as_device_ptr(), normed_ptr, rows_u32, bpr
            ))
            .map_err(stringify)?;
            cust::launch!(fn_res<<<res_grid, block_size, 0, stream>>>(
                hidden_ptr, normed_ptr, hidden_n
            ))
            .map_err(stringify)?;
        }

        gpu.return_f32_buffer(d_attn);
        Ok(())
    })
}

/// GPU-native FFN block: RMS-norm + gate + up + SiLU-mul + down + residual.
///
/// Runs the complete feed-forward network on the GPU without any CPU↔GPU data
/// movement.  Reads `activation.hidden`, writes back to `activation.hidden`.
/// Intermediate results (gate, up, silu output) stay in the activation buffers.
///
/// The hidden state is **not** downloaded; the caller owns the GPU-resident
/// hidden state across layers and only downloads it once (for logit sampling).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_ffn_q4k(
    ffn_norm: &[f32],
    eps: f32,
    gate_w: &[u8],
    gate_rows: usize,
    gate_cols: usize,
    up_w: &[u8],
    up_rows: usize,
    down_w: &[u8],
    down_rows: usize,
    down_cols: usize,
) -> Result<(), String> {
    if !gate_cols.is_multiple_of(256) || !down_cols.is_multiple_of(256) {
        return Err(format!(
            "gpu_ffn_q4k: gate_cols {gate_cols} or down_cols {down_cols} not multiple of 256"
        ));
    }
    let gate_bpr = (gate_cols / 256) as u32;
    let down_bpr = (down_cols / 256) as u32;
    let gate_u32 = gate_rows as u32;
    let up_u32 = up_rows as u32;
    let down_u32 = down_rows as u32;

    with_gpu(|gpu| {
        // --- FFN RMS-norm: hidden → normed ---
        let norm_key = f32_cache_key(ffn_norm);
        if !gpu.resident_f32.contains_key(&norm_key) {
            let buf = cust::memory::DeviceBuffer::from_slice(ffn_norm).map_err(stringify)?;
            gpu.resident_f32.insert(norm_key, buf);
        }
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_ptr = ab.hidden.as_device_ptr();
        let normed_ptr = ab.normed.as_device_ptr();
        let hidden_n = ab.hidden_size as u32;
        let norm_weight_ptr = gpu.resident_f32[&norm_key].as_device_ptr();
        let block_dim = ab.hidden_size.next_power_of_two().min(512) as u32;

        let shmem_ffn = block_dim * 4;
        let fn_norm = gpu
            .module
            .get_function(RMS_NORM_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(fn_norm<<<1, block_dim, shmem_ffn, stream>>>(
                hidden_ptr, norm_weight_ptr, normed_ptr, hidden_n, eps
            ))
            .map_err(stringify)?;
        }

        // --- Upload FFN weight matrices ---
        let gate_key = bytes_cache_key(gate_w);
        gpu.ensure_resident_quant(gate_key, gate_w)?;
        let up_key = bytes_cache_key(up_w);
        gpu.ensure_resident_quant(up_key, up_w)?;
        let down_key = bytes_cache_key(down_w);
        gpu.ensure_resident_quant(down_key, down_w)?;

        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let normed_ptr = ab.normed.as_device_ptr();
        let gate_buf_ptr = ab.ffn_gate.as_device_ptr();
        let up_buf_ptr = ab.ffn_up.as_device_ptr();
        let ffn_down_in_ptr = ab.ffn_down_in.as_device_ptr();
        let inter_n = ab.intermediate_size as u32;

        let block_size = 256_u32;
        let gate_grid = gate_u32.saturating_mul(32).div_ceil(block_size);
        let up_grid = up_u32.saturating_mul(32).div_ceil(block_size);
        let silu_grid = inter_n.div_ceil(block_size);
        let down_grid = down_u32.saturating_mul(32).div_ceil(block_size);
        let res_grid = hidden_n.div_ceil(block_size);

        let gate_ptr = gpu.resident_quant[&gate_key].as_device_ptr();
        let up_ptr = gpu.resident_quant[&up_key].as_device_ptr();
        let down_ptr = gpu.resident_quant[&down_key].as_device_ptr();

        // Auto-detect Q4K vs Q6K for each FFN weight by block byte-size.
        let gate_bpr_usize = gate_bpr as usize;
        let down_bpr_usize = down_bpr as usize;
        let gate_kern = if gate_bpr_usize > 0
            && gate_rows > 0
            && gate_w.len() / (gate_rows * gate_bpr_usize) >= 200
        {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else {
            GEMV_Q4K_F32IN_KERNEL_NAME
        };
        let up_kern = if gate_bpr_usize > 0
            && up_rows > 0
            && up_w.len() / (up_rows * gate_bpr_usize) >= 200
        {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else {
            GEMV_Q4K_F32IN_KERNEL_NAME
        };
        let down_kern = if down_bpr_usize > 0
            && down_rows > 0
            && down_w.len() / (down_rows * down_bpr_usize) >= 200
        {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else {
            GEMV_Q4K_F32IN_KERNEL_NAME
        };
        let fn_gate = gpu.module.get_function(gate_kern).map_err(stringify)?;
        let fn_up = gpu.module.get_function(up_kern).map_err(stringify)?;
        let fn_down = gpu.module.get_function(down_kern).map_err(stringify)?;
        let fn_silu = gpu
            .module
            .get_function(SILU_MUL_KERNEL_NAME)
            .map_err(stringify)?;
        let fn_res = gpu
            .module
            .get_function(RESIDUAL_ADD_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;

        // Reuse `normed` as the down-projection output buffer (safe: gate/up
        // GEMVs only READ normed; by the time down runs, normed is free).
        let down_out_ptr = normed_ptr;
        let hidden_ptr = ab.hidden.as_device_ptr();

        unsafe {
            // gate × normed → ffn_gate
            cust::launch!(fn_gate<<<gate_grid, block_size, 0, stream>>>(
                gate_ptr, normed_ptr, gate_buf_ptr, gate_u32, gate_bpr
            ))
            .map_err(stringify)?;
            // up × normed → ffn_up
            cust::launch!(fn_up<<<up_grid, block_size, 0, stream>>>(
                up_ptr, normed_ptr, up_buf_ptr, up_u32, gate_bpr
            ))
            .map_err(stringify)?;
            // silu(ffn_gate) * ffn_up → ffn_down_in
            cust::launch!(fn_silu<<<silu_grid, block_size, 0, stream>>>(
                gate_buf_ptr, up_buf_ptr, ffn_down_in_ptr, inter_n
            ))
            .map_err(stringify)?;
            // down × ffn_down_in → normed (reused as temp)
            cust::launch!(fn_down<<<down_grid, block_size, 0, stream>>>(
                down_ptr, ffn_down_in_ptr, down_out_ptr, down_u32, down_bpr
            ))
            .map_err(stringify)?;
            // hidden += normed (ffn delta)
            cust::launch!(fn_res<<<res_grid, block_size, 0, stream>>>(
                hidden_ptr, down_out_ptr, hidden_n
            ))
            .map_err(stringify)?;
        }
        // No D2H — hidden remains GPU-resident.
        Ok(())
    })
}

/// Download the current `activation.hidden` back to a CPU slice.
///
/// Triggers a device-to-host DMA transfer; the stream is synchronised before
/// the copy returns so the caller receives a consistent snapshot.
#[cfg(feature = "cuda")]
pub fn gpu_download_hidden(out: &mut [f32]) -> Result<(), String> {
    with_gpu(|gpu| {
        // Flush any pending kernel work on the stream before reading back.
        gpu.stream.synchronize().map_err(stringify)?;
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        if out.len() != ab.hidden_size {
            return Err(format!(
                "gpu_download_hidden: out len {} != hidden_size {}",
                out.len(),
                ab.hidden_size
            ));
        }
        ab.hidden.copy_to(out).map_err(stringify)
    })
}

/// GPU-resident lm_head (output projection): Q4K or Q6K weight × F32 normed hidden → F32 logits.
///
/// The weight matrix `weight_bytes` must be quantized (Q4_K or Q6_K); `rows` is the vocabulary
/// size, `hidden_size` is the input dimension.  The weight is uploaded once and kept resident.
/// The `normed` input (hidden_size floats) is uploaded, the kernel runs, and `vocab_size` logits
/// are downloaded into `logits`.
///
/// Calling this instead of the CPU `gemv_weight` path saves 6–16 ms per token on a 128 k-vocab
/// model because the weight read is bounded by GPU HBM bandwidth (~3 TB/s) rather than CPU RAM
/// bandwidth (~50 GB/s).
#[cfg(feature = "cuda")]
pub fn gpu_lm_head_quantized(
    weight_bytes: &[u8],
    rows: usize,
    hidden_size: usize,
    normed: &[f32],
    logits: &mut [f32],
) -> Result<(), String> {
    if !hidden_size.is_multiple_of(256) {
        return Err(format!(
            "gpu_lm_head: hidden_size {hidden_size} not a multiple of 256"
        ));
    }
    if normed.len() != hidden_size {
        return Err(format!(
            "gpu_lm_head: normed len {} != hidden_size {hidden_size}",
            normed.len()
        ));
    }
    if logits.len() != rows {
        return Err(format!(
            "gpu_lm_head: logits len {} != rows {rows}",
            logits.len()
        ));
    }

    let blocks_per_row = hidden_size / 256;
    let bpr_u32 = blocks_per_row as u32;
    let rows_u32 = rows as u32;

    let kern_name =
        if blocks_per_row > 0 && rows > 0 && weight_bytes.len() / (rows * blocks_per_row) >= 200 {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else {
            GEMV_Q4K_F32IN_KERNEL_NAME
        };

    with_gpu(|gpu| {
        // Keep the weight matrix resident across tokens.
        let w_key = bytes_cache_key(weight_bytes);
        gpu.ensure_resident_quant(w_key, weight_bytes)?;

        // Upload the F32 normed vector.
        let mut d_input = gpu.get_f32_buffer(hidden_size)?;
        d_input.copy_from(normed).map_err(stringify)?;

        // Allocate GPU output buffer for logits.
        let d_output = gpu.get_f32_buffer(rows)?;

        let block_size = 256_u32;
        // One warp (32 threads) per output row.
        let grid = (rows_u32 * 32).div_ceil(block_size);

        let fn_gemv = gpu.module.get_function(kern_name).map_err(stringify)?;
        let w_ptr = gpu.resident_quant[&w_key].as_device_ptr();
        let stream = &gpu.stream;

        unsafe {
            cust::launch!(fn_gemv<<<grid, block_size, 0, stream>>>(
                w_ptr, d_input.as_device_ptr(), d_output.as_device_ptr(),
                rows_u32, bpr_u32
            ))
            .map_err(stringify)?;
        }

        // Sync and download — logits needed on CPU immediately for sampling.
        gpu.stream.synchronize().map_err(stringify)?;
        d_output.copy_to(logits).map_err(stringify)?;

        gpu.return_f32_buffer(d_input);
        gpu.return_f32_buffer(d_output);
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// On-device attention (OX_GPU_ATTN): device-resident F16 KV cache + the
// rope / kv-append / flash-attention launchers.
//
// All of these run on `gpu.stream` and NEVER call `stream.synchronize()` and
// NEVER copy to/from host (the host slices passed to `gpu_attn_rope_append_flash`
// are uploaded into pooled device buffers only). The hidden state therefore
// stays GPU-resident across the whole layer range.
// ---------------------------------------------------------------------------

/// Allocate (or reset) the device-resident F16 KV cache.
///
/// * `num_layers` – number of KV layers (index space of `kv_layer_idx`).
/// * `kv_len` – per-token row stride = `kv_heads * head_dim`.
/// * `context_size` – row capacity per layer.
///
/// If the geometry matches the resident cache this is a no-op (the buffers are
/// kept and `kv_seq_len` is left MONOTONIC — it is NOT reset here, because
/// `gpu_kv_init` runs once per decode token and resetting per-token would zero
/// the sequence length mid-generation). A new prompt/sequence must call
/// [`gpu_kv_reset`] explicitly (the call site gates that on `pos == 0`).
/// Otherwise the K/V buffers are (re)allocated zeroed.
/// Idempotent, mirroring [`gpu_init_activation_buffers`].
#[cfg(feature = "cuda")]
pub fn gpu_kv_init(num_layers: usize, kv_len: usize, context_size: usize) -> Result<(), String> {
    if num_layers == 0 || kv_len == 0 || context_size == 0 {
        return Err(format!(
            "gpu_kv_init: invalid geometry num_layers={num_layers} kv_len={kv_len} context_size={context_size}"
        ));
    }
    with_gpu(|gpu| {
        if gpu.kv_layers == num_layers && gpu.kv_len == kv_len && gpu.kv_context == context_size {
            // Same geometry — keep buffers AND keep `kv_seq_len` monotonic.
            // Resetting here would zero the seq length on every token; the
            // new-sequence reset is driven by `gpu_kv_reset()` at the call site
            // (gated on `pos == 0`).
            return Ok(());
        }
        let elems = context_size
            .checked_mul(kv_len)
            .ok_or_else(|| "gpu_kv_init: context_size*kv_len overflow".to_string())?;
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_cache.push(cust::memory::DeviceBuffer::<u16>::zeroed(elems).map_err(stringify)?);
            v_cache.push(cust::memory::DeviceBuffer::<u16>::zeroed(elems).map_err(stringify)?);
        }
        gpu.kv_k_cache = k_cache;
        gpu.kv_v_cache = v_cache;
        gpu.kv_seq_len = vec![0; num_layers];
        gpu.kv_layers = num_layers;
        gpu.kv_len = kv_len;
        gpu.kv_context = context_size;
        Ok(())
    })
}

/// Reset every layer's sequence length to 0 (start of a new prompt/sequence).
///
/// The buffers are left allocated; `kv_seq_len` gates which rows are read, so
/// stale rows are never observed.
#[cfg(feature = "cuda")]
pub fn gpu_kv_reset() -> Result<(), String> {
    with_gpu(|gpu| {
        for s in gpu.kv_seq_len.iter_mut() {
            *s = 0;
        }
        Ok(())
    })
}

/// Roll a single layer's KV cache back to `pos` tokens (speculative rollback
/// parity with `KvCache::rewind_to`).
#[cfg(feature = "cuda")]
pub fn gpu_kv_rewind(layer: usize, pos: usize) -> Result<(), String> {
    with_gpu(|gpu| {
        if let Some(s) = gpu.kv_seq_len.get_mut(layer) {
            *s = pos.min(gpu.kv_context);
        }
        Ok(())
    })
}

/// Launch `rope_f32_kernel` on `gpu.stream`: in-place partial NeoX RoPE on the
/// device F32 Q (`d_q`, `[n_q_heads*head_dim]`) and K (`d_k`, `[n_kv_heads*head_dim]`).
///
/// One block per head (`gridDim.x = n_q_heads + n_kv_heads`), `blockDim.x = head_dim`
/// (>= rope_dim/2 pairs; lanes beyond `rope_dim/2` early-return). `pos == 0` is the
/// identity inside the kernel. Pure launch — no sync, no host copy.
///
/// `d_q`/`d_k` are taken as raw device pointers (`cust::memory::DevicePointer<f32>`)
/// so the caller can keep Q/K device-resident from the QKV GEMV across all three
/// attention kernels.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_rope_f32(
    gpu: &GpuState,
    d_q: cust::memory::DevicePointer<f32>,
    d_k: cust::memory::DevicePointer<f32>,
    pos: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
) -> Result<(), String> {
    let func = gpu
        .module
        .get_function(ROPE_F32_KERNEL_NAME)
        .map_err(stringify)?;
    let grid = n_q_heads + n_kv_heads;
    let block = head_dim;
    let stream = &gpu.stream;
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(
            d_q, d_k, pos, n_q_heads, n_kv_heads, head_dim, rope_dim, theta
        ))
        .map_err(stringify)?;
    }
    Ok(())
}

/// Launch `kv_append_f16_kernel` on `gpu.stream`: cast the post-RoPE F32 K/V
/// (`d_k`/`d_v`, `[kv_len]`) into row `phys_pos` of the F16 KV cache for layer
/// `kv_layer_idx`. One thread per element; `gridDim.x = ceil(kv_len/256)`.
///
/// `phys_pos` is the physical row (caller passes `pos % context_size`). Updates
/// `kv_seq_len[kv_layer_idx]` host-side after the launch (bookkeeping only — no
/// sync required). Pure launch — no sync, no host copy.
#[cfg(feature = "cuda")]
pub(super) fn launch_kv_append_f16(
    gpu: &mut GpuState,
    kv_layer_idx: usize,
    d_k: cust::memory::DevicePointer<f32>,
    d_v: cust::memory::DevicePointer<f32>,
    phys_pos: u32,
    logical_pos: usize,
) -> Result<(), String> {
    if kv_layer_idx >= gpu.kv_layers {
        return Err(format!(
            "launch_kv_append_f16: kv_layer_idx {kv_layer_idx} out of range (kv_layers {})",
            gpu.kv_layers
        ));
    }
    let kv_len = gpu.kv_len as u32;
    let context = gpu.kv_context;
    let func = gpu
        .module
        .get_function(KV_APPEND_F16_KERNEL_NAME)
        .map_err(stringify)?;
    let kc_ptr = gpu.kv_k_cache[kv_layer_idx].as_device_ptr();
    let vc_ptr = gpu.kv_v_cache[kv_layer_idx].as_device_ptr();
    let block = 256_u32;
    let grid = kv_len.div_ceil(block);
    let stream = &gpu.stream;
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(
            d_k, d_v, kc_ptr, vc_ptr, phys_pos, kv_len
        ))
        .map_err(stringify)?;
    }
    if let Some(s) = gpu.kv_seq_len.get_mut(kv_layer_idx) {
        *s = (logical_pos + 1).min(context);
    }
    Ok(())
}

/// Orchestrate on-device attention for one decode token (env `OX_GPU_ATTN`).
///
/// Uploads the post-QKV F32 `q`/`k`/`v` (no biases / no per-head norm on the
/// gpu-native eligible layers), then runs — on a single stream with NO
/// intermediate host sync — partial NeoX RoPE, the F16 KV-cache append at row
/// `pos % context`, and the GQA online-softmax flash-attention decode reading the
/// resident F16 KV cache. A single end-of-attention sync downloads the F32
/// attention result into `attn_out` (the buffer that feeds the wo GEMV).
///
/// This mirrors the CPU attention island in `layers.rs` exactly:
/// * RoPE: split-half, partial `rope_dim`, per-layer `theta`, `pos == 0` identity.
/// * KV store: F16 half bits, row layout `[pos][kv_head*head_dim + d]`.
/// * scale = `1/sqrt(head_dim)`, GQA `kv_head = q_head/(q_heads/kv_heads)`.
/// * sliding window via `base_row`/`eff_seq_len`.
///
/// The host `KvCache` is NOT touched here; under `OX_GPU_ATTN` the device cache
/// is authoritative for the whole run (see the comment at the call site).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_attn_rope_append_flash(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    rope_dim: usize,
    theta: f32,
    pos: usize,
    kv_layer_idx: usize,
    layer_window: usize,
    attn_out: &mut [f32],
) -> Result<(), String> {
    // Simplifying-invariant guards (spec 3.6). Any failure → Err. NOTE: under
    // OX_GPU_ATTN (a whole-run mode) the host KvCache is never populated, so the
    // caller treats this Err as a HARD error (it cannot safely fall back to the
    // CPU attention island, which would read an all-zero host cache). These
    // guards therefore protect against unsupported configurations that abort the
    // run, not transient per-token fallbacks.
    if n_q_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Err("gpu_attn: zero head geometry".into());
    }
    if n_q_heads * head_dim != q_len || n_kv_heads * head_dim != kv_len {
        return Err(format!(
            "gpu_attn: head geometry mismatch q_len={q_len} kv_len={kv_len} head_dim={head_dim} n_q={n_q_heads} n_kv={n_kv_heads}"
        ));
    }
    if !n_q_heads.is_multiple_of(n_kv_heads) {
        return Err(format!(
            "gpu_attn: GQA divisibility {n_q_heads}%{n_kv_heads}"
        ));
    }
    // head_dim must be a power of two: the flash-attention kernel reduces the
    // per-key dot product with a binary tree (stride = head_dim>>1, ...), which
    // is only exact for power-of-two widths. rope_dim must be even (split-half).
    if !head_dim.is_power_of_two() || head_dim > 256 || rope_dim > head_dim || rope_dim % 2 != 0 {
        return Err(format!(
            "gpu_attn: head_dim {head_dim} / rope_dim {rope_dim} unsupported"
        ));
    }
    if attn_out.len() < q_len {
        return Err("gpu_attn: attn_out too small".into());
    }

    let seq_len = pos + 1;
    let (eff_seq_len, base_row) = if layer_window != 0 && seq_len > layer_window {
        (layer_window, seq_len - layer_window)
    } else {
        (seq_len, 0)
    };
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    with_gpu(|gpu| {
        if kv_layer_idx >= gpu.kv_layers {
            return Err(format!(
                "gpu_attn: kv_layer_idx {kv_layer_idx} out of range (kv_layers {})",
                gpu.kv_layers
            ));
        }
        if gpu.kv_len != kv_len {
            return Err(format!(
                "gpu_attn: device kv_len {} != layer kv_len {kv_len}",
                gpu.kv_len
            ));
        }

        // Ring-buffer wraparound guard (blocker fix). The device KV buffers are
        // sized exactly `kv_context * kv_len`; `kv_append` writes at the physical
        // row `pos % kv_context`, but `flash_attn_decode_kernel` reads LOGICAL
        // rows `[base_row, base_row + eff_seq_len)` directly as physical rows
        // (no modulo). Once the sequence outgrows the context window the read
        // window either goes out of bounds (global layer, base_row==0,
        // eff_seq_len==seq_len) or reads wrapped/overwritten rows (sliding
        // window). Returning Err here removes the illegal device read; the caller
        // (OX_GPU_ATTN whole-run mode) surfaces it as a hard error rather than
        // silently corrupting output, since the host KvCache is not warm.
        if seq_len > gpu.kv_context || base_row + eff_seq_len > gpu.kv_context {
            return Err(format!(
                "gpu_attn: seq_len {seq_len} (base_row {base_row} + eff {eff_seq_len}) exceeds kv_context {} (wraparound unsupported)",
                gpu.kv_context
            ));
        }

        // Upload q/k/v into pooled device buffers (kept device-resident across
        // all three kernels; no intermediate sync).
        let mut d_q = gpu.get_f32_buffer(q_len)?;
        let mut d_k = gpu.get_f32_buffer(kv_len)?;
        let mut d_v = gpu.get_f32_buffer(kv_len)?;
        d_q.copy_from(&q[..q_len]).map_err(stringify)?;
        d_k.copy_from(&k[..kv_len]).map_err(stringify)?;
        d_v.copy_from(&v[..kv_len]).map_err(stringify)?;
        let mut d_attn = gpu.get_f32_buffer(q_len)?;

        let q_ptr = d_q.as_device_ptr();
        let k_ptr = d_k.as_device_ptr();
        let v_ptr = d_v.as_device_ptr();
        let attn_ptr = d_attn.as_device_ptr();

        // 1) partial NeoX RoPE on Q and K (in-place, F32).
        launch_rope_f32(
            gpu,
            q_ptr,
            k_ptr,
            pos as u32,
            n_q_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            rope_dim as u32,
            theta,
        )?;

        // 2) append post-RoPE K/V into the F16 device cache at row pos%context.
        let phys_pos = (pos % gpu.kv_context) as u32;
        launch_kv_append_f16(gpu, kv_layer_idx, k_ptr, v_ptr, phys_pos, pos)?;

        // 3) GQA online-softmax flash-attention decode → d_attn.
        //    Split-K (env OX_GPU_ATTN_SPLITK) partitions the KV sequence across
        //    extra blocks for better SM occupancy on long context; ns == 1
        //    falls back to the byte-for-byte unchanged single-block kernel.
        launch_flash_attn_decode(
            gpu,
            kv_layer_idx,
            q_ptr,
            attn_ptr,
            eff_seq_len as u32,
            base_row as u32,
            n_q_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            scale,
        )?;

        // Single end-of-attention sync + download of the attention result.
        gpu.stream.synchronize().map_err(stringify)?;
        d_attn.copy_to(&mut attn_out[..q_len]).map_err(stringify)?;

        gpu.return_f32_buffer(d_q);
        gpu.return_f32_buffer(d_k);
        gpu.return_f32_buffer(d_v);
        gpu.return_f32_buffer(d_attn);
        Ok(())
    })
}

/// Fully fused, device-resident attention block for one decode token under
/// `OX_GPU_ATTN`.
///
/// Performs the WHOLE attention block inside a SINGLE `with_gpu()` closure with
/// NO `stream.synchronize()` and NO host `copy_to`/`copy_from` anywhere. Q/K/V
/// and the attention output stay in pooled device buffers across every kernel:
///
///   rms_norm(ab.hidden) → ab.normed
///     → Q4K/Q6K GEMV → d_q / d_k / d_v
///     → rope_f32(d_q, d_k)
///     → kv_append_f16(d_k, d_v)
///     → flash_attn_decode(d_q, kv cache) → d_attn
///     → Wo GEMV(d_attn) → ab.normed
///     → residual-add: ab.hidden += ab.normed
///
/// This collapses the three host-marshalling steps (`gpu_attn_rms_and_qkv_q4k`,
/// `gpu_attn_rope_append_flash`, `gpu_wo_residual_q4k`) into one device-resident
/// pass, eliminating the per-layer host round-trips (one sync + 3 D2H, 3 H2D +
/// sync + 1 D2H, 1 H2D). The kernel launches and their order are IDENTICAL to
/// the three functions above; only the data path changes. The `ab.hidden`
/// residual accumulator and the F16 KV cache are the only persistent state
/// mutated; the single per-token D2H (`gpu_download_hidden`) is unchanged.
///
/// Reproduces `gpu_wo_residual_q4k`'s semantics exactly: `normed = Wo·attn`
/// then `hidden += normed`. No post-attention norm is applied (the gpu-native
/// path never runs the Gemma sandwich post-attn norm).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_attn_block_fused_q4k(
    // --- RMS-norm + QKV (from gpu_attn_rms_and_qkv_q4k) ---
    attn_norm: &[f32],
    eps: f32,
    wq: &[u8],
    q_len: usize,
    hidden_size: usize,
    wk: &[u8],
    kv_len: usize,
    wv: &[u8],
    // --- attention geometry (from gpu_attn_rope_append_flash) ---
    head_dim: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    rope_dim: usize,
    theta: f32,
    pos: usize,
    kv_layer_idx: usize,
    layer_window: usize,
    // --- Wo + residual (from gpu_wo_residual_q4k) ---
    wo: &[u8],
    wo_rows: usize,
    wo_cols: usize,
) -> Result<(), String> {
    // --- QKV guards (from gpu_attn_rms_and_qkv_q4k) ---
    if !hidden_size.is_multiple_of(256) {
        return Err(format!(
            "gpu_attn_block_fused: hidden_size {hidden_size} not multiple of 256"
        ));
    }
    let blocks_per_row = hidden_size / 256;
    let bpr_u32 = blocks_per_row as u32;
    let q_u32 = q_len as u32;
    let kv_u32 = kv_len as u32;

    // --- Wo guards (from gpu_wo_residual_q4k) ---
    if !wo_cols.is_multiple_of(256) {
        return Err(format!(
            "gpu_attn_block_fused: wo_cols {wo_cols} not multiple of 256"
        ));
    }
    let wo_bpr = (wo_cols / 256) as u32;
    let wo_bpr_usize = wo_cols / 256;
    let wo_rows_u32 = wo_rows as u32;

    // --- attention geometry guards (from gpu_attn_rope_append_flash) ---
    if n_q_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Err("gpu_attn_block_fused: zero head geometry".into());
    }
    if n_q_heads * head_dim != q_len || n_kv_heads * head_dim != kv_len {
        return Err(format!(
            "gpu_attn_block_fused: head geometry mismatch q_len={q_len} kv_len={kv_len} head_dim={head_dim} n_q={n_q_heads} n_kv={n_kv_heads}"
        ));
    }
    if !n_q_heads.is_multiple_of(n_kv_heads) {
        return Err(format!(
            "gpu_attn_block_fused: GQA divisibility {n_q_heads}%{n_kv_heads}"
        ));
    }
    if !head_dim.is_power_of_two() || head_dim > 256 || rope_dim > head_dim || rope_dim % 2 != 0 {
        return Err(format!(
            "gpu_attn_block_fused: head_dim {head_dim} / rope_dim {rope_dim} unsupported"
        ));
    }

    let seq_len = pos + 1;
    let (eff_seq_len, base_row) = if layer_window != 0 && seq_len > layer_window {
        (layer_window, seq_len - layer_window)
    } else {
        (seq_len, 0)
    };
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    with_gpu(|gpu| {
        // --- Inner attention guards + ring-buffer wraparound (from
        // gpu_attn_rope_append_flash). Placed before any allocation. ---
        if kv_layer_idx >= gpu.kv_layers {
            return Err(format!(
                "gpu_attn_block_fused: kv_layer_idx {kv_layer_idx} out of range (kv_layers {})",
                gpu.kv_layers
            ));
        }
        if gpu.kv_len != kv_len {
            return Err(format!(
                "gpu_attn_block_fused: device kv_len {} != layer kv_len {kv_len}",
                gpu.kv_len
            ));
        }
        if seq_len > gpu.kv_context || base_row + eff_seq_len > gpu.kv_context {
            return Err(format!(
                "gpu_attn_block_fused: seq_len {seq_len} (base_row {base_row} + eff {eff_seq_len}) exceeds kv_context {} (wraparound unsupported)",
                gpu.kv_context
            ));
        }

        // ============================================================
        // (A) RMS-norm: hidden → normed  (from gpu_attn_rms_and_qkv_q4k)
        // ============================================================
        let norm_key = f32_cache_key(attn_norm);
        if !gpu.resident_f32.contains_key(&norm_key) {
            let buf = cust::memory::DeviceBuffer::from_slice(attn_norm).map_err(stringify)?;
            gpu.resident_f32.insert(norm_key, buf);
        }

        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_ptr = ab.hidden.as_device_ptr();
        let normed_ptr = ab.normed.as_device_ptr();
        let hidden_size_u32 = ab.hidden_size as u32;
        let weight_ptr = gpu.resident_f32[&norm_key].as_device_ptr();

        let block_dim = ab.hidden_size.next_power_of_two().min(512) as u32;
        let shmem_attn = block_dim * 4;
        let function_norm = gpu
            .module
            .get_function(RMS_NORM_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(function_norm<<<1, block_dim, shmem_attn, stream>>>(
                hidden_ptr, weight_ptr, normed_ptr, hidden_size_u32, eps
            ))
            .map_err(stringify)?;
        }

        // ============================================================
        // (B) QKV GEMVs: normed → d_q / d_k / d_v  (device-resident)
        // ============================================================
        // Detect Q4K vs Q6K from block byte-size (Q4K=144, Q6K=210 bytes / 256-value block).
        let quant_kern_name = |w: &[u8], rows: usize| -> &'static str {
            let bsz = if rows > 0 && blocks_per_row > 0 {
                w.len() / (rows * blocks_per_row)
            } else {
                144
            };
            if bsz >= 200 {
                GEMV_Q6K_F32IN_KERNEL_NAME
            } else {
                GEMV_Q4K_F32IN_KERNEL_NAME
            }
        };
        let qname_q = quant_kern_name(wq, q_len);
        let qname_k = quant_kern_name(wk, kv_len);
        let qname_v = quant_kern_name(wv, kv_len);

        let q_key = bytes_cache_key(wq);
        gpu.ensure_resident_quant(q_key, wq)?;
        let k_key = bytes_cache_key(wk);
        gpu.ensure_resident_quant(k_key, wk)?;
        let v_key = bytes_cache_key(wv);
        gpu.ensure_resident_quant(v_key, wv)?;

        // Local pooled output buffers — stay device-resident, never downloaded.
        let d_q = gpu.get_f32_buffer(q_len)?;
        let d_k = gpu.get_f32_buffer(kv_len)?;
        let d_v = gpu.get_f32_buffer(kv_len)?;

        let block_size = 256_u32;
        let stream = &gpu.stream;

        // Re-borrow ab.normed after the mutable ensure_resident_quant calls.
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let normed_ptr = ab.normed.as_device_ptr();

        let q_grid = q_u32.saturating_mul(32).div_ceil(block_size);
        let k_grid = kv_u32.saturating_mul(32).div_ceil(block_size);

        let wq_ptr = gpu.resident_quant[&q_key].as_device_ptr();
        let wk_ptr = gpu.resident_quant[&k_key].as_device_ptr();
        let wv_ptr = gpu.resident_quant[&v_key].as_device_ptr();

        let fn_q = gpu.module.get_function(qname_q).map_err(stringify)?;
        let fn_k = gpu.module.get_function(qname_k).map_err(stringify)?;
        let fn_v = gpu.module.get_function(qname_v).map_err(stringify)?;

        // Capture device pointers for the downstream device-resident kernels so
        // the d_q/d_k/d_v owned buffers can be reused while `gpu` is re-borrowed
        // mutably by launch_kv_append_f16.
        let d_q_ptr = d_q.as_device_ptr();
        let d_k_ptr = d_k.as_device_ptr();
        let d_v_ptr = d_v.as_device_ptr();

        unsafe {
            cust::launch!(fn_q<<<q_grid, block_size, 0, stream>>>(
                wq_ptr, normed_ptr, d_q_ptr, q_u32, bpr_u32
            ))
            .map_err(stringify)?;
            cust::launch!(fn_k<<<k_grid, block_size, 0, stream>>>(
                wk_ptr, normed_ptr, d_k_ptr, kv_u32, bpr_u32
            ))
            .map_err(stringify)?;
            cust::launch!(fn_v<<<k_grid, block_size, 0, stream>>>(
                wv_ptr, normed_ptr, d_v_ptr, kv_u32, bpr_u32
            ))
            .map_err(stringify)?;
        }

        // ============================================================
        // (C) partial NeoX RoPE on d_q, d_k (in-place)  (from rope/append/flash)
        // ============================================================
        launch_rope_f32(
            gpu,
            d_q_ptr,
            d_k_ptr,
            pos as u32,
            n_q_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            rope_dim as u32,
            theta,
        )?;

        // ============================================================
        // (D) append post-RoPE K/V into the F16 device cache at row pos%context.
        // ============================================================
        let phys_pos = (pos % gpu.kv_context) as u32;
        launch_kv_append_f16(gpu, kv_layer_idx, d_k_ptr, d_v_ptr, phys_pos, pos)?;

        // ============================================================
        // (E) GQA online-softmax flash-attention decode → d_attn (device-resident).
        // ============================================================
        let d_attn = gpu.get_f32_buffer(q_len)?;
        let d_attn_ptr = d_attn.as_device_ptr();
        // Split-K (env OX_GPU_ATTN_SPLITK) for SM occupancy on long context;
        // ns == 1 falls back to the unchanged single-block kernel. Note the
        // splitk launcher allocates/returns its own pooled scratch internally,
        // which is safe here because `d_attn` is already held and the pool is
        // single-stream-ordered (see launch_flash_attn_decode_splitk).
        launch_flash_attn_decode(
            gpu,
            kv_layer_idx,
            d_q_ptr,
            d_attn_ptr,
            eff_seq_len as u32,
            base_row as u32,
            n_q_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            scale,
        )?;

        // ============================================================
        // (E') Env-gated attention debug dump (OX_ATTN_DUMP), GPU path.
        //
        // Copy the few small device vectors to host ONCE (one-shot, behind the
        // env flag) BEFORE step (F) overwrites `ab.normed` with the Wo result.
        // Same labels / order / sizes as the CPU island in layers.rs so the
        // operator can diff the two files. Debug-only: a single stream sync here
        // is acceptable (perf irrelevant; never runs on the default path).
        // ============================================================
        if crate::attn_dump::should_dump() {
            gpu.stream.synchronize().map_err(stringify)?;
            // `ab.normed` still holds the post-RMSNorm input to QKV at this point.
            let ab = gpu
                .activation
                .as_ref()
                .ok_or_else(|| "activation buffers not initialised".to_string())?;
            let mut norm_in = vec![0.0_f32; ab.hidden_size];
            ab.normed.copy_to(&mut norm_in[..]).map_err(stringify)?;
            let mut q_host = vec![0.0_f32; q_len];
            let mut k_host = vec![0.0_f32; kv_len];
            let mut v_host = vec![0.0_f32; kv_len];
            let mut attn_host = vec![0.0_f32; q_len];
            d_q.copy_to(&mut q_host[..]).map_err(stringify)?;
            d_k.copy_to(&mut k_host[..]).map_err(stringify)?;
            d_v.copy_to(&mut v_host[..]).map_err(stringify)?;
            d_attn.copy_to(&mut attn_host[..]).map_err(stringify)?;
            crate::attn_dump::write_block(
                "gpu",
                pos,
                kv_layer_idx,
                kv_layer_idx,
                &norm_in,
                &q_host,
                &k_host,
                &v_host,
                &attn_host,
            );
        }

        // ============================================================
        // (F) Wo GEMV: d_attn → normed, then residual: hidden += normed
        //     (from gpu_wo_residual_q4k — reads d_attn instead of an upload)
        // ============================================================
        let wo_key = bytes_cache_key(wo);
        gpu.ensure_resident_quant(wo_key, wo)?;

        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let normed_ptr = ab.normed.as_device_ptr();
        let hidden_ptr = ab.hidden.as_device_ptr();
        let hidden_n = ab.hidden_size as u32;

        let wo_grid = wo_rows_u32.saturating_mul(32).div_ceil(block_size);
        let res_grid = hidden_n.div_ceil(block_size);
        let wo_ptr = gpu.resident_quant[&wo_key].as_device_ptr();

        let wo_kern_name =
            if wo_bpr_usize > 0 && wo_rows > 0 && wo.len() / (wo_rows * wo_bpr_usize) >= 200 {
                GEMV_Q6K_F32IN_KERNEL_NAME
            } else {
                GEMV_Q4K_F32IN_KERNEL_NAME
            };
        let fn_wo = gpu.module.get_function(wo_kern_name).map_err(stringify)?;
        let fn_res = gpu
            .module
            .get_function(RESIDUAL_ADD_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;

        unsafe {
            cust::launch!(fn_wo<<<wo_grid, block_size, 0, stream>>>(
                wo_ptr, d_attn_ptr, normed_ptr, wo_rows_u32, wo_bpr
            ))
            .map_err(stringify)?;
            cust::launch!(fn_res<<<res_grid, block_size, 0, stream>>>(
                hidden_ptr, normed_ptr, hidden_n
            ))
            .map_err(stringify)?;
        }

        // ============================================================
        // (G) Return pooled buffers (no sync, no host copy).
        // ============================================================
        gpu.return_f32_buffer(d_q);
        gpu.return_f32_buffer(d_k);
        gpu.return_f32_buffer(d_v);
        gpu.return_f32_buffer(d_attn);
        Ok(())
    })
}
