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
        return Err(format!("gpu_attn_rms_and_qkv: hidden_size {hidden_size} not multiple of 256"));
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

        let ab = gpu.activation.as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_ptr = ab.hidden.as_device_ptr();
        let normed_ptr = ab.normed.as_device_ptr();
        let hidden_size_u32 = ab.hidden_size as u32;
        let weight_ptr = gpu.resident_f32[&norm_key].as_device_ptr();

        let block_dim = ab.hidden_size.next_power_of_two().min(512) as u32;
        let shmem_attn = block_dim * 4;
        let function_norm = gpu.module.get_function(RMS_NORM_KERNEL_NAME).map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(function_norm<<<1, block_dim, shmem_attn, stream>>>(
                hidden_ptr, weight_ptr, normed_ptr, hidden_size_u32, eps
            )).map_err(stringify)?;
        }

        // Detect Q4K vs Q6K from block byte-size (Q4K=144, Q6K=210 bytes per 256-value block).
        let quant_kern_name = |w: &[u8], rows: usize| -> &'static str {
            let bsz = if rows > 0 && blocks_per_row > 0 { w.len() / (rows * blocks_per_row) } else { 144 };
            if bsz >= 200 { GEMV_Q6K_F32IN_KERNEL_NAME } else { GEMV_Q4K_F32IN_KERNEL_NAME }
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

        let ab = gpu.activation.as_ref()
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
            )).map_err(stringify)?;
            cust::launch!(fn_k<<<k_grid, block_size, 0, stream>>>(
                wk_ptr, normed_ptr, d_k.as_device_ptr(), kv_u32, bpr_u32
            )).map_err(stringify)?;
            cust::launch!(fn_v<<<k_grid, block_size, 0, stream>>>(
                wv_ptr, normed_ptr, d_v.as_device_ptr(), kv_u32, bpr_u32
            )).map_err(stringify)?;
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
        return Err(format!("gpu_wo_residual_q4k: cols {cols} not multiple of 256"));
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
        let ab = gpu.activation.as_ref()
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
        let fn_res = gpu.module.get_function(RESIDUAL_ADD_KERNEL_NAME).map_err(stringify)?;
        let stream = &gpu.stream;

        unsafe {
            cust::launch!(fn_wo<<<wo_grid, block_size, 0, stream>>>(
                wo_ptr, d_attn.as_device_ptr(), normed_ptr, rows_u32, bpr
            )).map_err(stringify)?;
            cust::launch!(fn_res<<<res_grid, block_size, 0, stream>>>(
                hidden_ptr, normed_ptr, hidden_n
            )).map_err(stringify)?;
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
        let ab = gpu.activation.as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_ptr = ab.hidden.as_device_ptr();
        let normed_ptr = ab.normed.as_device_ptr();
        let hidden_n = ab.hidden_size as u32;
        let norm_weight_ptr = gpu.resident_f32[&norm_key].as_device_ptr();
        let block_dim = ab.hidden_size.next_power_of_two().min(512) as u32;

        let shmem_ffn = block_dim * 4;
        let fn_norm = gpu.module.get_function(RMS_NORM_KERNEL_NAME).map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(fn_norm<<<1, block_dim, shmem_ffn, stream>>>(
                hidden_ptr, norm_weight_ptr, normed_ptr, hidden_n, eps
            )).map_err(stringify)?;
        }

        // --- Upload FFN weight matrices ---
        let gate_key = bytes_cache_key(gate_w);
        gpu.ensure_resident_quant(gate_key, gate_w)?;
        let up_key = bytes_cache_key(up_w);
        gpu.ensure_resident_quant(up_key, up_w)?;
        let down_key = bytes_cache_key(down_w);
        gpu.ensure_resident_quant(down_key, down_w)?;

        let ab = gpu.activation.as_ref()
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
        let gate_kern = if gate_bpr_usize > 0 && gate_rows > 0 && gate_w.len() / (gate_rows * gate_bpr_usize) >= 200 {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else { GEMV_Q4K_F32IN_KERNEL_NAME };
        let up_kern = if gate_bpr_usize > 0 && up_rows > 0 && up_w.len() / (up_rows * gate_bpr_usize) >= 200 {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else { GEMV_Q4K_F32IN_KERNEL_NAME };
        let down_kern = if down_bpr_usize > 0 && down_rows > 0 && down_w.len() / (down_rows * down_bpr_usize) >= 200 {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else { GEMV_Q4K_F32IN_KERNEL_NAME };
        let fn_gate = gpu.module.get_function(gate_kern).map_err(stringify)?;
        let fn_up   = gpu.module.get_function(up_kern).map_err(stringify)?;
        let fn_down = gpu.module.get_function(down_kern).map_err(stringify)?;
        let fn_silu = gpu.module.get_function(SILU_MUL_KERNEL_NAME).map_err(stringify)?;
        let fn_res  = gpu.module.get_function(RESIDUAL_ADD_KERNEL_NAME).map_err(stringify)?;
        let stream = &gpu.stream;

        // Reuse `normed` as the down-projection output buffer (safe: gate/up
        // GEMVs only READ normed; by the time down runs, normed is free).
        let down_out_ptr = normed_ptr;
        let hidden_ptr = ab.hidden.as_device_ptr();

        unsafe {
            // gate × normed → ffn_gate
            cust::launch!(fn_gate<<<gate_grid, block_size, 0, stream>>>(
                gate_ptr, normed_ptr, gate_buf_ptr, gate_u32, gate_bpr
            )).map_err(stringify)?;
            // up × normed → ffn_up
            cust::launch!(fn_up<<<up_grid, block_size, 0, stream>>>(
                up_ptr, normed_ptr, up_buf_ptr, up_u32, gate_bpr
            )).map_err(stringify)?;
            // silu(ffn_gate) * ffn_up → ffn_down_in
            cust::launch!(fn_silu<<<silu_grid, block_size, 0, stream>>>(
                gate_buf_ptr, up_buf_ptr, ffn_down_in_ptr, inter_n
            )).map_err(stringify)?;
            // down × ffn_down_in → normed (reused as temp)
            cust::launch!(fn_down<<<down_grid, block_size, 0, stream>>>(
                down_ptr, ffn_down_in_ptr, down_out_ptr, down_u32, down_bpr
            )).map_err(stringify)?;
            // hidden += normed (ffn delta)
            cust::launch!(fn_res<<<res_grid, block_size, 0, stream>>>(
                hidden_ptr, down_out_ptr, hidden_n
            )).map_err(stringify)?;
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

    let kern_name = if blocks_per_row > 0
        && rows > 0
        && weight_bytes.len() / (rows * blocks_per_row) >= 200
    {
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
