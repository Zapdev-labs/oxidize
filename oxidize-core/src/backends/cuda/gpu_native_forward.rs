#[allow(unused_imports)]
use super::*;

pub(super) const fn select_projection_gemv_block_size(
    is_q4k: bool,
    rows: u32,
    override_block: Option<u32>,
) -> u32 {
    if !is_q4k {
        return 256;
    }
    match override_block {
        Some(block @ (128 | 256 | 512 | 1024)) => block,
        _ if rows <= 2048 => 128,
        _ if rows <= 8192 => 1024,
        _ => 128,
    }
}

pub(super) fn projection_gemv_block_size(is_q4k: bool, rows: u32) -> u32 {
    static OVERRIDE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    let override_block = *OVERRIDE.get_or_init(|| {
        std::env::var("OX_GPU_GEMV_BLOCK")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
    });
    select_projection_gemv_block_size(is_q4k, rows, override_block)
}

/// Q4_K or Q6_K projection GEMV from a device-resident F32 activation.
/// Routes through [`launch_gemv_f32in_device`] so multi-warp, fused MMQ, and
/// block-size tuning share one dispatch table (fused MMQ used to bypass MW).
#[cfg(feature = "cuda")]
pub(super) fn launch_q4k_or_q6k_projection_gemv(
    gpu: &mut GpuState,
    kern_name: &str,
    w_ptr: cust::memory::DevicePointer<u8>,
    d_input: cust::memory::DevicePointer<f32>,
    d_output: cust::memory::DevicePointer<f32>,
    rows: u32,
    blocks_per_row: u32,
) -> Result<(), String> {
    super::gemv_quantized::launch_gemv_f32in_device(
        gpu,
        kern_name,
        w_ptr,
        d_input,
        d_output,
        rows,
        blocks_per_row,
    )
}

pub(super) fn ox_gpu_fused_residual_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("OX_GPU_FUSED_RESIDUAL")
            .map(|value| value != "0")
            .unwrap_or(false)
    })
}

/// Q4K projection + in-place residual add. Uses fused MMQ (shared Q8_K) when
/// eligible; never routes through the slow f32-in residual kernel.
#[cfg(feature = "cuda")]
pub(super) fn launch_q4k_proj_residual_add(
    gpu: &mut GpuState,
    kern_name: &str,
    w_ptr: cust::memory::DevicePointer<u8>,
    d_input: cust::memory::DevicePointer<f32>,
    d_scratch: cust::memory::DevicePointer<f32>,
    hidden_ptr: cust::memory::DevicePointer<f32>,
    rows: u32,
    blocks_per_row: u32,
    hidden_n: u32,
) -> Result<(), String> {
    if ox_gpu_fused_residual_enabled()
        && kern_name == GEMV_Q4K_F32IN_KERNEL_NAME
        && !super::gemv_quantized::q4k_fused_mmq_eligible(blocks_per_row)
    {
        let wo_block = projection_gemv_block_size(true, rows);
        let wo_grid = rows.saturating_mul(32).div_ceil(wo_block);
        let fn_res = gpu
            .module
            .get_function(residual_projection_kernel_name(kern_name))
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(fn_res<<<wo_grid, wo_block, 0, stream>>>(
                w_ptr, d_input, hidden_ptr, rows, blocks_per_row
            ))
            .map_err(stringify)?;
        }
        return Ok(());
    }

    launch_q4k_or_q6k_projection_gemv(
        gpu,
        kern_name,
        w_ptr,
        d_input,
        d_scratch,
        rows,
        blocks_per_row,
    )?;
    let fn_res = gpu
        .module
        .get_function(RESIDUAL_ADD_KERNEL_NAME)
        .map_err(stringify)?;
    let block_size = 256_u32;
    let res_grid = hidden_n.div_ceil(block_size);
    let stream = &gpu.stream;
    unsafe {
        cust::launch!(fn_res<<<res_grid, block_size, 0, stream>>>(
            hidden_ptr, d_scratch, hidden_n
        ))
        .map_err(stringify)?;
    }
    Ok(())
}

fn residual_projection_kernel_name(projection_kernel: &str) -> &'static str {
    if projection_kernel == GEMV_Q6K_F32IN_KERNEL_NAME {
        GEMV_Q6K_F32IN_RESIDUAL_KERNEL_NAME
    } else {
        GEMV_Q4K_F32IN_RESIDUAL_KERNEL_NAME
    }
}

/// Opt-in: route the Q4_K lm_head through the multi-row kernel that caches the
/// F32 activation in shared memory and computes a tile of rows per block. OFF by
/// default (`OX_GPU_LMHEAD_MULTIROW=1`); numerically identical to the default
/// one-warp-per-row kernel. Ignored when `OX_GPU_FUSED_MMQ` is set (fused MMQ
/// already gives a multi-row DP4A lm_head).
#[cfg(feature = "cuda")]
pub(super) fn ox_gpu_lmhead_multirow_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("OX_GPU_LMHEAD_MULTIROW")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// Opt-in: fuse the SwiGLU FFN gate/up projections + SiLU into a single kernel
/// launch (`OX_GPU_FFN_FUSE=1`). Collapses three per-layer launches (gate GEMV,
/// up GEMV, silu_mul) and two intermediate-size VRAM round-trips into one.
/// Numerically identical; only applies when both gate and up weights are Q4_K.
#[cfg(feature = "cuda")]
crate::cuda::ox_env_flag!(ox_gpu_ffn_fuse_enabled, "OX_GPU_FFN_FUSE", false);

pub(super) const fn select_ffn_fusion(
    gate_is_q4k: bool,
    up_is_q4k: bool,
    rows: usize,
    blocks_per_row: usize,
    enabled: bool,
) -> bool {
    enabled && gate_is_q4k && up_is_q4k && rows > 0 && blocks_per_row > 0
}

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

        let stream = &gpu.stream;

        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let normed_ptr = ab.normed.as_device_ptr();

        let q_block = projection_gemv_block_size(qname_q == GEMV_Q4K_F32IN_KERNEL_NAME, q_u32);
        let k_block = projection_gemv_block_size(qname_k == GEMV_Q4K_F32IN_KERNEL_NAME, kv_u32);
        let v_block = projection_gemv_block_size(qname_v == GEMV_Q4K_F32IN_KERNEL_NAME, kv_u32);
        let q_grid = q_u32.saturating_mul(32).div_ceil(q_block);
        let k_grid = kv_u32.saturating_mul(32).div_ceil(k_block);
        let v_grid = kv_u32.saturating_mul(32).div_ceil(v_block);

        let wq_ptr = gpu.resident_quant[&q_key].as_device_ptr();
        let wk_ptr = gpu.resident_quant[&k_key].as_device_ptr();
        let wv_ptr = gpu.resident_quant[&v_key].as_device_ptr();

        let fn_q = gpu.module.get_function(qname_q).map_err(stringify)?;
        let fn_k = gpu.module.get_function(qname_k).map_err(stringify)?;
        let fn_v = gpu.module.get_function(qname_v).map_err(stringify)?;

        unsafe {
            cust::launch!(fn_q<<<q_grid, q_block, 0, stream>>>(
                wq_ptr, normed_ptr, d_q.as_device_ptr(), q_u32, bpr_u32
            ))
            .map_err(stringify)?;
            cust::launch!(fn_k<<<k_grid, k_block, 0, stream>>>(
                wk_ptr, normed_ptr, d_k.as_device_ptr(), kv_u32, bpr_u32
            ))
            .map_err(stringify)?;
            cust::launch!(fn_v<<<v_grid, v_block, 0, stream>>>(
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

        let residual_block = 256_u32;
        let wo_ptr = gpu.resident_quant[&wo_key].as_device_ptr();

        let bpr_usize = (cols / 256) as usize;
        let wo_kern_name = if bpr_usize > 0 && rows > 0 && wo.len() / (rows * bpr_usize) >= 200 {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else {
            GEMV_Q4K_F32IN_KERNEL_NAME
        };
        let wo_block =
            projection_gemv_block_size(wo_kern_name == GEMV_Q4K_F32IN_KERNEL_NAME, rows_u32);
        let wo_grid = rows_u32.saturating_mul(32).div_ceil(wo_block);
        let res_grid = hidden_n.div_ceil(residual_block);
        let fn_wo = gpu.module.get_function(wo_kern_name).map_err(stringify)?;
        let fn_res = gpu
            .module
            .get_function(RESIDUAL_ADD_KERNEL_NAME)
            .map_err(stringify)?;
        let fn_wo_residual = gpu
            .module
            .get_function(residual_projection_kernel_name(wo_kern_name))
            .map_err(stringify)?;
        let fuse_residual = ox_gpu_fused_residual_enabled();
        let stream = &gpu.stream;

        if fuse_residual
            && wo_kern_name == GEMV_Q4K_F32IN_KERNEL_NAME
            && !super::gemv_quantized::q4k_fused_mmq_eligible(bpr)
        {
            unsafe {
                cust::launch!(fn_wo_residual<<<wo_grid, wo_block, 0, stream>>>(
                    wo_ptr, d_attn.as_device_ptr(), hidden_ptr, rows_u32, bpr
                ))
                .map_err(stringify)?;
            }
        } else {
            launch_q4k_proj_residual_add(
                gpu,
                wo_kern_name,
                wo_ptr,
                d_attn.as_device_ptr(),
                normed_ptr,
                hidden_ptr,
                rows_u32,
                bpr,
                hidden_n,
            )?;
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
        let silu_grid = inter_n.div_ceil(block_size);
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
        let fused_gate_grid = gate_u32.saturating_mul(32).div_ceil(block_size);

        // Reuse `normed` as the down-projection output buffer (safe: gate/up
        // GEMVs only READ normed; by the time down runs, normed is free).
        let down_out_ptr = normed_ptr;
        let hidden_ptr = ab.hidden.as_device_ptr();

        // Fuse gate+up+silu into one launch when enabled and both weights are
        // Q4_K (the dot helper the fused kernel uses is Q4_K-only). gate and up
        // share the [gate_rows × gate_bpr] shape, so one warp-per-row grid over
        // gate_rows writes silu(gate)*up straight into ffn_down_in — no gate/up
        // VRAM round-trip. Byte-identical to the gate→up→silu_mul sequence.
        let fuse_ffn = select_ffn_fusion(
            gate_kern == GEMV_Q4K_F32IN_KERNEL_NAME,
            up_kern == GEMV_Q4K_F32IN_KERNEL_NAME,
            gate_rows,
            gate_bpr as usize,
            ox_gpu_ffn_fuse_enabled(),
        );
        let layer_q8k = super::gemv_quantized::ox_gpu_layer_q8k_enabled()
            && gate_kern == GEMV_Q4K_F32IN_KERNEL_NAME
            && up_kern == GEMV_Q4K_F32IN_KERNEL_NAME
            && down_kern == GEMV_Q4K_F32IN_KERNEL_NAME;
        let q8kin_splits = super::gemv_quantized::ox_gpu_q8kin_splits();
        let fn_gate_up_silu = fuse_ffn && !layer_q8k;

        if layer_q8k {
            let xq8k_ptr = ab.xq8k.as_device_ptr();
            let xq8k_ffn_ptr = ab.xq8k_ffn.as_device_ptr();
            super::gemv_quantized::launch_quantize_f32_to_q8k_device_ptr(
                gpu, normed_ptr, xq8k_ptr, gate_bpr,
            )?;
            super::gemv_quantized::launch_gemv_q4k_q8kin_device(
                gpu,
                gate_ptr,
                gate_u32,
                gate_bpr,
                xq8k_ptr,
                gate_buf_ptr,
                q8kin_splits,
            )?;
            super::gemv_quantized::launch_gemv_q4k_q8kin_device(
                gpu,
                up_ptr,
                up_u32,
                gate_bpr,
                xq8k_ptr,
                up_buf_ptr,
                q8kin_splits,
            )?;
            let fn_silu = gpu
                .module
                .get_function(SILU_MUL_KERNEL_NAME)
                .map_err(stringify)?;
            let stream = &gpu.stream;
            unsafe {
                cust::launch!(fn_silu<<<silu_grid, block_size, 0, stream>>>(
                    gate_buf_ptr, up_buf_ptr, ffn_down_in_ptr, inter_n
                ))
                .map_err(stringify)?;
            }
            super::gemv_quantized::launch_quantize_f32_to_q8k_device_ptr(
                gpu,
                ffn_down_in_ptr,
                xq8k_ffn_ptr,
                down_bpr,
            )?;
            super::gemv_quantized::launch_gemv_q4k_q8kin_device(
                gpu,
                down_ptr,
                down_u32,
                down_bpr,
                xq8k_ffn_ptr,
                down_out_ptr,
                q8kin_splits,
            )?;
            let fn_res = gpu
                .module
                .get_function(RESIDUAL_ADD_KERNEL_NAME)
                .map_err(stringify)?;
            let stream = &gpu.stream;
            unsafe {
                cust::launch!(fn_res<<<res_grid, block_size, 0, stream>>>(
                    hidden_ptr, down_out_ptr, hidden_n
                ))
                .map_err(stringify)?;
            }
        } else {
            if fn_gate_up_silu {
                let fn_fused = gpu
                    .module
                    .get_function(GEMV_Q4K_F32IN_GATE_UP_SILU_KERNEL_NAME)
                    .map_err(stringify)?;
                let stream = &gpu.stream;
                unsafe {
                    cust::launch!(fn_fused<<<fused_gate_grid, block_size, 0, stream>>>(
                        gate_ptr, up_ptr, normed_ptr, ffn_down_in_ptr, gate_u32, gate_bpr
                    ))
                    .map_err(stringify)?;
                }
            } else {
                launch_q4k_or_q6k_projection_gemv(
                    gpu,
                    gate_kern,
                    gate_ptr,
                    normed_ptr,
                    gate_buf_ptr,
                    gate_u32,
                    gate_bpr,
                )?;
                launch_q4k_or_q6k_projection_gemv(
                    gpu, up_kern, up_ptr, normed_ptr, up_buf_ptr, up_u32, gate_bpr,
                )?;
                let fn_silu = gpu
                    .module
                    .get_function(SILU_MUL_KERNEL_NAME)
                    .map_err(stringify)?;
                let stream = &gpu.stream;
                unsafe {
                    cust::launch!(fn_silu<<<silu_grid, block_size, 0, stream>>>(
                        gate_buf_ptr, up_buf_ptr, ffn_down_in_ptr, inter_n
                    ))
                    .map_err(stringify)?;
                }
            }
            launch_q4k_proj_residual_add(
                gpu,
                down_kern,
                down_ptr,
                ffn_down_in_ptr,
                down_out_ptr,
                hidden_ptr,
                down_u32,
                down_bpr,
                hidden_n,
            )?;
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

        let block_size =
            projection_gemv_block_size(kern_name == GEMV_Q4K_F32IN_KERNEL_NAME, rows_u32);
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

/// Device-resident lm_head: RMSNorm on `activation.hidden`, GEMV from
/// `activation.normed` → logits, single sync at the end.
///
/// Used when the hidden state never left the GPU during the layer stack
/// (gpu_native decode). Avoids re-uploading the normed vector for lm_head.
#[cfg(feature = "cuda")]
pub fn gpu_final_head_device_resident(
    norm_weight: &[f32],
    eps: f32,
    weight_bytes: &[u8],
    vocab_size: usize,
    hidden_size: usize,
    logits: &mut [f32],
    last_hidden: &mut [f32],
) -> Result<(), String> {
    if !hidden_size.is_multiple_of(256) {
        return Err(format!(
            "gpu_final_head: hidden_size {hidden_size} not a multiple of 256"
        ));
    }
    if logits.len() != vocab_size || last_hidden.len() != hidden_size {
        return Err("gpu_final_head: output buffer size mismatch".to_string());
    }

    super::gpu_kernels::gpu_rms_norm(norm_weight, eps)?;

    let blocks_per_row = hidden_size / 256;
    let bpr_u32 = blocks_per_row as u32;
    let rows_u32 = vocab_size as u32;
    let is_q6k = blocks_per_row > 0
        && vocab_size > 0
        && weight_bytes.len() / (vocab_size * blocks_per_row) >= 200;

    // lm_head: Q8K-in MW (default wide vocab), fused MMQ, multi-row f32-in, or MW f32-in.
    enum LmHeadKern {
        F32In(&'static str),
        Multirow,
        FusedMmq,
        Q8kinMw,
    }
    let use_lmhead_q8k = !is_q6k
        && super::gemv_quantized::ox_gpu_gemv_mw_enabled()
        && (super::gemv_quantized::ox_gpu_lmhead_q8k_enabled()
            || super::gemv_quantized::ox_gpu_layer_q8k_enabled());
    let strategy = if is_q6k {
        LmHeadKern::F32In(GEMV_Q6K_F32IN_KERNEL_NAME)
    } else if super::gemv_quantized::ox_gpu_fused_mmq_enabled()
        && super::gemv_quantized::q4k_fused_mmq_eligible(bpr_u32)
    {
        LmHeadKern::FusedMmq
    } else if use_lmhead_q8k {
        LmHeadKern::Q8kinMw
    } else if (ox_gpu_lmhead_multirow_enabled() || rows_u32 >= 8192)
        && !super::gemv_quantized::ox_gpu_gemv_mw_enabled()
        && hidden_size.saturating_mul(4) <= 44 * 1024
    {
        LmHeadKern::Multirow
    } else {
        LmHeadKern::F32In(GEMV_Q4K_F32IN_KERNEL_NAME)
    };

    with_gpu(|gpu| {
        let w_key = bytes_cache_key(weight_bytes);
        gpu.ensure_resident_quant(w_key, weight_bytes)?;

        let normed_ptr = {
            let ab = gpu
                .activation
                .as_ref()
                .ok_or_else(|| "activation buffers not initialised".to_string())?;
            ab.normed.as_device_ptr()
        };

        let d_output = gpu.get_f32_buffer(vocab_size)?;
        let block_size = 256_u32;
        let grid = rows_u32.saturating_mul(32).div_ceil(block_size);
        let w_ptr = gpu.resident_quant[&w_key].as_device_ptr();

        match strategy {
            LmHeadKern::Q8kinMw => {
                let xq8k_ptr = gpu
                    .activation
                    .as_ref()
                    .ok_or_else(|| "activation buffers not initialised".to_string())?
                    .xq8k
                    .as_device_ptr();
                super::gemv_quantized::launch_quantize_f32_to_q8k_device_ptr(
                    gpu, normed_ptr, xq8k_ptr, bpr_u32,
                )?;
                super::gemv_quantized::launch_gemv_q4k_q8kin_mw_device(
                    gpu,
                    w_ptr,
                    rows_u32,
                    bpr_u32,
                    xq8k_ptr,
                    d_output.as_device_ptr(),
                )?;
            }
            LmHeadKern::FusedMmq => {
                super::gemv_quantized::launch_fused_mmq_device(
                    gpu,
                    w_ptr,
                    normed_ptr,
                    d_output.as_device_ptr(),
                    rows_u32,
                    bpr_u32,
                )?;
            }
            LmHeadKern::Multirow => {
                let shmem = (hidden_size * std::mem::size_of::<f32>()) as u32;
                let fn_gemv = gpu
                    .module
                    .get_function(GEMV_Q4K_F32IN_MULTIROW_KERNEL_NAME)
                    .map_err(stringify)?;
                let stream = &gpu.stream;
                unsafe {
                    cust::launch!(fn_gemv<<<grid, block_size, shmem, stream>>>(
                        w_ptr, normed_ptr, d_output.as_device_ptr(), rows_u32, bpr_u32
                    ))
                    .map_err(stringify)?;
                }
            }
            LmHeadKern::F32In(kern_name) => {
                super::launch_q4k_or_q6k_projection_gemv(
                    gpu,
                    kern_name,
                    w_ptr,
                    normed_ptr,
                    d_output.as_device_ptr(),
                    rows_u32,
                    bpr_u32,
                )?;
            }
        }

        gpu.stream.synchronize().map_err(stringify)?;
        d_output.copy_to(logits).map_err(stringify)?;
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        ab.normed.copy_to(last_hidden).map_err(stringify)?;
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
        //    Decode dispatch automatically partitions long KV sequences across
        //    extra blocks for better SM occupancy and keeps short contexts on
        //    the unchanged single-block kernel.
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
    // --- per-head Q/K RMSNorm (Qwen3-style QK-norm). Empty slices = no norm.
    //     When present, length MUST equal head_dim; applied per head AFTER the
    //     QKV projection and BEFORE RoPE, matching the CPU rms_norm_f32 path. ---
    q_norm: &[f32],
    k_norm: &[f32],
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
    // QK-norm (if present) must be a per-head [head_dim] weight — the rms_norm
    // launch below assumes one block per head over exactly head_dim elements.
    if !q_norm.is_empty() && q_norm.len() != head_dim {
        return Err(format!(
            "gpu_attn_block_fused: q_norm len {} != head_dim {head_dim}",
            q_norm.len()
        ));
    }
    if !k_norm.is_empty() && k_norm.len() != head_dim {
        return Err(format!(
            "gpu_attn_block_fused: k_norm len {} != head_dim {head_dim}",
            k_norm.len()
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

        // Re-borrow ab.normed after the mutable ensure_resident_quant calls.
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let normed_ptr = ab.normed.as_device_ptr();

        let wq_ptr = gpu.resident_quant[&q_key].as_device_ptr();
        let wk_ptr = gpu.resident_quant[&k_key].as_device_ptr();
        let wv_ptr = gpu.resident_quant[&v_key].as_device_ptr();

        let d_q_ptr = d_q.as_device_ptr();
        let d_k_ptr = d_k.as_device_ptr();
        let d_v_ptr = d_v.as_device_ptr();

        let layer_q8k = super::gemv_quantized::ox_gpu_layer_q8k_enabled();
        let all_q4k = qname_q == GEMV_Q4K_F32IN_KERNEL_NAME
            && qname_k == GEMV_Q4K_F32IN_KERNEL_NAME
            && qname_v == GEMV_Q4K_F32IN_KERNEL_NAME;
        let q8kin_splits = super::gemv_quantized::ox_gpu_q8kin_splits();

        if layer_q8k && all_q4k {
            let xq8k_ptr = gpu
                .activation
                .as_ref()
                .ok_or_else(|| "activation buffers not initialised".to_string())?
                .xq8k
                .as_device_ptr();
            super::gemv_quantized::launch_quantize_f32_to_q8k_device_ptr(
                gpu, normed_ptr, xq8k_ptr, bpr_u32,
            )?;
            super::gemv_quantized::launch_gemv_q4k_q8kin_device(
                gpu,
                wq_ptr,
                q_u32,
                bpr_u32,
                xq8k_ptr,
                d_q_ptr,
                q8kin_splits,
            )?;
            super::gemv_quantized::launch_gemv_q4k_q8kin_device(
                gpu,
                wk_ptr,
                kv_u32,
                bpr_u32,
                xq8k_ptr,
                d_k_ptr,
                q8kin_splits,
            )?;
            super::gemv_quantized::launch_gemv_q4k_q8kin_device(
                gpu,
                wv_ptr,
                kv_u32,
                bpr_u32,
                xq8k_ptr,
                d_v_ptr,
                q8kin_splits,
            )?;
        } else {
            let q4k_fused_qkv = super::gemv_quantized::ox_gpu_fused_qkv_enabled()
                && super::gemv_quantized::ox_gpu_gemv_mw_enabled()
                && qname_q == GEMV_Q4K_F32IN_KERNEL_NAME
                && qname_k == GEMV_Q4K_F32IN_KERNEL_NAME
                && qname_v == GEMV_Q4K_F32IN_KERNEL_NAME;

            if q4k_fused_qkv {
                let xq8k_ptr = gpu
                    .activation
                    .as_ref()
                    .ok_or_else(|| "activation buffers not initialised".to_string())?
                    .xq8k
                    .as_device_ptr();
                super::gemv_quantized::launch_quantize_f32_to_q8k_device_ptr(
                    gpu, normed_ptr, xq8k_ptr, bpr_u32,
                )?;
                super::gemv_quantized::launch_gemv_q4k_q8kin_qkv_mw_device(
                    gpu, wq_ptr, wk_ptr, wv_ptr, xq8k_ptr, d_q_ptr, d_k_ptr, d_v_ptr, q_u32,
                    kv_u32, bpr_u32,
                )?;
            } else if super::gemv_quantized::ox_gpu_fused_qkv_enabled()
                && super::gemv_quantized::ox_gpu_fused_mmq_enabled()
                && qname_q == GEMV_Q4K_F32IN_KERNEL_NAME
                && qname_k == GEMV_Q4K_F32IN_KERNEL_NAME
                && qname_v == GEMV_Q4K_F32IN_KERNEL_NAME
                && super::gemv_quantized::q4k_fused_mmq_eligible(bpr_u32)
            {
                super::gemv_quantized::launch_gemv_q4k_q8k_fused_qkv_mw_device(
                    gpu, wq_ptr, wk_ptr, wv_ptr, normed_ptr, d_q_ptr, d_k_ptr, d_v_ptr, q_u32,
                    kv_u32, bpr_u32,
                )?;
            } else {
                super::launch_q4k_or_q6k_projection_gemv(
                    gpu, qname_q, wq_ptr, normed_ptr, d_q_ptr, q_u32, bpr_u32,
                )?;
                super::launch_q4k_or_q6k_projection_gemv(
                    gpu, qname_k, wk_ptr, normed_ptr, d_k_ptr, kv_u32, bpr_u32,
                )?;
                super::launch_q4k_or_q6k_projection_gemv(
                    gpu, qname_v, wv_ptr, normed_ptr, d_v_ptr, kv_u32, bpr_u32,
                )?;
            }
        }

        // ============================================================
        // (B') per-head Q/K RMSNorm (Qwen3-style QK-norm), in-place on d_q/d_k,
        //      applied AFTER the QKV projection and BEFORE RoPE. Reuses the
        //      per-row rms_norm_f32_kernel with one block per head (rows =
        //      n_heads, hidden_size = head_dim, reading the shared [head_dim]
        //      norm weight) — numerically equivalent to the CPU rms_norm_f32
        //      applied per head in layers.rs. Inert (skipped) for models without
        //      QK-norm (Llama/Mistral pass empty slices).
        // ============================================================
        if !q_norm.is_empty() || !k_norm.is_empty() {
            if !q_norm.is_empty() {
                let qn_key = f32_cache_key(q_norm);
                if !gpu.resident_f32.contains_key(&qn_key) {
                    let buf = cust::memory::DeviceBuffer::from_slice(q_norm).map_err(stringify)?;
                    gpu.resident_f32.insert(qn_key, buf);
                }
            }
            if !k_norm.is_empty() {
                let kn_key = f32_cache_key(k_norm);
                if !gpu.resident_f32.contains_key(&kn_key) {
                    let buf = cust::memory::DeviceBuffer::from_slice(k_norm).map_err(stringify)?;
                    gpu.resident_f32.insert(kn_key, buf);
                }
            }
            let hd_u32 = head_dim as u32;
            let shmem_norm = hd_u32 * 4;
            let fn_rms = gpu
                .module
                .get_function(RMS_NORM_KERNEL_NAME)
                .map_err(stringify)?;
            let stream = &gpu.stream;
            unsafe {
                if !q_norm.is_empty() {
                    let qn_ptr = gpu.resident_f32[&f32_cache_key(q_norm)].as_device_ptr();
                    cust::launch!(fn_rms<<<n_q_heads as u32, hd_u32, shmem_norm, stream>>>(
                        d_q_ptr, qn_ptr, d_q_ptr, hd_u32, eps
                    ))
                    .map_err(stringify)?;
                }
                if !k_norm.is_empty() {
                    let kn_ptr = gpu.resident_f32[&f32_cache_key(k_norm)].as_device_ptr();
                    cust::launch!(fn_rms<<<n_kv_heads as u32, hd_u32, shmem_norm, stream>>>(
                        d_k_ptr, kn_ptr, d_k_ptr, hd_u32, eps
                    ))
                    .map_err(stringify)?;
                }
            }
        }

        let d_attn = gpu.get_f32_buffer(q_len)?;
        let d_attn_ptr = d_attn.as_device_ptr();

        // ============================================================
        // (C–E) RoPE + KV append + flash decode (GPH kernels under CUDA graph).
        // ============================================================
        if super::cuda_decode_graph::decode_graph_use_gph(pos, gpu.kv_context) {
            let d_state = super::cuda_decode_graph::decode_d_state_ptr(gpu)?;
            super::cuda_decode_graph::launch_rope_f32_gph(
                gpu,
                d_state,
                d_q_ptr,
                d_k_ptr,
                n_q_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                rope_dim as u32,
                theta,
            )?;
            super::cuda_decode_graph::launch_kv_append_f16_gph(
                gpu,
                kv_layer_idx,
                d_state,
                d_k_ptr,
                d_v_ptr,
            )?;
            super::cuda_decode_graph::update_kv_seq_len_after_gph_append(gpu, kv_layer_idx, pos);
            super::cuda_decode_graph::launch_flash_attn_decode_gph(
                gpu,
                kv_layer_idx,
                d_state,
                d_q_ptr,
                d_attn_ptr,
                layer_window as u32,
                n_q_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                scale,
            )?;
        } else {
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
            let phys_pos = (pos % gpu.kv_context) as u32;
            launch_kv_append_f16(gpu, kv_layer_idx, d_k_ptr, d_v_ptr, phys_pos, pos)?;
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
        }

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

        let res_grid = hidden_n.div_ceil(block_size);
        let wo_ptr = gpu.resident_quant[&wo_key].as_device_ptr();

        let wo_kern_name =
            if wo_bpr_usize > 0 && wo_rows > 0 && wo.len() / (wo_rows * wo_bpr_usize) >= 200 {
                GEMV_Q6K_F32IN_KERNEL_NAME
            } else {
                GEMV_Q4K_F32IN_KERNEL_NAME
            };
        let q8kin_splits = super::gemv_quantized::ox_gpu_q8kin_splits();

        let wo_layer_q8k = super::gemv_quantized::ox_gpu_layer_q8k_enabled()
            && wo_kern_name == GEMV_Q4K_F32IN_KERNEL_NAME;

        if wo_layer_q8k {
            let xq8k_ptr = gpu
                .activation
                .as_ref()
                .ok_or_else(|| "activation buffers not initialised".to_string())?
                .xq8k
                .as_device_ptr();
            super::gemv_quantized::launch_quantize_f32_to_q8k_device_ptr(
                gpu, d_attn_ptr, xq8k_ptr, wo_bpr,
            )?;
            super::gemv_quantized::launch_gemv_q4k_q8kin_device(
                gpu,
                wo_ptr,
                wo_rows_u32,
                wo_bpr,
                xq8k_ptr,
                normed_ptr,
                q8kin_splits,
            )?;
            let fn_res = gpu
                .module
                .get_function(RESIDUAL_ADD_KERNEL_NAME)
                .map_err(stringify)?;
            let stream = &gpu.stream;
            unsafe {
                cust::launch!(fn_res<<<res_grid, block_size, 0, stream>>>(
                    hidden_ptr, normed_ptr, hidden_n
                ))
                .map_err(stringify)?;
            }
        } else {
            launch_q4k_proj_residual_add(
                gpu,
                wo_kern_name,
                wo_ptr,
                d_attn_ptr,
                normed_ptr,
                hidden_ptr,
                wo_rows_u32,
                wo_bpr,
                hidden_n,
            )?;
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

// ===========================================================================
// Batched device decode forward (OX_GPU_BATCHED, B <= 8).
//
// Serves B concurrent decode rows (one token each, distinct sequences) in one
// weighted pass per projection matrix using the bN Q4_K×Q8_K GEMV, plus a
// per-seq attention loop reusing the single-seq RoPE / kv-append / flash-decode
// kernels. Elementwise ops (rmsnorm / residual / silu) loop the existing
// single-row / flat launchers B times. Entirely separate device state
// (`batched_activation` + `kv_*_batched`) from the single-token path, so the
// default forward is byte-identical. Reached only via the (out-of-this-stage)
// `forward_batch_gpu` under `OX_GPU_BATCHED=1`.
// ===========================================================================

/// Maximum batch the bN kernel (`gemv_q4k_q8kin_bN_kernel`, `ncols` limit)
/// serves in this MVP. Callers reject larger batches and loop single-seq.
#[cfg(feature = "cuda")]
pub const GPU_BATCHED_MAX_B: usize = 8;

/// Lazily allocate (or reallocate on geometry/B change) the batched activation
/// buffers. Mirrors [`gpu_init_activation_buffers`]: a no-op when the geometry
/// and batch already match. `hidden_size` and `intermediate_size` must be
/// multiples of 256 (Q8_K block size); `bn_rows` is the largest projection row
/// count that the bN-output scratch must hold (max of q_len/kv_len/inter and any
/// lm-head row count the caller drives through the batched path).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_batched_activation_init(
    batch: usize,
    hidden_size: usize,
    intermediate_size: usize,
    q_len: usize,
    kv_len: usize,
    bn_rows: usize,
) -> Result<(), String> {
    if batch == 0 || batch > GPU_BATCHED_MAX_B {
        return Err(format!(
            "gpu_batched_activation_init: batch {batch} out of range (1..={GPU_BATCHED_MAX_B})"
        ));
    }
    if !hidden_size.is_multiple_of(256) || !intermediate_size.is_multiple_of(256) {
        return Err(format!(
            "gpu_batched_activation_init: hidden_size {hidden_size} / intermediate_size {intermediate_size} must be multiples of 256"
        ));
    }
    if hidden_size == 0 || intermediate_size == 0 || q_len == 0 || kv_len == 0 || bn_rows == 0 {
        return Err("gpu_batched_activation_init: zero geometry".to_string());
    }
    with_gpu(|gpu| {
        let bn_cap = bn_rows.max(q_len).max(kv_len).max(intermediate_size);
        if let Some(ref ab) = gpu.batched_activation {
            if ab.batch == batch
                && ab.hidden_size == hidden_size
                && ab.intermediate_size == intermediate_size
                && ab.q_len == q_len
                && ab.kv_len == kv_len
                && ab.bn_rows >= bn_cap
            {
                return Ok(());
            }
        }
        let bpr_hidden = hidden_size / 256;
        let bpr_inter = intermediate_size / 256;
        let z = |n: usize| cust::memory::DeviceBuffer::<f32>::zeroed(n).map_err(stringify);
        let zu = |n: usize| cust::memory::DeviceBuffer::<u8>::zeroed(n).map_err(stringify);
        gpu.batched_activation = Some(GpuBatchedActivation {
            hidden: z(batch * hidden_size)?,
            normed: z(batch * hidden_size)?,
            q: z(batch * q_len)?,
            k: z(batch * kv_len)?,
            v: z(batch * kv_len)?,
            attn: z(batch * q_len)?,
            attn_proj: z(batch * hidden_size)?,
            ffn_gate: z(batch * intermediate_size)?,
            ffn_up: z(batch * intermediate_size)?,
            ffn_down: z(batch * intermediate_size)?,
            bn_out: z(batch * bn_cap)?,
            xq8k: zu(batch * bpr_hidden * BLOCK_Q8_K_BYTES)?,
            xq8k_ffn: zu(batch * bpr_inter * BLOCK_Q8_K_BYTES)?,
            batch,
            hidden_size,
            intermediate_size,
            q_len,
            kv_len,
            bn_rows: bn_cap,
        });
        Ok(())
    })
}

/// Allocate (or reset on geometry/B change) the batched per-layer KV cache.
/// Each layer buffer holds `batch` contiguous sequence regions, each sized
/// `context_size * kv_len` u16 (mirroring [`gpu_kv_init`] per region). A no-op
/// when geometry and batch already match (the seq-length counters are left
/// MONOTONIC; a new prompt must call [`gpu_kv_batched_reset`]).
#[cfg(feature = "cuda")]
pub fn gpu_kv_batched_init(
    num_layers: usize,
    kv_len: usize,
    context_size: usize,
    batch: usize,
) -> Result<(), String> {
    if num_layers == 0 || kv_len == 0 || context_size == 0 || batch == 0 {
        return Err(format!(
            "gpu_kv_batched_init: invalid geometry num_layers={num_layers} kv_len={kv_len} context_size={context_size} batch={batch}"
        ));
    }
    if batch > GPU_BATCHED_MAX_B {
        return Err(format!(
            "gpu_kv_batched_init: batch {batch} exceeds max {GPU_BATCHED_MAX_B}"
        ));
    }
    with_gpu(|gpu| {
        let geom_match = gpu.kv_batched_b == batch
            && gpu.kv_k_batched.len() == num_layers
            && gpu.kv_len == kv_len
            && gpu.kv_context == context_size;
        if geom_match {
            return Ok(());
        }
        let region = context_size
            .checked_mul(kv_len)
            .ok_or_else(|| "gpu_kv_batched_init: context_size*kv_len overflow".to_string())?;
        let elems = region
            .checked_mul(batch)
            .ok_or_else(|| "gpu_kv_batched_init: region*batch overflow".to_string())?;
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_cache.push(cust::memory::DeviceBuffer::<u16>::zeroed(elems).map_err(stringify)?);
            v_cache.push(cust::memory::DeviceBuffer::<u16>::zeroed(elems).map_err(stringify)?);
        }
        gpu.kv_k_batched = k_cache;
        gpu.kv_v_batched = v_cache;
        gpu.kv_batched_b = batch;
        gpu.kv_batched_seq_len = vec![vec![0usize; batch]; num_layers];
        // Keep the shared geometry fields consistent for the validators.
        gpu.kv_len = kv_len;
        gpu.kv_context = context_size;
        Ok(())
    })
}

/// Reset every batched `[layer][seq]` sequence-length counter to 0 (new prompt).
/// Buffers stay allocated; the counters gate which rows are read.
#[cfg(feature = "cuda")]
pub fn gpu_kv_batched_reset() -> Result<(), String> {
    with_gpu(|gpu| {
        for layer in gpu.kv_batched_seq_len.iter_mut() {
            for s in layer.iter_mut() {
                *s = 0;
            }
        }
        Ok(())
    })
}

/// Batched KV-append: cast sequence `seq`'s post-RoPE F32 K/V into row
/// `phys_pos` of its own region inside the batched layer buffer, then bump
/// `kv_batched_seq_len[layer][seq]`. The single-seq `kv_append_f16_kernel` is
/// reused with the cache pointers offset by the region base (`base_off*kv_len`).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_kv_append_f16_batched(
    gpu: &mut GpuState,
    kv_layer_idx: usize,
    seq: usize,
    d_k: cust::memory::DevicePointer<f32>,
    d_v: cust::memory::DevicePointer<f32>,
    phys_pos: u32,
    logical_pos: usize,
) -> Result<(), String> {
    if kv_layer_idx >= gpu.kv_k_batched.len() {
        return Err(format!(
            "launch_kv_append_f16_batched: layer {kv_layer_idx} out of range ({})",
            gpu.kv_k_batched.len()
        ));
    }
    if seq >= gpu.kv_batched_b {
        return Err(format!(
            "launch_kv_append_f16_batched: seq {seq} out of range ({})",
            gpu.kv_batched_b
        ));
    }
    let kv_len = gpu.kv_len as u32;
    let context = gpu.kv_context;
    let base_off = seq
        .checked_mul(context)
        .and_then(|r| r.checked_mul(gpu.kv_len))
        .ok_or_else(|| "launch_kv_append_f16_batched: region offset overflow".to_string())?;
    let func = gpu
        .module
        .get_function(KV_APPEND_F16_KERNEL_NAME)
        .map_err(stringify)?;
    let kc_ptr = gpu.kv_k_batched[kv_layer_idx]
        .as_device_ptr()
        .wrapping_add(base_off);
    let vc_ptr = gpu.kv_v_batched[kv_layer_idx]
        .as_device_ptr()
        .wrapping_add(base_off);
    let block = 256_u32;
    let grid = kv_len.div_ceil(block);
    let stream = &gpu.stream;
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(
            d_k, d_v, kc_ptr, vc_ptr, phys_pos, kv_len
        ))
        .map_err(stringify)?;
    }
    if let Some(layer) = gpu.kv_batched_seq_len.get_mut(kv_layer_idx) {
        if let Some(s) = layer.get_mut(seq) {
            *s = (logical_pos + 1).min(context);
        }
    }
    Ok(())
}

/// Borrowed view of one transformer layer's weights + per-layer scalars for the
/// batched device forward. Weight byte slices are Q4_K-quantized (validated by
/// the caller's eligibility gate). Norm slices are F32; QK-norm slices are empty
/// for non-Qwen3 layers.
#[cfg(feature = "cuda")]
pub struct BatchedLayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub eps: f32,
    pub wq: &'a [u8],
    pub wk: &'a [u8],
    pub wv: &'a [u8],
    pub wo: &'a [u8],
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub down: &'a [u8],
    /// Per-head Q/K RMSNorm (Qwen3); empty = none. Length must equal head_dim.
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub kv_layer_idx: usize,
    pub layer_window: usize,
    pub theta: f32,
}

/// Geometry shared by every layer of the batched forward.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub struct BatchedGeom {
    pub batch: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub q_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub rope_dim: usize,
}

/// Ensure a per-row F32 norm weight is resident; return its device pointer.
#[cfg(feature = "cuda")]
fn resident_f32_ptr(
    gpu: &mut GpuState,
    w: &[f32],
) -> Result<cust::memory::DevicePointer<f32>, String> {
    let key = f32_cache_key(w);
    if !gpu.resident_f32.contains_key(&key) {
        let buf = cust::memory::DeviceBuffer::from_slice(w).map_err(stringify)?;
        gpu.resident_f32.insert(key, buf);
    }
    Ok(gpu.resident_f32[&key].as_device_ptr())
}

/// Run ONE transformer layer of the batched forward on `gpu.stream` with no
/// host sync. Reads/writes `batched_activation.hidden` (the B-row residual
/// accumulator) and the batched F16 KV cache. `pos[b]` is sequence `b`'s
/// position; `kv_seq_len_pre[b]` is its token count BEFORE this step.
///
/// Launch sequence (B = batch):
///   (1) attn RMSNorm: B× rms_norm_f32_kernel (hidden[b] -> normed[b])
///   (2) quantize normed -> xq8k (B× quantize_f32_to_q8k, one column each)
///   (3) QKV bN GEMVs (1 weight pass each) -> bn_out[rows,B] -> transpose -> q/k/v[B,rows]
///   (3') optional Qwen3 QK-norm: B× per-head rms over q/k
///   (4) per-seq: rope -> kv_append(batched) -> flash_decode(batched) -> attn[b]
///   (5) Wo: quantize attn (B cols) -> bN -> transpose -> attn_proj[B,h]
///   (6) residual: hidden += attn_proj (flat over B*h)
///   (7) ffn RMSNorm: B× (hidden[b] -> normed[b])
///   (8) quantize normed -> xq8k
///   (9) gate/up bN -> transpose -> ffn_gate/ffn_up [B,inter]
///  (10) silu_mul (flat over B*inter) -> ffn_down
///  (11) quantize ffn_down -> xq8k_ffn ; down bN -> transpose -> normed[B,h]
///  (12) residual: hidden += normed (flat over B*h)
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_forward_batch_layer(
    w: &BatchedLayerWeights<'_>,
    geom: BatchedGeom,
    pos: &[usize],
    kv_seq_len_pre: &[usize],
) -> Result<(), String> {
    let b = geom.batch;
    if pos.len() != b || kv_seq_len_pre.len() != b {
        return Err("gpu_forward_batch_layer: pos/kv_seq_len_pre length != batch".to_string());
    }
    if !geom.hidden_size.is_multiple_of(256) || !geom.intermediate_size.is_multiple_of(256) {
        return Err("gpu_forward_batch_layer: hidden/intermediate not multiple of 256".to_string());
    }
    let bpr_hidden = (geom.hidden_size / 256) as u32;
    let bpr_inter = (geom.intermediate_size / 256) as u32;
    let h_u32 = geom.hidden_size as u32;
    let q_u32 = geom.q_len as u32;
    let kv_u32 = geom.kv_len as u32;
    let inter_u32 = geom.intermediate_size as u32;
    let b_u32 = b as u32;
    let scale = 1.0_f32 / (geom.head_dim as f32).sqrt();

    with_gpu(|gpu| {
        // Resolve resident weight pointers / norm pointers up front.
        let attn_norm_ptr = resident_f32_ptr(gpu, w.attn_norm)?;
        let ffn_norm_ptr = resident_f32_ptr(gpu, w.ffn_norm)?;
        let q_norm_ptr = if w.q_norm.is_empty() {
            None
        } else {
            Some(resident_f32_ptr(gpu, w.q_norm)?)
        };
        let k_norm_ptr = if w.k_norm.is_empty() {
            None
        } else {
            Some(resident_f32_ptr(gpu, w.k_norm)?)
        };
        let wq_key = bytes_cache_key(w.wq);
        gpu.ensure_resident_quant(wq_key, w.wq)?;
        let wk_key = bytes_cache_key(w.wk);
        gpu.ensure_resident_quant(wk_key, w.wk)?;
        let wv_key = bytes_cache_key(w.wv);
        gpu.ensure_resident_quant(wv_key, w.wv)?;
        let wo_key = bytes_cache_key(w.wo);
        gpu.ensure_resident_quant(wo_key, w.wo)?;
        let gate_key = bytes_cache_key(w.gate);
        gpu.ensure_resident_quant(gate_key, w.gate)?;
        let up_key = bytes_cache_key(w.up);
        gpu.ensure_resident_quant(up_key, w.up)?;
        let down_key = bytes_cache_key(w.down);
        gpu.ensure_resident_quant(down_key, w.down)?;

        let wq_ptr = gpu.resident_quant[&wq_key].as_device_ptr();
        let wk_ptr = gpu.resident_quant[&wk_key].as_device_ptr();
        let wv_ptr = gpu.resident_quant[&wv_key].as_device_ptr();
        let wo_ptr = gpu.resident_quant[&wo_key].as_device_ptr();
        let gate_ptr = gpu.resident_quant[&gate_key].as_device_ptr();
        let up_ptr = gpu.resident_quant[&up_key].as_device_ptr();
        let down_ptr = gpu.resident_quant[&down_key].as_device_ptr();

        // Snapshot batched-activation pointers (buffers are not reallocated here).
        let (
            hidden_ptr,
            normed_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            attn_ptr,
            attn_proj_ptr,
            ffn_gate_ptr,
            ffn_up_ptr,
            ffn_down_ptr,
            bn_out_ptr,
            xq8k_ptr,
            xq8k_ffn_ptr,
        ) = {
            let ab = gpu
                .batched_activation
                .as_ref()
                .ok_or_else(|| "batched activation not initialised".to_string())?;
            if ab.batch != b
                || ab.hidden_size != geom.hidden_size
                || ab.q_len != geom.q_len
                || ab.kv_len != geom.kv_len
                || ab.intermediate_size != geom.intermediate_size
            {
                return Err("gpu_forward_batch_layer: batched activation geometry mismatch".into());
            }
            (
                ab.hidden.as_device_ptr(),
                ab.normed.as_device_ptr(),
                ab.q.as_device_ptr(),
                ab.k.as_device_ptr(),
                ab.v.as_device_ptr(),
                ab.attn.as_device_ptr(),
                ab.attn_proj.as_device_ptr(),
                ab.ffn_gate.as_device_ptr(),
                ab.ffn_up.as_device_ptr(),
                ab.ffn_down.as_device_ptr(),
                ab.bn_out.as_device_ptr(),
                ab.xq8k.as_device_ptr(),
                ab.xq8k_ffn.as_device_ptr(),
            )
        };

        let block256 = 256_u32;

        // ---- (1) attn RMSNorm per row ----
        {
            let block_dim = geom.hidden_size.next_power_of_two().min(512) as u32;
            let shmem = block_dim * 4;
            let fn_rms = gpu
                .module
                .get_function(RMS_NORM_KERNEL_NAME)
                .map_err(stringify)?;
            let stream = &gpu.stream;
            for bi in 0..b {
                let off = bi * geom.hidden_size;
                unsafe {
                    cust::launch!(fn_rms<<<1, block_dim, shmem, stream>>>(
                        hidden_ptr.wrapping_add(off),
                        attn_norm_ptr,
                        normed_ptr.wrapping_add(off),
                        h_u32,
                        w.eps
                    ))
                    .map_err(stringify)?;
                }
            }
        }

        // ---- (2) quantize normed -> xq8k (column-major over B) ----
        for bi in 0..b {
            let in_off = bi * geom.hidden_size;
            let out_off = bi * (bpr_hidden as usize) * BLOCK_Q8_K_BYTES;
            super::gemv_quantized::launch_quantize_f32_to_q8k_device_ptr(
                gpu,
                normed_ptr.wrapping_add(in_off),
                xq8k_ptr.wrapping_add(out_off),
                bpr_hidden,
            )?;
        }

        // ---- (3) QKV bN GEMVs -> bn_out[rows,B] -> transpose -> q/k/v[B,rows] ----
        super::gemv_quantized::launch_gemv_q4k_q8kin_bN_device(
            gpu, wq_ptr, q_u32, bpr_hidden, b_u32, xq8k_ptr, bn_out_ptr,
        )?;
        super::gemv_quantized::launch_transpose_rowB(gpu, bn_out_ptr, q_ptr, q_u32, b_u32)?;
        super::gemv_quantized::launch_gemv_q4k_q8kin_bN_device(
            gpu, wk_ptr, kv_u32, bpr_hidden, b_u32, xq8k_ptr, bn_out_ptr,
        )?;
        super::gemv_quantized::launch_transpose_rowB(gpu, bn_out_ptr, k_ptr, kv_u32, b_u32)?;
        super::gemv_quantized::launch_gemv_q4k_q8kin_bN_device(
            gpu, wv_ptr, kv_u32, bpr_hidden, b_u32, xq8k_ptr, bn_out_ptr,
        )?;
        super::gemv_quantized::launch_transpose_rowB(gpu, bn_out_ptr, v_ptr, kv_u32, b_u32)?;

        // ---- (3') optional Qwen3 QK-norm (per head, in-place on q/k) ----
        if q_norm_ptr.is_some() || k_norm_ptr.is_some() {
            let hd_u32 = geom.head_dim as u32;
            let shmem_norm = hd_u32 * 4;
            let fn_rms = gpu
                .module
                .get_function(RMS_NORM_KERNEL_NAME)
                .map_err(stringify)?;
            let stream = &gpu.stream;
            for bi in 0..b {
                if let Some(qn) = q_norm_ptr {
                    let off = bi * geom.q_len;
                    let qp = q_ptr.wrapping_add(off);
                    unsafe {
                        cust::launch!(fn_rms<<<geom.n_q_heads as u32, hd_u32, shmem_norm, stream>>>(
                            qp, qn, qp, hd_u32, w.eps
                        ))
                        .map_err(stringify)?;
                    }
                }
                if let Some(kn) = k_norm_ptr {
                    let off = bi * geom.kv_len;
                    let kp = k_ptr.wrapping_add(off);
                    unsafe {
                        cust::launch!(fn_rms<<<geom.n_kv_heads as u32, hd_u32, shmem_norm, stream>>>(
                            kp, kn, kp, hd_u32, w.eps
                        ))
                        .map_err(stringify)?;
                    }
                }
            }
        }

        // ---- (4) per-seq attention: rope -> kv_append -> flash_decode ----
        for bi in 0..b {
            let qp = q_ptr.wrapping_add(bi * geom.q_len);
            let kp = k_ptr.wrapping_add(bi * geom.kv_len);
            let vp = v_ptr.wrapping_add(bi * geom.kv_len);
            let ap = attn_ptr.wrapping_add(bi * geom.q_len);
            let p = pos[bi];
            let seq_len = kv_seq_len_pre[bi] + 1;
            let (eff_seq_len, base_row) = if w.layer_window != 0 && seq_len > w.layer_window {
                (w.layer_window, seq_len - w.layer_window)
            } else {
                (seq_len, 0)
            };
            launch_rope_f32(
                gpu,
                qp,
                kp,
                p as u32,
                geom.n_q_heads as u32,
                geom.n_kv_heads as u32,
                geom.head_dim as u32,
                geom.rope_dim as u32,
                w.theta,
            )?;
            let phys_pos = (p % gpu.kv_context) as u32;
            launch_kv_append_f16_batched(gpu, w.kv_layer_idx, bi, kp, vp, phys_pos, p)?;
            launch_flash_attn_decode_batched(
                gpu,
                w.kv_layer_idx,
                bi,
                qp,
                ap,
                eff_seq_len as u32,
                base_row as u32,
                geom.n_q_heads as u32,
                geom.n_kv_heads as u32,
                geom.head_dim as u32,
                scale,
            )?;
        }

        // ---- (5) Wo: quantize attn (B cols) -> bN -> transpose -> attn_proj ----
        // attn has q_len values per row; q_len must be a multiple of 256 for the
        // Q8_K quantizer (q_len == n_q_heads*head_dim).
        if !geom.q_len.is_multiple_of(256) {
            return Err(format!(
                "gpu_forward_batch_layer: q_len {} not a multiple of 256 (Wo Q8_K input)",
                geom.q_len
            ));
        }
        let bpr_q = (geom.q_len / 256) as u32;
        // Reuse xq8k region (sized for hidden, which has >= q_len/256 blocks).
        for bi in 0..b {
            let in_off = bi * geom.q_len;
            let out_off = bi * (bpr_q as usize) * BLOCK_Q8_K_BYTES;
            super::gemv_quantized::launch_quantize_f32_to_q8k_device_ptr(
                gpu,
                attn_ptr.wrapping_add(in_off),
                xq8k_ptr.wrapping_add(out_off),
                bpr_q,
            )?;
        }
        super::gemv_quantized::launch_gemv_q4k_q8kin_bN_device(
            gpu, wo_ptr, h_u32, bpr_q, b_u32, xq8k_ptr, bn_out_ptr,
        )?;
        super::gemv_quantized::launch_transpose_rowB(gpu, bn_out_ptr, attn_proj_ptr, h_u32, b_u32)?;

        // ---- (6) residual: hidden += attn_proj (flat over B*h) ----
        {
            let n = b_u32.saturating_mul(h_u32);
            let grid = n.div_ceil(block256);
            let fn_res = gpu
                .module
                .get_function(RESIDUAL_ADD_KERNEL_NAME)
                .map_err(stringify)?;
            let stream = &gpu.stream;
            unsafe {
                cust::launch!(fn_res<<<grid, block256, 0, stream>>>(
                    hidden_ptr, attn_proj_ptr, n
                ))
                .map_err(stringify)?;
            }
        }

        // ---- (7) ffn RMSNorm per row ----
        {
            let block_dim = geom.hidden_size.next_power_of_two().min(512) as u32;
            let shmem = block_dim * 4;
            let fn_rms = gpu
                .module
                .get_function(RMS_NORM_KERNEL_NAME)
                .map_err(stringify)?;
            let stream = &gpu.stream;
            for bi in 0..b {
                let off = bi * geom.hidden_size;
                unsafe {
                    cust::launch!(fn_rms<<<1, block_dim, shmem, stream>>>(
                        hidden_ptr.wrapping_add(off),
                        ffn_norm_ptr,
                        normed_ptr.wrapping_add(off),
                        h_u32,
                        w.eps
                    ))
                    .map_err(stringify)?;
                }
            }
        }

        // ---- (8) quantize normed -> xq8k ----
        for bi in 0..b {
            let in_off = bi * geom.hidden_size;
            let out_off = bi * (bpr_hidden as usize) * BLOCK_Q8_K_BYTES;
            super::gemv_quantized::launch_quantize_f32_to_q8k_device_ptr(
                gpu,
                normed_ptr.wrapping_add(in_off),
                xq8k_ptr.wrapping_add(out_off),
                bpr_hidden,
            )?;
        }

        // ---- (9) gate/up bN -> transpose -> ffn_gate/ffn_up ----
        super::gemv_quantized::launch_gemv_q4k_q8kin_bN_device(
            gpu, gate_ptr, inter_u32, bpr_hidden, b_u32, xq8k_ptr, bn_out_ptr,
        )?;
        super::gemv_quantized::launch_transpose_rowB(
            gpu,
            bn_out_ptr,
            ffn_gate_ptr,
            inter_u32,
            b_u32,
        )?;
        super::gemv_quantized::launch_gemv_q4k_q8kin_bN_device(
            gpu, up_ptr, inter_u32, bpr_hidden, b_u32, xq8k_ptr, bn_out_ptr,
        )?;
        super::gemv_quantized::launch_transpose_rowB(
            gpu, bn_out_ptr, ffn_up_ptr, inter_u32, b_u32,
        )?;

        // ---- (10) silu_mul (flat over B*inter) -> ffn_down ----
        {
            let n = b_u32.saturating_mul(inter_u32);
            let grid = n.div_ceil(block256);
            let fn_silu = gpu
                .module
                .get_function(SILU_MUL_KERNEL_NAME)
                .map_err(stringify)?;
            let stream = &gpu.stream;
            unsafe {
                cust::launch!(fn_silu<<<grid, block256, 0, stream>>>(
                    ffn_gate_ptr, ffn_up_ptr, ffn_down_ptr, n
                ))
                .map_err(stringify)?;
            }
        }

        // ---- (11) quantize ffn_down -> xq8k_ffn ; down bN -> transpose -> normed ----
        for bi in 0..b {
            let in_off = bi * geom.intermediate_size;
            let out_off = bi * (bpr_inter as usize) * BLOCK_Q8_K_BYTES;
            super::gemv_quantized::launch_quantize_f32_to_q8k_device_ptr(
                gpu,
                ffn_down_ptr.wrapping_add(in_off),
                xq8k_ffn_ptr.wrapping_add(out_off),
                bpr_inter,
            )?;
        }
        super::gemv_quantized::launch_gemv_q4k_q8kin_bN_device(
            gpu,
            down_ptr,
            h_u32,
            bpr_inter,
            b_u32,
            xq8k_ffn_ptr,
            bn_out_ptr,
        )?;
        super::gemv_quantized::launch_transpose_rowB(gpu, bn_out_ptr, normed_ptr, h_u32, b_u32)?;

        // ---- (12) residual: hidden += normed (flat over B*h) ----
        {
            let n = b_u32.saturating_mul(h_u32);
            let grid = n.div_ceil(block256);
            let fn_res = gpu
                .module
                .get_function(RESIDUAL_ADD_KERNEL_NAME)
                .map_err(stringify)?;
            let stream = &gpu.stream;
            unsafe {
                cust::launch!(fn_res<<<grid, block256, 0, stream>>>(
                    hidden_ptr, normed_ptr, n
                ))
                .map_err(stringify)?;
            }
        }

        Ok(())
    })
}

/// Upload B per-row embedded hidden states `[B * hidden_size]` into the batched
/// activation `hidden` buffer (row-major `[b][feature]`). The caller embeds the
/// B tokens on the host before the layer loop; the batched analogue of
/// [`gpu_upload_hidden`].
#[cfg(feature = "cuda")]
pub fn gpu_batched_upload_hidden(hidden_rows: &[f32]) -> Result<(), String> {
    with_gpu(|gpu| {
        let ab = gpu
            .batched_activation
            .as_mut()
            .ok_or_else(|| "batched activation not initialised".to_string())?;
        let expect = ab.batch * ab.hidden_size;
        if hidden_rows.len() != expect {
            return Err(format!(
                "gpu_batched_upload_hidden: len {} != batch*hidden_size {expect}",
                hidden_rows.len()
            ));
        }
        ab.hidden.copy_from(hidden_rows).map_err(stringify)
    })
}

/// Download the B per-row hidden states `[B * hidden_size]` from the batched
/// activation `hidden` buffer (syncs the stream first).
#[cfg(feature = "cuda")]
pub fn gpu_batched_download_hidden(out: &mut [f32]) -> Result<(), String> {
    with_gpu(|gpu| {
        gpu.stream.synchronize().map_err(stringify)?;
        let ab = gpu
            .batched_activation
            .as_ref()
            .ok_or_else(|| "batched activation not initialised".to_string())?;
        let expect = ab.batch * ab.hidden_size;
        if out.len() != expect {
            return Err(format!(
                "gpu_batched_download_hidden: out len {} != batch*hidden_size {expect}",
                out.len()
            ));
        }
        ab.hidden.copy_to(out).map_err(stringify)
    })
}

/// Batched final head: per-row final RMSNorm (into `normed`) then a bN lm_head
/// GEMV over the Q4_K output projection, transposed back to per-row logits and
/// downloaded as `[B * vocab_size]`. Mirrors [`gpu_final_head_device_resident`]
/// but for B rows in one weight pass. `hidden_size` must be a multiple of 256;
/// the activation buffer must have been sized with `bn_rows >= vocab_size`.
#[cfg(feature = "cuda")]
pub fn gpu_batched_final_head(
    norm_weight: &[f32],
    eps: f32,
    weight_bytes: &[u8],
    vocab_size: usize,
    hidden_size: usize,
    batch: usize,
    logits: &mut [f32],
) -> Result<(), String> {
    if !hidden_size.is_multiple_of(256) {
        return Err(format!(
            "gpu_batched_final_head: hidden_size {hidden_size} not a multiple of 256"
        ));
    }
    if logits.len() != batch * vocab_size {
        return Err(format!(
            "gpu_batched_final_head: logits len {} != batch*vocab {}",
            logits.len(),
            batch * vocab_size
        ));
    }
    let bpr_hidden = (hidden_size / 256) as u32;
    let vocab_u32 = vocab_size as u32;
    let h_u32 = hidden_size as u32;
    let b_u32 = batch as u32;

    with_gpu(|gpu| {
        let norm_ptr = resident_f32_ptr(gpu, norm_weight)?;
        let w_key = bytes_cache_key(weight_bytes);
        gpu.ensure_resident_quant(w_key, weight_bytes)?;
        let w_ptr = gpu.resident_quant[&w_key].as_device_ptr();

        let (hidden_ptr, normed_ptr, xq8k_ptr, bn_out_ptr) = {
            let ab = gpu
                .batched_activation
                .as_ref()
                .ok_or_else(|| "batched activation not initialised".to_string())?;
            if ab.batch != batch || ab.hidden_size != hidden_size {
                return Err("gpu_batched_final_head: activation geometry mismatch".into());
            }
            if ab.bn_rows < vocab_size {
                return Err(format!(
                    "gpu_batched_final_head: bn_out rows {} < vocab {vocab_size}",
                    ab.bn_rows
                ));
            }
            (
                ab.hidden.as_device_ptr(),
                ab.normed.as_device_ptr(),
                ab.xq8k.as_device_ptr(),
                ab.bn_out.as_device_ptr(),
            )
        };

        // ---- per-row final RMSNorm ----
        {
            let block_dim = hidden_size.next_power_of_two().min(512) as u32;
            let shmem = block_dim * 4;
            let fn_rms = gpu
                .module
                .get_function(RMS_NORM_KERNEL_NAME)
                .map_err(stringify)?;
            let stream = &gpu.stream;
            for bi in 0..batch {
                let off = bi * hidden_size;
                unsafe {
                    cust::launch!(fn_rms<<<1, block_dim, shmem, stream>>>(
                        hidden_ptr.wrapping_add(off),
                        norm_ptr,
                        normed_ptr.wrapping_add(off),
                        h_u32,
                        eps
                    ))
                    .map_err(stringify)?;
                }
            }
        }

        // ---- quantize normed -> xq8k ----
        for bi in 0..batch {
            let in_off = bi * hidden_size;
            let out_off = bi * (bpr_hidden as usize) * BLOCK_Q8_K_BYTES;
            super::gemv_quantized::launch_quantize_f32_to_q8k_device_ptr(
                gpu,
                normed_ptr.wrapping_add(in_off),
                xq8k_ptr.wrapping_add(out_off),
                bpr_hidden,
            )?;
        }

        // ---- bN lm_head -> bn_out[vocab,B] -> transpose -> logits[B,vocab] ----
        super::gemv_quantized::launch_gemv_q4k_q8kin_bN_device(
            gpu, w_ptr, vocab_u32, bpr_hidden, b_u32, xq8k_ptr, bn_out_ptr,
        )?;
        let d_logits = gpu.get_f32_buffer(batch * vocab_size)?;
        super::gemv_quantized::launch_transpose_rowB(
            gpu,
            bn_out_ptr,
            d_logits.as_device_ptr(),
            vocab_u32,
            b_u32,
        )?;
        gpu.stream.synchronize().map_err(stringify)?;
        d_logits.copy_to(logits).map_err(stringify)?;
        gpu.return_f32_buffer(d_logits);
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Numerical-correctness test: batched GPU forward == B single-seq forwards.
//
// Runs ONLY on Modal (real CUDA device + a real Q4_K GGUF). `#[ignore]` so it
// never runs in the default workspace test pass; the Modal action invokes it
// with `--include-ignored`. The reference is the per-seq CPU `forward_batch`
// (each sequence decoded alone), which the batched device path must match by
// argmax for every row.
// ---------------------------------------------------------------------------
#[cfg(all(test, feature = "cuda"))]
mod batched_parity_tests {
    use crate::inference::{InferenceConfig, InferenceModel, SeqKv};
    use crate::model::{Logits, Token};
    use crate::model_loader::{GgufModelLoader, ModelLoader};

    fn argmax(v: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best = i;
            }
        }
        best
    }

    /// GGUF used by the test. Override with `OX_BATCHED_TEST_GGUF`; the default
    /// matches the model the Modal action downloads (Qwen3-4B-Q4_K_M).
    fn test_gguf_path() -> String {
        std::env::var("OX_BATCHED_TEST_GGUF")
            .unwrap_or_else(|_| "/root/models/Qwen3-4B-Q4_K_M.gguf".to_string())
    }

    fn load_model() -> InferenceModel {
        let path = test_gguf_path();
        let loader = GgufModelLoader;
        let mapped = loader
            .load(std::path::Path::new(&path))
            .unwrap_or_else(|e| panic!("load gguf {path}: {e}"));
        let cfg = InferenceConfig::from_gguf(&mapped);
        InferenceModel::load_from_gguf(&mapped, cfg, true)
            .unwrap_or_else(|e| panic!("load_from_gguf {path}: {e}"))
    }

    /// Per-seq reference on the SAME device path (B=1): decode `prompt` then
    /// `decode_steps` greedy steps via `forward_batch_gpu`, returning the full
    /// final-step logits. Using the GPU path (not CPU) is deliberate — the batched
    /// kernel quantizes activations to Q8_K (dp4a), so the correct equivalence is
    /// "batched B rows == B independent single-seq GPU runs", both dp4a. (CPU f32
    /// would differ by quantization rounding, a separate known effect.)
    fn reference_run_gpu(
        model: &mut InferenceModel,
        prompt: &[Token],
        decode_steps: usize,
        cap: usize,
    ) -> Logits {
        let kvl = model.kv_layer_count();
        let kw = model.kv_row_len();
        let mut kv = vec![SeqKv::new(kvl, cap, kw)];
        let mut pos = 0usize;
        let mut last: Token = prompt[0];
        let mut out: Logits = Vec::new();
        for &t in prompt {
            let mut o = model
                .forward_batch_gpu(&[(t, pos)], &mut kv, true)
                .expect("ref forward_batch_gpu seed")
                .expect("ref batched GPU path must be active");
            last = argmax(&o[0]) as Token;
            out = o.remove(0);
            pos += 1;
        }
        for _ in 0..decode_steps {
            let mut o = model
                .forward_batch_gpu(&[(last, pos)], &mut kv, true)
                .expect("ref forward_batch_gpu decode")
                .expect("ref batched GPU active");
            last = argmax(&o[0]) as Token;
            out = o.remove(0);
            pos += 1;
        }
        out
    }

    /// The batched device path (B rows in one pass) must produce per-sequence
    /// logits BIT-FOR-BIT (within fp reduction-order tolerance) equal to running
    /// each sequence alone through the same device path (B=1). EQUAL-LENGTH prompts
    /// keep all rows lockstep (the single-pass batched path's requirement) so the
    /// comparison is at the same generation depth for every sequence.
    #[test]
    #[ignore = "requires a CUDA device + real Q4_K GGUF (Modal only)"]
    fn batched_gpu_matches_single_seq() {
        unsafe { std::env::set_var("OX_GPU_BATCHED", "1") };

        // B = 4 distinct prompts, all the SAME length (lockstep batched path).
        let prompts: Vec<Vec<Token>> = vec![
            vec![1, 2, 3, 4],
            vec![10, 20, 30, 40],
            vec![5, 6, 7, 8],
            vec![100, 200, 300, 400],
        ];
        let b = prompts.len();
        let prompt_len = prompts[0].len();
        assert!(prompts.iter().all(|p| p.len() == prompt_len));
        let decode_steps = 4usize;
        let cap = 64usize;

        // --- Reference: each sequence alone via the B=1 device path. ---
        let mut model = load_model();
        let refs: Vec<Logits> = prompts
            .iter()
            .map(|p| reference_run_gpu(&mut model, p, decode_steps, cap))
            .collect();

        // --- Batched device path: all B sequences together, lockstep. ---
        let mut model = load_model();
        let kvl = model.kv_layer_count();
        let kw = model.kv_row_len();
        let mut kv: Vec<SeqKv> = (0..b).map(|_| SeqKv::new(kvl, cap, kw)).collect();
        let mut pos = 0usize;
        let mut last = vec![0 as Token; b];
        let mut final_out: Vec<Logits> = vec![Vec::new(); b];

        for step in 0..(prompt_len + decode_steps) {
            let rows: Vec<(Token, usize)> = (0..b)
                .map(|s| {
                    let tok = if step < prompt_len {
                        prompts[s][step]
                    } else {
                        last[s]
                    };
                    (tok, pos)
                })
                .collect();
            let out = model
                .forward_batch_gpu(&rows, &mut kv, true)
                .expect("forward_batch_gpu seed/decode")
                .expect("batched GPU path must be active under OX_GPU_BATCHED=1");
            for s in 0..b {
                last[s] = argmax(&out[s]) as Token;
            }
            final_out = out;
            pos += 1;
        }

        // Per-sequence parity: exact argmax + tight per-logit tolerance.
        for s in 0..b {
            let ref_logits = &refs[s];
            let got = &final_out[s];
            assert_eq!(
                got.len(),
                ref_logits.len(),
                "seq {s}: logit length mismatch"
            );
            let ra = argmax(ref_logits);
            let ga = argmax(got);
            assert_eq!(
                ga, ra,
                "seq {s}: batched argmax {ga} != single-seq-GPU argmax {ra}"
            );
            let max_abs = ref_logits
                .iter()
                .zip(got.iter())
                .map(|(a, c)| (a - c).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_abs < 1e-2,
                "seq {s}: max|Δ| {max_abs} between batched and single-seq-GPU exceeds 1e-2"
            );
        }
        unsafe { std::env::remove_var("OX_GPU_BATCHED") };
    }
}
