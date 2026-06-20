use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_flash_attn_decode_legacy(
    gpu: &GpuState,
    kv_layer_idx: usize,
    d_q: cust::memory::DevicePointer<f32>,
    d_attn: cust::memory::DevicePointer<f32>,
    eff_seq_len: u32,
    base_row: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scale: f32,
) -> Result<(), String> {
    if kv_layer_idx >= gpu.kv_layers {
        return Err(format!(
            "launch_flash_attn_decode: kv_layer_idx {kv_layer_idx} out of range (kv_layers {})",
            gpu.kv_layers
        ));
    }
    let func = gpu
        .module
        .get_function(FLASH_ATTN_DECODE_KERNEL_NAME)
        .map_err(stringify)?;
    let kc_ptr = gpu.kv_k_cache[kv_layer_idx].as_device_ptr();
    let vc_ptr = gpu.kv_v_cache[kv_layer_idx].as_device_ptr();
    let shmem = (2 * head_dim + 1) * 4;
    let stream = &gpu.stream;
    // SAFETY: all device pointers refer to live buffers in this CUDA context;
    // launch dimensions match the kernel contract and the stream outlives the launch.
    unsafe {
        cust::launch!(func<<<n_q_heads, head_dim, shmem, stream>>>(
            d_q, kc_ptr, vc_ptr, d_attn,
            eff_seq_len, base_row, n_q_heads, n_kv_heads, head_dim, scale
        ))
        .map_err(stringify)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_flash_attn_decode_splitk(
    gpu: &mut GpuState,
    kv_layer_idx: usize,
    d_q: cust::memory::DevicePointer<f32>,
    d_attn: cust::memory::DevicePointer<f32>,
    eff_seq_len: u32,
    base_row: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scale: f32,
    num_splits: u32,
) -> Result<(), String> {
    if kv_layer_idx >= gpu.kv_layers {
        return Err(format!(
            "launch_flash_attn_decode_splitk: kv_layer_idx {kv_layer_idx} out of range (kv_layers {})",
            gpu.kv_layers
        ));
    }
    let kc_ptr = gpu.kv_k_cache[kv_layer_idx].as_device_ptr();
    let vc_ptr = gpu.kv_v_cache[kv_layer_idx].as_device_ptr();
    let n_slots = n_q_heads as usize * num_splits as usize;
    let d_pmax = gpu.get_f32_buffer(n_slots)?;
    let d_psum = gpu.get_f32_buffer(n_slots)?;
    let d_pacc = gpu.get_f32_buffer(n_slots * head_dim as usize)?;
    let pmax_ptr = d_pmax.as_device_ptr();
    let psum_ptr = d_psum.as_device_ptr();
    let pacc_ptr = d_pacc.as_device_ptr();
    // shmem = (2*head_dim + 1) * 4 bytes. head_dim is bounded by the OX_GPU_ATTN
    // caller (<= 256, the largest target head_dim), so the worst case is
    // (2*256+1)*4 = 2052 B — well under the 48 KB default dynamic-shared-memory
    // limit, so no cudaFuncAttributeMaxDynamicSharedMemorySize opt-in is needed.
    debug_assert!(
        head_dim <= 256,
        "flash_attn_decode_splitk: head_dim {head_dim} > 256 would risk shmem > 48KB default"
    );
    let shmem = (2 * head_dim + 1) * 4;

    let launch_result = (|| -> Result<(), String> {
        let split_func = gpu
            .module
            .get_function(FLASH_ATTN_DECODE_SPLITK_KERNEL_NAME)
            .map_err(stringify)?;
        let reduce_func = gpu
            .module
            .get_function(FLASH_ATTN_DECODE_REDUCE_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        // Grid dim ordering: cust's `launch!` maps the tuple `(a, b)` to
        // GridSize { x: a, y: b, z: 1 }, so `(n_q_heads, num_splits)` gives
        // gridDim.x = n_q_heads and gridDim.y = num_splits. The kernel reads
        // blockIdx.x = qh and blockIdx.y = split_idx, matching this ordering.
        // SAFETY: pointers reference live, correctly sized device buffers; grid and
        // shared-memory dimensions match the split kernel's parameter contract.
        unsafe {
            cust::launch!(split_func<<<(n_q_heads, num_splits), head_dim, shmem, stream>>>(
                d_q, kc_ptr, vc_ptr,
                pmax_ptr, psum_ptr, pacc_ptr,
                eff_seq_len, base_row, n_q_heads, n_kv_heads, head_dim, num_splits, scale
            ))
        }
        .and_then(|()| {
            // SAFETY: partial buffers remain live through this ordered stream launch,
            // and output has n_q_heads * head_dim writable f32 lanes.
            unsafe {
                cust::launch!(reduce_func<<<n_q_heads, head_dim, 0u32, stream>>>(
                    pmax_ptr, psum_ptr, pacc_ptr, d_attn,
                    n_q_heads, head_dim, num_splits
                ))
            }
        })
        .map_err(stringify)
    })();

    gpu.return_f32_buffer(d_pmax);
    gpu.return_f32_buffer(d_psum);
    gpu.return_f32_buffer(d_pacc);
    launch_result
}

fn selected_split_count(gpu: &GpuState, query_heads: u32, sequence_length: u32) -> u32 {
    if std::env::var_os("OX_FLASH_DECODE_FORCE_LEGACY").is_some() {
        return 1;
    }
    if let Some(forced) = std::env::var_os("OX_FLASH_DECODE_SPLITS")
        .and_then(|value| value.to_str().and_then(|text| text.parse::<u32>().ok()))
    {
        // Clamp to [1, sequence_length]: split_count > seq_len would launch
        // blocks for empty splits (the reduce kernel skips them via
        // `partial_sum[idx] > 0.0f`, so output is still correct, but those
        // blocks waste SM resources). seq_len==0 cannot reach here (decode
        // always has >=1 key), but max(1) keeps the launch geometry valid.
        return forced.clamp(1, sequence_length.max(1));
    }
    SplitKPlan::select(
        gpu.sm_count as usize,
        query_heads as usize,
        sequence_length as usize,
    )
    .map_or(1, |plan| plan.split_count as u32)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum DecodeSelection {
    Legacy,
    SplitK(u32),
}

thread_local! {
    static LAST_TRACE: std::cell::Cell<Option<DecodeSelection>> = const { std::cell::Cell::new(None) };
}

fn trace_selection(selection: DecodeSelection) {
    if std::env::var_os("OX_FLASH_DECODE_TRACE").is_none() {
        return;
    }
    LAST_TRACE.with(|last| {
        if last.get() == Some(selection) {
            return;
        }
        match selection {
            DecodeSelection::Legacy => eprintln!("flash_decode path=legacy splits=1"),
            DecodeSelection::SplitK(count) => {
                eprintln!("flash_decode path=split_k splits={count}")
            }
        }
        last.set(Some(selection));
    });
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_flash_attn_decode(
    gpu: &mut GpuState,
    kv_layer_idx: usize,
    d_q: cust::memory::DevicePointer<f32>,
    d_attn: cust::memory::DevicePointer<f32>,
    eff_seq_len: u32,
    base_row: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scale: f32,
) -> Result<(), String> {
    let split_count = selected_split_count(gpu, n_q_heads, eff_seq_len);
    if split_count > 1 {
        trace_selection(DecodeSelection::SplitK(split_count));
        launch_flash_attn_decode_splitk(
            gpu,
            kv_layer_idx,
            d_q,
            d_attn,
            eff_seq_len,
            base_row,
            n_q_heads,
            n_kv_heads,
            head_dim,
            scale,
            split_count,
        )
    } else {
        trace_selection(DecodeSelection::Legacy);
        launch_flash_attn_decode_legacy(
            gpu,
            kv_layer_idx,
            d_q,
            d_attn,
            eff_seq_len,
            base_row,
            n_q_heads,
            n_kv_heads,
            head_dim,
            scale,
        )
    }
}
