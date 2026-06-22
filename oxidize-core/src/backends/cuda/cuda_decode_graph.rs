//! CUDA graph capture for single-token decode (`OX_GPU_CUDA_GRAPH`).
//!
//! Collapses the per-layer kernel launch sequence (RMSNorm → QKV MMQ → RoPE →
//! KV append → flash decode → Wo → FFN) into one `cuGraphLaunch` per layer.
//! Position-dependent scalars flow through a 3-word device buffer (`d_state`) so
//! the captured topology stays valid as the sequence grows.

use super::*;
use std::mem::MaybeUninit;

#[cfg(feature = "cuda")]
fn cu_ok(status: cust::sys::CUresult) -> Result<(), String> {
    if status == cust::sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("CUDA driver error: {status:?}"))
    }
}

/// CUDA graph decode OFF by default until capture includes the full MMQ stack
/// without regressing vs eager decode. Set `OX_GPU_CUDA_GRAPH=1` to enable.
#[cfg(feature = "cuda")]
pub fn ox_gpu_cuda_graph_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("OX_GPU_CUDA_GRAPH")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    })
}

#[cfg(feature = "cuda")]
pub(super) fn decode_graph_use_gph(pos: usize, context: usize) -> bool {
    gpu_decode_graph_eligible(pos, context)
}

#[cfg(feature = "cuda")]
pub fn gpu_decode_graph_eligible(pos: usize, context: usize) -> bool {
    ox_gpu_cuda_graph_enabled() && pos > 0 && pos + 1 <= context
}

#[cfg(feature = "cuda")]
fn ensure_decode_d_state(gpu: &mut GpuState) -> Result<(), String> {
    if gpu.decode_d_state.is_none() {
        gpu.decode_d_state =
            Some(cust::memory::DeviceBuffer::<u32>::zeroed(3).map_err(stringify)?);
    }
    Ok(())
}

/// Patch live per-token scalars before each graph replay (one tiny H2D).
#[cfg(feature = "cuda")]
pub fn gpu_decode_graph_set_token(
    pos: u32,
    context: u32,
    token_id: u32,
) -> Result<(), String> {
    with_gpu(|gpu| {
        ensure_decode_d_state(gpu)?;
        let host = [pos, context, token_id];
        gpu.decode_d_state
            .as_mut()
            .expect("decode d_state")
            .copy_from(&host)
            .map_err(stringify)
    })
}

#[cfg(feature = "cuda")]
pub fn gpu_decode_graph_reset(num_layers: usize) -> Result<(), String> {
    with_gpu(|gpu| {
        gpu.decode_layer_graphs.clear();
        gpu.decode_graph_layers = num_layers;
        gpu.decode_layer_graphs.resize_with(num_layers, || None);
        Ok(())
    })
}

#[cfg(feature = "cuda")]
pub(super) fn gpu_decode_graph_layer_ready(gpu: &GpuState, layer_idx: usize) -> bool {
    gpu.decode_layer_graphs
        .get(layer_idx)
        .is_some_and(|g| g.is_some())
}

#[cfg(feature = "cuda")]
pub(super) fn gpu_decode_graph_try_launch(
    gpu: &GpuState,
    layer_idx: usize,
) -> Result<bool, String> {
    let Some(Some(exec)) = gpu.decode_layer_graphs.get(layer_idx) else {
        return Ok(false);
    };
    let stream = gpu.stream.as_inner();
    unsafe {
        cu_ok(cust::sys::cuGraphLaunch(exec.0, stream))?;
    }
    Ok(true)
}

#[cfg(feature = "cuda")]
pub(super) fn gpu_decode_graph_begin_capture(gpu: &mut GpuState) -> Result<(), String> {
    ensure_decode_d_state(gpu)?;
    let stream = gpu.stream.as_inner();
    unsafe {
        cu_ok(cust::sys::cuStreamBeginCapture_v2(
            stream,
            cust::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        ))?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(super) fn gpu_decode_graph_end_capture(
    gpu: &mut GpuState,
    layer_idx: usize,
) -> Result<(), String> {
    if layer_idx >= gpu.decode_layer_graphs.len() {
        return Err(format!(
            "gpu_decode_graph_end_capture: layer {layer_idx} out of range ({})",
            gpu.decode_layer_graphs.len()
        ));
    }
    let stream = gpu.stream.as_inner();
    unsafe {
        let mut graph = MaybeUninit::<cust::sys::CUgraph>::uninit();
        cu_ok(cust::sys::cuStreamEndCapture(stream, graph.as_mut_ptr()))?;
        let graph = graph.assume_init();
        let mut exec = MaybeUninit::<cust::sys::CUgraphExec>::uninit();
        cu_ok(cust::sys::cuGraphInstantiateWithFlags(
            exec.as_mut_ptr(),
            graph,
            0,
        ))?;
        cu_ok(cust::sys::cuGraphDestroy(graph))?;
        gpu.decode_layer_graphs[layer_idx] = Some(CudaGraphExec(exec.assume_init()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GPH kernel launchers — read pos/context/token_id from d_state[0..3].
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub(super) fn launch_rope_f32_gph(
    gpu: &GpuState,
    d_state: cust::memory::DevicePointer<u32>,
    d_q: cust::memory::DevicePointer<f32>,
    d_k: cust::memory::DevicePointer<f32>,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
) -> Result<(), String> {
    let func = gpu
        .module
        .get_function(ROPE_F32_GPH_KERNEL_NAME)
        .map_err(stringify)?;
    let grid = n_q_heads + n_kv_heads;
    let stream = &gpu.stream;
    unsafe {
        cust::launch!(func<<<grid, head_dim, 0, stream>>>(
            d_q, d_k, d_state, n_q_heads, n_kv_heads, head_dim, rope_dim, theta
        ))
        .map_err(stringify)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(super) fn launch_kv_append_f16_gph(
    gpu: &GpuState,
    kv_layer_idx: usize,
    d_state: cust::memory::DevicePointer<u32>,
    d_k: cust::memory::DevicePointer<f32>,
    d_v: cust::memory::DevicePointer<f32>,
) -> Result<(), String> {
    if kv_layer_idx >= gpu.kv_layers {
        return Err(format!(
            "launch_kv_append_f16_gph: layer {kv_layer_idx} out of range"
        ));
    }
    let kv_len = gpu.kv_len as u32;
    let func = gpu
        .module
        .get_function(KV_APPEND_F16_GPH_KERNEL_NAME)
        .map_err(stringify)?;
    let kc_ptr = gpu.kv_k_cache[kv_layer_idx].as_device_ptr();
    let vc_ptr = gpu.kv_v_cache[kv_layer_idx].as_device_ptr();
    let block = 256_u32;
    let grid = kv_len.div_ceil(block);
    let stream = &gpu.stream;
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(
            d_k, d_v, kc_ptr, vc_ptr, d_state, kv_len
        ))
        .map_err(stringify)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_flash_attn_decode_gph(
    gpu: &GpuState,
    kv_layer_idx: usize,
    d_state: cust::memory::DevicePointer<u32>,
    d_q: cust::memory::DevicePointer<f32>,
    d_attn: cust::memory::DevicePointer<f32>,
    layer_window: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scale: f32,
) -> Result<(), String> {
    if kv_layer_idx >= gpu.kv_layers {
        return Err(format!(
            "launch_flash_attn_decode_gph: layer {kv_layer_idx} out of range"
        ));
    }
    let func = gpu
        .module
        .get_function(FLASH_ATTN_DECODE_GPH_KERNEL_NAME)
        .map_err(stringify)?;
    let kc_ptr = gpu.kv_k_cache[kv_layer_idx].as_device_ptr();
    let vc_ptr = gpu.kv_v_cache[kv_layer_idx].as_device_ptr();
    let shmem = (2 * head_dim + 1) * 4;
    let stream = &gpu.stream;
    unsafe {
        cust::launch!(func<<<n_q_heads, head_dim, shmem, stream>>>(
            d_q,
            kc_ptr,
            vc_ptr,
            d_attn,
            d_state,
            layer_window,
            n_q_heads,
            n_kv_heads,
            head_dim,
            scale
        ))
        .map_err(stringify)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(super) fn decode_d_state_ptr(
    gpu: &mut GpuState,
) -> Result<cust::memory::DevicePointer<u32>, String> {
    ensure_decode_d_state(gpu)?;
    Ok(gpu
        .decode_d_state
        .as_ref()
        .expect("decode d_state")
        .as_device_ptr())
}

#[cfg(feature = "cuda")]
pub(super) fn update_kv_seq_len_after_gph_append(
    gpu: &mut GpuState,
    kv_layer_idx: usize,
    pos: usize,
) {
    if let Some(s) = gpu.kv_seq_len.get_mut(kv_layer_idx) {
        *s = (pos + 1).min(gpu.kv_context);
    }
}

#[cfg(feature = "cuda")]
pub fn gpu_decode_layer_graph_begin(
    layer_idx: usize,
    pos: usize,
    context: usize,
) -> Result<bool, String> {
    if !gpu_decode_graph_eligible(pos, context) {
        return Ok(false);
    }
    with_gpu(|gpu| {
        if gpu_decode_graph_layer_ready(gpu, layer_idx) {
            return gpu_decode_graph_try_launch(gpu, layer_idx);
        }
        gpu_decode_graph_begin_capture(gpu)?;
        Ok(false)
    })
}

#[cfg(feature = "cuda")]
pub fn gpu_decode_layer_graph_end(
    layer_idx: usize,
    pos: usize,
    context: usize,
) -> Result<(), String> {
    if !gpu_decode_graph_eligible(pos, context) {
        return Ok(());
    }
    with_gpu(|gpu| {
        if gpu_decode_graph_layer_ready(gpu, layer_idx) {
            return Ok(());
        }
        gpu_decode_graph_end_capture(gpu, layer_idx)
    })
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn cuda_graph_disabled_by_default() {
        let prev = std::env::var_os("OX_GPU_CUDA_GRAPH");
        std::env::remove_var("OX_GPU_CUDA_GRAPH");
        assert!(!ox_gpu_cuda_graph_enabled());
        match prev {
            Some(v) => std::env::set_var("OX_GPU_CUDA_GRAPH", v),
            None => std::env::remove_var("OX_GPU_CUDA_GRAPH"),
        }
    }

    #[test]
    fn graph_eligible_requires_pos_within_context() {
        assert!(!gpu_decode_graph_eligible(0, 4096));
        assert!(gpu_decode_graph_eligible(1, 4096));
        assert!(!gpu_decode_graph_eligible(4096, 4096));
    }
}
