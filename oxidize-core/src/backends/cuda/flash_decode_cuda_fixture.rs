use super::*;
use cust::event::{Event, EventFlags};
use cust::memory::CopyDestination;

pub(super) fn run_flash_decode_parity_case(
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    base_row: usize,
    extreme_scores: bool,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let context = base_row + seq_len;
    let kv_width = kv_heads * head_dim;
    let q_len = query_heads * head_dim;
    let mut q = vec![0.0_f32; q_len];
    for (index, value) in q.iter_mut().enumerate() {
        let magnitude = if extreme_scores { 32.0 } else { 0.125 };
        *value = ((index % 17) as f32 - 8.0) * magnitude;
    }
    let mut keys = vec![0_u16; context * kv_width];
    let mut values = vec![0_u16; context * kv_width];
    for row in 0..context {
        for lane in 0..kv_width {
            let index = row * kv_width + lane;
            let key = (((row * 13 + lane * 7) % 29) as f32 - 14.0) / 16.0;
            let value = (((row * 5 + lane * 11) % 31) as f32 - 15.0) / 32.0;
            keys[index] = crate::kv_cache::f32_to_f16_bits(key);
            values[index] = crate::kv_cache::f32_to_f16_bits(value);
        }
    }

    let mut gpu = gpu_init()?;
    gpu.kv_k_cache = vec![cust::memory::DeviceBuffer::from_slice(&keys).map_err(stringify)?];
    gpu.kv_v_cache = vec![cust::memory::DeviceBuffer::from_slice(&values).map_err(stringify)?];
    gpu.kv_layers = 1;
    gpu.kv_len = kv_width;
    gpu.kv_context = context;
    gpu.kv_seq_len = vec![context];
    let d_q = cust::memory::DeviceBuffer::from_slice(&q).map_err(stringify)?;
    let d_split = cust::memory::DeviceBuffer::<f32>::zeroed(q_len).map_err(stringify)?;
    let d_legacy = cust::memory::DeviceBuffer::<f32>::zeroed(q_len).map_err(stringify)?;
    let scale = 1.0 / (head_dim as f32).sqrt();

    launch_flash_attn_decode_splitk(
        &mut gpu,
        0,
        d_q.as_device_ptr(),
        d_split.as_device_ptr(),
        seq_len as u32,
        base_row as u32,
        query_heads as u32,
        kv_heads as u32,
        head_dim as u32,
        scale,
        7,
    )?;
    launch_flash_attn_decode_legacy(
        &gpu,
        0,
        d_q.as_device_ptr(),
        d_legacy.as_device_ptr(),
        seq_len as u32,
        base_row as u32,
        query_heads as u32,
        kv_heads as u32,
        head_dim as u32,
        scale,
    )?;
    gpu.stream.synchronize().map_err(stringify)?;
    let mut split = vec![0.0_f32; q_len];
    let mut legacy = vec![0.0_f32; q_len];
    d_split.copy_to(&mut split).map_err(stringify)?;
    d_legacy.copy_to(&mut legacy).map_err(stringify)?;
    Ok((split, legacy))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn benchmark_flash_decode_case(
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    warmup_count: usize,
    sample_count: usize,
) -> Result<(usize, Vec<f64>, Vec<f64>), String> {
    if warmup_count != 3 || sample_count != 10 {
        return Err("flash-decode benchmark requires 3 warmups and 10 samples".to_owned());
    }

    let kv_width = kv_heads * head_dim;
    let q = (0..query_heads * head_dim)
        .map(|index| ((index % 17) as f32 - 8.0) / 64.0)
        .collect::<Vec<_>>();
    let keys = (0..seq_len * kv_width)
        .map(|index| crate::kv_cache::f32_to_f16_bits(((index % 29) as f32 - 14.0) / 16.0))
        .collect::<Vec<_>>();
    let values = (0..seq_len * kv_width)
        .map(|index| crate::kv_cache::f32_to_f16_bits(((index % 31) as f32 - 15.0) / 32.0))
        .collect::<Vec<_>>();

    let mut gpu = gpu_init()?;
    let plan = SplitKPlan::select(gpu.sm_count as usize, query_heads, seq_len)
        .ok_or_else(|| "GPU shape does not select split-K flash decode".to_owned())?;
    gpu.kv_k_cache = vec![cust::memory::DeviceBuffer::from_slice(&keys).map_err(stringify)?];
    gpu.kv_v_cache = vec![cust::memory::DeviceBuffer::from_slice(&values).map_err(stringify)?];
    gpu.kv_layers = 1;
    gpu.kv_len = kv_width;
    gpu.kv_context = seq_len;
    gpu.kv_seq_len = vec![seq_len];
    let d_q = cust::memory::DeviceBuffer::from_slice(&q).map_err(stringify)?;
    let d_output =
        cust::memory::DeviceBuffer::<f32>::zeroed(query_heads * head_dim).map_err(stringify)?;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for _ in 0..warmup_count {
        launch_flash_attn_decode_legacy(
            &gpu,
            0,
            d_q.as_device_ptr(),
            d_output.as_device_ptr(),
            seq_len as u32,
            0,
            query_heads as u32,
            kv_heads as u32,
            head_dim as u32,
            scale,
        )?;
        launch_flash_attn_decode_splitk(
            &mut gpu,
            0,
            d_q.as_device_ptr(),
            d_output.as_device_ptr(),
            seq_len as u32,
            0,
            query_heads as u32,
            kv_heads as u32,
            head_dim as u32,
            scale,
            plan.split_count as u32,
        )?;
    }
    gpu.stream.synchronize().map_err(stringify)?;

    let start = Event::new(EventFlags::DEFAULT).map_err(stringify)?;
    let stop = Event::new(EventFlags::DEFAULT).map_err(stringify)?;
    let mut legacy_ms = Vec::with_capacity(sample_count);
    let mut split_ms = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        start.record(&gpu.stream).map_err(stringify)?;
        launch_flash_attn_decode_legacy(
            &gpu,
            0,
            d_q.as_device_ptr(),
            d_output.as_device_ptr(),
            seq_len as u32,
            0,
            query_heads as u32,
            kv_heads as u32,
            head_dim as u32,
            scale,
        )?;
        stop.record(&gpu.stream).map_err(stringify)?;
        stop.synchronize().map_err(stringify)?;
        legacy_ms.push(f64::from(stop.elapsed_time_f32(&start).map_err(stringify)?));

        start.record(&gpu.stream).map_err(stringify)?;
        launch_flash_attn_decode_splitk(
            &mut gpu,
            0,
            d_q.as_device_ptr(),
            d_output.as_device_ptr(),
            seq_len as u32,
            0,
            query_heads as u32,
            kv_heads as u32,
            head_dim as u32,
            scale,
            plan.split_count as u32,
        )?;
        stop.record(&gpu.stream).map_err(stringify)?;
        stop.synchronize().map_err(stringify)?;
        split_ms.push(f64::from(stop.elapsed_time_f32(&start).map_err(stringify)?));
    }

    Ok((plan.split_count, legacy_ms, split_ms))
}
