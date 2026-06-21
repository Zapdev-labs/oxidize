use super::*;
use cust::event::{Event, EventFlags};
use cust::memory::CopyDestination;

pub(super) struct FfnCudaEventBenchmark {
    pub(super) eager_output: Vec<f32>,
    pub(super) fused_output: Vec<f32>,
    pub(super) eager_ms: Vec<f64>,
    pub(super) fused_ms: Vec<f64>,
}

pub(super) fn run_q4k_gate_up_silu_parity_case(
    rows: usize,
    blocks_per_row: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let cols = blocks_per_row
        .checked_mul(QK_K)
        .ok_or_else(|| "Q4_K fixture column count overflow".to_owned())?;
    let weight_bytes = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(BLOCK_Q4_K_SIZE))
        .ok_or_else(|| "Q4_K fixture weight size overflow".to_owned())?;
    let gate_weights = q4k_fixture_weights(weight_bytes, 0x3141_5926);
    let up_weights = q4k_fixture_weights(weight_bytes, 0x2718_2818);
    let input = (0..cols)
        .map(|index| ((index as f32 * 0.013).cos() * 0.75) + ((index % 7) as f32 - 3.0) * 0.03)
        .collect::<Vec<_>>();

    let gpu = gpu_init()?;
    let d_gate_weights =
        cust::memory::DeviceBuffer::from_slice(&gate_weights).map_err(stringify)?;
    let d_up_weights = cust::memory::DeviceBuffer::from_slice(&up_weights).map_err(stringify)?;
    let d_input = cust::memory::DeviceBuffer::from_slice(&input).map_err(stringify)?;
    let d_gate = cust::memory::DeviceBuffer::<f32>::zeroed(rows).map_err(stringify)?;
    let d_up = cust::memory::DeviceBuffer::<f32>::zeroed(rows).map_err(stringify)?;
    let d_eager = cust::memory::DeviceBuffer::<f32>::zeroed(rows).map_err(stringify)?;
    let d_fused = cust::memory::DeviceBuffer::<f32>::zeroed(rows).map_err(stringify)?;

    let gemv = gpu
        .module
        .get_function(GEMV_Q4K_F32IN_KERNEL_NAME)
        .map_err(stringify)?;
    let silu = gpu
        .module
        .get_function(SILU_MUL_KERNEL_NAME)
        .map_err(stringify)?;
    let fused = gpu
        .module
        .get_function(GEMV_Q4K_F32IN_GATE_UP_SILU_KERNEL_NAME)
        .map_err(stringify)?;
    let rows_u32 = u32::try_from(rows).map_err(|_| "fixture rows exceed u32".to_owned())?;
    let blocks_u32 =
        u32::try_from(blocks_per_row).map_err(|_| "fixture blocks exceed u32".to_owned())?;
    let block_size = 256_u32;
    let grid_size = rows_u32.saturating_mul(32).div_ceil(block_size);
    let element_grid = rows_u32.div_ceil(block_size);
    let stream = &gpu.stream;

    // SAFETY: Category 8 (FFI boundary). Every CUDA pointer refers to a live device
    // allocation sized from the checked rows/blocks geometry, and launch dimensions
    // match the kernel ABI declared in the checked-in CUDA source.
    unsafe {
        cust::launch!(gemv<<<grid_size, block_size, 0, stream>>>(
            d_gate_weights.as_device_ptr(),
            d_input.as_device_ptr(),
            d_gate.as_device_ptr(),
            rows_u32,
            blocks_u32
        ))
        .map_err(stringify)?;
        cust::launch!(gemv<<<grid_size, block_size, 0, stream>>>(
            d_up_weights.as_device_ptr(),
            d_input.as_device_ptr(),
            d_up.as_device_ptr(),
            rows_u32,
            blocks_u32
        ))
        .map_err(stringify)?;
        cust::launch!(silu<<<element_grid, block_size, 0, stream>>>(
            d_gate.as_device_ptr(),
            d_up.as_device_ptr(),
            d_eager.as_device_ptr(),
            rows_u32
        ))
        .map_err(stringify)?;
        cust::launch!(fused<<<grid_size, block_size, 0, stream>>>(
            d_gate_weights.as_device_ptr(),
            d_up_weights.as_device_ptr(),
            d_input.as_device_ptr(),
            d_fused.as_device_ptr(),
            rows_u32,
            blocks_u32
        ))
        .map_err(stringify)?;
    }
    gpu.stream.synchronize().map_err(stringify)?;

    let mut eager = vec![0.0_f32; rows];
    let mut fused_output = vec![0.0_f32; rows];
    d_eager.copy_to(&mut eager).map_err(stringify)?;
    d_fused.copy_to(&mut fused_output).map_err(stringify)?;
    Ok((eager, fused_output))
}

pub(super) fn run_q4k_gate_up_silu_cuda_event_benchmark(
    rows: usize,
    blocks_per_row: usize,
) -> Result<FfnCudaEventBenchmark, String> {
    const WARMUP_COUNT: usize = 3;
    const SAMPLE_COUNT: usize = 10;

    let cols = blocks_per_row
        .checked_mul(QK_K)
        .ok_or_else(|| "Q4_K fixture column count overflow".to_owned())?;
    let weight_bytes = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(BLOCK_Q4_K_SIZE))
        .ok_or_else(|| "Q4_K fixture weight size overflow".to_owned())?;
    let gate_weights = q4k_fixture_weights(weight_bytes, 0x3141_5926);
    let up_weights = q4k_fixture_weights(weight_bytes, 0x2718_2818);
    let input = (0..cols)
        .map(|index| ((index as f32 * 0.013).cos() * 0.75) + ((index % 7) as f32 - 3.0) * 0.03)
        .collect::<Vec<_>>();

    let gpu = gpu_init()?;
    let d_gate_weights =
        cust::memory::DeviceBuffer::from_slice(&gate_weights).map_err(stringify)?;
    let d_up_weights = cust::memory::DeviceBuffer::from_slice(&up_weights).map_err(stringify)?;
    let d_input = cust::memory::DeviceBuffer::from_slice(&input).map_err(stringify)?;
    let d_gate = cust::memory::DeviceBuffer::<f32>::zeroed(rows).map_err(stringify)?;
    let d_up = cust::memory::DeviceBuffer::<f32>::zeroed(rows).map_err(stringify)?;
    let d_eager = cust::memory::DeviceBuffer::<f32>::zeroed(rows).map_err(stringify)?;
    let d_fused = cust::memory::DeviceBuffer::<f32>::zeroed(rows).map_err(stringify)?;
    let gemv = gpu
        .module
        .get_function(GEMV_Q4K_F32IN_KERNEL_NAME)
        .map_err(stringify)?;
    let silu = gpu
        .module
        .get_function(SILU_MUL_KERNEL_NAME)
        .map_err(stringify)?;
    let fused = gpu
        .module
        .get_function(GEMV_Q4K_F32IN_GATE_UP_SILU_KERNEL_NAME)
        .map_err(stringify)?;
    let rows_u32 = u32::try_from(rows).map_err(|_| "fixture rows exceed u32".to_owned())?;
    let blocks_u32 =
        u32::try_from(blocks_per_row).map_err(|_| "fixture blocks exceed u32".to_owned())?;
    let block_size = 256_u32;
    let grid_size = rows_u32.saturating_mul(32).div_ceil(block_size);
    let element_grid = rows_u32.div_ceil(block_size);
    let stream = &gpu.stream;

    let launch_eager = || -> Result<(), String> {
        // SAFETY: Category 8 (FFI boundary). Buffers and launch geometry match the CUDA ABI.
        unsafe {
            cust::launch!(gemv<<<grid_size, block_size, 0, stream>>>(
                d_gate_weights.as_device_ptr(), d_input.as_device_ptr(),
                d_gate.as_device_ptr(), rows_u32, blocks_u32
            ))
            .map_err(stringify)?;
            cust::launch!(gemv<<<grid_size, block_size, 0, stream>>>(
                d_up_weights.as_device_ptr(), d_input.as_device_ptr(),
                d_up.as_device_ptr(), rows_u32, blocks_u32
            ))
            .map_err(stringify)?;
            cust::launch!(silu<<<element_grid, block_size, 0, stream>>>(
                d_gate.as_device_ptr(), d_up.as_device_ptr(),
                d_eager.as_device_ptr(), rows_u32
            ))
            .map_err(stringify)?;
        }
        Ok(())
    };
    let launch_fused = || -> Result<(), String> {
        // SAFETY: Category 8 (FFI boundary). Buffers and launch geometry match the CUDA ABI.
        unsafe {
            cust::launch!(fused<<<grid_size, block_size, 0, stream>>>(
                d_gate_weights.as_device_ptr(), d_up_weights.as_device_ptr(),
                d_input.as_device_ptr(), d_fused.as_device_ptr(), rows_u32, blocks_u32
            ))
            .map_err(stringify)?;
        }
        Ok(())
    };

    for _ in 0..WARMUP_COUNT {
        launch_eager()?;
        launch_fused()?;
    }
    gpu.stream.synchronize().map_err(stringify)?;

    let start = Event::new(EventFlags::DEFAULT).map_err(stringify)?;
    let stop = Event::new(EventFlags::DEFAULT).map_err(stringify)?;
    let mut eager_ms = Vec::with_capacity(SAMPLE_COUNT);
    let mut fused_ms = Vec::with_capacity(SAMPLE_COUNT);
    for _ in 0..SAMPLE_COUNT {
        start.record(&gpu.stream).map_err(stringify)?;
        launch_eager()?;
        stop.record(&gpu.stream).map_err(stringify)?;
        stop.synchronize().map_err(stringify)?;
        eager_ms.push(f64::from(stop.elapsed_time_f32(&start).map_err(stringify)?));

        start.record(&gpu.stream).map_err(stringify)?;
        launch_fused()?;
        stop.record(&gpu.stream).map_err(stringify)?;
        stop.synchronize().map_err(stringify)?;
        fused_ms.push(f64::from(stop.elapsed_time_f32(&start).map_err(stringify)?));
    }

    launch_eager()?;
    launch_fused()?;
    gpu.stream.synchronize().map_err(stringify)?;
    let mut eager_output = vec![0.0_f32; rows];
    let mut fused_output = vec![0.0_f32; rows];
    d_eager.copy_to(&mut eager_output).map_err(stringify)?;
    d_fused.copy_to(&mut fused_output).map_err(stringify)?;
    Ok(FfnCudaEventBenchmark {
        eager_output,
        fused_output,
        eager_ms,
        fused_ms,
    })
}

fn q4k_fixture_weights(byte_len: usize, mut state: u32) -> Vec<u8> {
    let mut weights = vec![0_u8; byte_len];
    for block in weights.chunks_exact_mut(BLOCK_Q4_K_SIZE) {
        block[0] = 0x00;
        block[1] = 0x34;
        block[2] = 0x00;
        block[3] = 0x28;
        for (index, scale) in block[4..16].iter_mut().enumerate() {
            *scale = ((index * 5 + 3) & 0x3f) as u8;
        }
        for quant in &mut block[16..] {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *quant = (state >> 16) as u8;
        }
    }
    weights
}
