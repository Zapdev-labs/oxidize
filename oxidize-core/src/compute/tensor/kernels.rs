use crate::gguf::GgufQuantizationType;
use crate::quantization::{
    BLOCK_NVFP4_SIZE, BLOCK_Q2_K_SIZE, BLOCK_Q4_K_SIZE, BLOCK_Q6_K_SIZE, BLOCK_Q8_0_SIZE, QK8_0,
    QK_K, QK_NVFP4, QK_NVFP4_SUB,
};
use rayon::prelude::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::errors::{
    AttentionError, GemmError, GemvError, LayerNormError, LinearActivationError, RmsNormError,
    RopeError, SoftmaxError, SwiGluError,
};
use super::types::{ActivationFn, DType};

const E2M1_DOUBLED_VALUES: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];
const FLASH_ATTENTION_BLOCK_TOKENS: usize = 64;
const PARALLEL_GEMV_MIN_OPS: usize = 1 << 20;

/// Rows per spin-pool dispatch chunk. Small chunks cost nothing under static
/// partitioning (no claim contention) and cut straggler imbalance on
/// mid-sized regions; 8 still holds two 4-row kernel quads.
const GEMV_CHUNK_ROWS: usize = 32;

const TRANSPOSED_GEMV_COL_CHUNK: usize = QK_K;


pub fn gemv_f32(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let expected_matrix_len = rows.saturating_mul(cols);
    if matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    if crate::gpu_dispatch::active_gpu().is_some() {
        return crate::gpu_dispatch::gemv_f32(matrix, rows, cols, vector, output)
            .map_err(GemvError::Cuda);
    }

    #[cfg(feature = "webgpu")]
    if crate::webgpu::should_use_webgpu_gemv(rows, cols) {
        crate::webgpu::validate_gemv_dims(matrix, rows, cols, vector, output)
            .map_err(|err| GemvError::WebGpu(format!("WebGPU GEMV validation failed: {err:?}")))?;
        gemv_f32_cpu(matrix, cols, vector, output);
        return Ok(());
    }

    #[cfg(feature = "metal")]
    if crate::metal::should_use_mps_gemv(rows, cols) {
        crate::metal::validate_gemv_dims(matrix, rows, cols, vector, output)
            .map_err(|err| GemvError::Metal(format!("MPS GEMV validation failed: {err:?}")))?;
        gemv_f32_cpu(matrix, cols, vector, output);
        return Ok(());
    }

    gemv_f32_cpu(matrix, cols, vector, output);
    Ok(())
}

/// Batched GEMM against a quantized weight matrix.
///
/// Equivalent to calling [`gemv_quantized_f32`] once per batch element, but the
/// weight matrix is scanned a single time and each decoded block is reused
/// across the batch. This collapses the dominant memory-bandwidth cost of
/// prompt-processing into a single weight load per layer instead of `batch`
/// loads, which is the primary win over the per-token `forward_token` loop.
///
/// * `rows` — output features (`out_dim`).
/// * `cols` — input features (`in_dim`).
/// * `inputs` — row-major `[batch, in_dim]`.
/// * `outputs` — row-major `[batch, out_dim]`.
pub fn gemm_quantized_f32(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), GemvError> {
    if batch == 0 {
        return Ok(());
    }
    if batch == 1 {
        return gemv_quantized_f32(quantization, quantized_matrix, rows, cols, inputs, outputs);
    }

    let expected_inputs = batch.saturating_mul(cols);
    if inputs.len() != expected_inputs {
        return Err(GemvError::InvalidVectorLength {
            expected: expected_inputs,
            actual: inputs.len(),
        });
    }
    let expected_outputs = batch.saturating_mul(rows);
    if outputs.len() != expected_outputs {
        return Err(GemvError::InvalidOutputLength {
            expected: expected_outputs,
            actual: outputs.len(),
        });
    }

    let profile_start = gemv_profile::enabled().then(std::time::Instant::now);
    let result = gemm_quantized_f32_inner(
        quantization,
        quantized_matrix,
        rows,
        cols,
        inputs,
        outputs,
        batch,
    );
    if let Some(start) = profile_start {
        gemv_profile::record(
            format!("gemm{batch} {quantization:?}"),
            rows,
            cols,
            quantized_matrix.len(),
            start.elapsed().as_nanos() as u64,
        );
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn gemm_quantized_f32_inner(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), GemvError> {
    // Fast path: decode each block once into a scratch f32 buffer, then do
    // `batch` AVX2 FMA dot products against it. Saves repeating the per-block
    // dequant for every batch token.
    match quantization {
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M
            if cols.is_multiple_of(QK_K) && q4_k_q8_k_avx2_available() =>
        {
            return gemm_q4_k_q8_k_fused(quantized_matrix, rows, cols, inputs, outputs, batch);
        }
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M
            if cols.is_multiple_of(QK_K) =>
        {
            return gemm_q4_k_decode_once(quantized_matrix, rows, cols, inputs, outputs, batch);
        }
        GgufQuantizationType::Q6_K if cols.is_multiple_of(QK_K) => {
            return gemm_q6_k_decode_once(quantized_matrix, rows, cols, inputs, outputs, batch);
        }
        GgufQuantizationType::Q8_0 if cols.is_multiple_of(QK8_0) => {
            return gemm_q8_0_decode_once(quantized_matrix, rows, cols, inputs, outputs, batch);
        }
        GgufQuantizationType::IQ1_S if cols.is_multiple_of(QK_K) => {
            return gemm_iq1_s_decode_once(quantized_matrix, rows, cols, inputs, outputs, batch);
        }
        GgufQuantizationType::IQ1_M if cols.is_multiple_of(QK_K) => {
            return gemm_iq1_m_decode_once(quantized_matrix, rows, cols, inputs, outputs, batch);
        }
        GgufQuantizationType::NVFP4 if cols.is_multiple_of(QK_NVFP4) => {
            return gemm_nvfp4_decode_once(quantized_matrix, rows, cols, inputs, outputs, batch);
        }
        _ => {}
    }

    // Generic K-quant path for Q2_K / Q6_K — uses existing per-block dot fns
    // and re-decodes per batch token. Slower but correct; optimize next.
    let k_quant = match quantization {
        GgufQuantizationType::Q2_K => Some((BLOCK_Q2_K_SIZE, q2_k_dot as fn(&[u8], &[f32]) -> f32)),
        GgufQuantizationType::Q6_K => Some((BLOCK_Q6_K_SIZE, q6_k_dot as fn(&[u8], &[f32]) -> f32)),
        _ => None,
    };

    if let Some((block_size, dot_fn)) = k_quant
        && cols.is_multiple_of(QK_K)
    {
        return gemm_k_quant_block(
            quantized_matrix,
            rows,
            cols,
            inputs,
            outputs,
            batch,
            QK_K,
            block_size,
            dot_fn,
            quantization,
        );
    }

    // Generic fallback: per-batch GEMV. Slower (rescans weight matrix per token)
    // but always correct; takes over for quant types or shapes the fused path
    // does not yet handle.
    for t in 0..batch {
        let in_slice = &inputs[t * cols..(t + 1) * cols];
        let out_slice = &mut outputs[t * rows..(t + 1) * rows];
        gemv_quantized_f32(
            quantization,
            quantized_matrix,
            rows,
            cols,
            in_slice,
            out_slice,
        )?;
    }
    Ok(())
}

/// AVX2 unpack of a 32-byte qs slice into 32 f32 values via
/// `dl * nibble - ml`. `high_nibble = true` selects the upper 4 bits, else
/// the lower 4 bits.
///
/// # Safety
/// `qs_ptr` addresses ≥32 bytes; `out_ptr` addresses ≥32 writable f32s. AVX2+FMA required.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn decode_q4_k_group_avx2(
    qs_ptr: *const u8,
    dl: f32,
    ml: f32,
    high_nibble: bool,
    out_ptr: *mut f32,
) {
    let dl_v = _mm256_set1_ps(dl);
    let ml_v = _mm256_set1_ps(ml);
    let mask = _mm_set1_epi8(0x0F);
    // 32 bytes split into 4 lanes of 8 bytes.
    for lane in 0..4 {
        let bytes = _mm_loadl_epi64(qs_ptr.add(lane * 8) as *const __m128i);
        let nibbles = if high_nibble {
            _mm_and_si128(_mm_srli_epi16(bytes, 4), mask)
        } else {
            _mm_and_si128(bytes, mask)
        };
        let i32x8 = _mm256_cvtepu8_epi32(nibbles);
        let f32x8 = _mm256_cvtepi32_ps(i32x8);
        // dl * f - ml
        let result = _mm256_fmsub_ps(f32x8, dl_v, ml_v);
        _mm256_storeu_ps(out_ptr.add(lane * 8), result);
    }
}

/// Decode a single 144-byte Q4_K block into 256 f32 values.
/// Matches [`q4_k_value`] semantics exactly.
#[inline]
fn decode_q4_k_block(block: &[u8], out: &mut [f32]) {
    debug_assert!(out.len() >= QK_K);
    debug_assert!(block.len() >= BLOCK_Q4_K_SIZE);
    let d = f16_le_to_f32([block[0], block[1]]);
    let min = f16_le_to_f32([block[2], block[3]]);
    let scales = &block[4..16];
    let qs = &block[16..144];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                for g in 0..8 {
                    let (sc, m) = get_scale_min_k4(g, scales);
                    let dl = d * sc as f32;
                    let ml = min * m as f32;
                    let pair = g / 2;
                    let qs_ptr = qs.as_ptr().add(pair * 32);
                    let out_ptr = out.as_mut_ptr().add(g * 32);
                    decode_q4_k_group_avx2(qs_ptr, dl, ml, g & 1 != 0, out_ptr);
                }
            }
            return;
        }
    }

    for g in 0..8 {
        let (sc, m) = get_scale_min_k4(g, scales);
        let dl = d * sc as f32;
        let ml = min * m as f32;
        let pair = g / 2;
        let qs_off = pair * 32;
        let out_off = g * 32;
        if g & 1 == 0 {
            for l in 0..32 {
                out[out_off + l] = dl * ((qs[qs_off + l] & 0x0f) as f32) - ml;
            }
        } else {
            for l in 0..32 {
                out[out_off + l] = dl * ((qs[qs_off + l] >> 4) as f32) - ml;
            }
        }
    }
}

/// Decode a single 34-byte Q8_0 block into 32 f32 values.
#[inline]
fn decode_q8_0_block(block: &[u8], out: &mut [f32]) {
    debug_assert!(out.len() >= QK8_0);
    debug_assert!(block.len() >= BLOCK_Q8_0_SIZE);
    let scale = f16_le_to_f32([block[0], block[1]]);
    for l in 0..QK8_0 {
        out[l] = (block[2 + l] as i8) as f32 * scale;
    }
}

/// AVX2 + FMA dot product over `len` f32 elements. `len` is expected to be a
/// multiple of 8; a tail loop handles any remainder.
///
/// # Safety
/// `a` and `b` must each address at least `len` initialized f32 elements; `len` may be
/// zero. Caller must ensure AVX2+FMA is available.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 32 <= len {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.add(i + 8)),
            _mm256_loadu_ps(b.add(i + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.add(i + 16)),
            _mm256_loadu_ps(b.add(i + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.add(i + 24)),
            _mm256_loadu_ps(b.add(i + 24)),
            acc3,
        );
        i += 32;
    }
    while i + 8 <= len {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc0);
        i += 8;
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let mut sum = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
    while i < len {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

/// Eight dot products that share the `a` load. Writes results to `out[0..8]`.
/// At eight accumulators we exceed the 16-ymm register file by a few — the
/// compiler spills the inputs but the load amortization on `a` is the dominant
/// effect.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn, dead_code)]
unsafe fn dot8_f32_avx2(a: *const f32, b: [*const f32; 8], len: usize, out: &mut [f32; 8]) {
    let mut acc = [_mm256_setzero_ps(); 8];
    let mut i = 0;
    while i + 8 <= len {
        let av = _mm256_loadu_ps(a.add(i));
        acc[0] = _mm256_fmadd_ps(av, _mm256_loadu_ps(b[0].add(i)), acc[0]);
        acc[1] = _mm256_fmadd_ps(av, _mm256_loadu_ps(b[1].add(i)), acc[1]);
        acc[2] = _mm256_fmadd_ps(av, _mm256_loadu_ps(b[2].add(i)), acc[2]);
        acc[3] = _mm256_fmadd_ps(av, _mm256_loadu_ps(b[3].add(i)), acc[3]);
        acc[4] = _mm256_fmadd_ps(av, _mm256_loadu_ps(b[4].add(i)), acc[4]);
        acc[5] = _mm256_fmadd_ps(av, _mm256_loadu_ps(b[5].add(i)), acc[5]);
        acc[6] = _mm256_fmadd_ps(av, _mm256_loadu_ps(b[6].add(i)), acc[6]);
        acc[7] = _mm256_fmadd_ps(av, _mm256_loadu_ps(b[7].add(i)), acc[7]);
        i += 8;
    }
    let hsum = |a: __m256| -> f32 {
        let lo = _mm256_castps256_ps128(a);
        let hi = _mm256_extractf128_ps(a, 1);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s);
        let sums = _mm_add_ps(s, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    };
    for j in 0..8 {
        out[j] = hsum(acc[j]);
    }
    while i < len {
        let av = *a.add(i);
        for j in 0..8 {
            out[j] += av * *b[j].add(i);
        }
        i += 1;
    }
}

/// Four dot products that share the `a` load: `(a·b0, a·b1, a·b2, a·b3)`.
/// Halves the L1 traffic on `a` versus four separate `dot_f32_avx2` calls,
/// which is the dominant cost in batched quantized GEMM after the per-row
/// weight scan amortization.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn dot4_f32_avx2(
    a: *const f32,
    b0: *const f32,
    b1: *const f32,
    b2: *const f32,
    b3: *const f32,
    len: usize,
) -> (f32, f32, f32, f32) {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        let av = _mm256_loadu_ps(a.add(i));
        acc0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b0.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b1.add(i)), acc1);
        acc2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b2.add(i)), acc2);
        acc3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b3.add(i)), acc3);
        i += 8;
    }
    let hsum = |acc: __m256| -> f32 {
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s);
        let sums = _mm_add_ps(s, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    };
    let mut s0 = hsum(acc0);
    let mut s1 = hsum(acc1);
    let mut s2 = hsum(acc2);
    let mut s3 = hsum(acc3);
    while i < len {
        let av = *a.add(i);
        s0 += av * *b0.add(i);
        s1 += av * *b1.add(i);
        s2 += av * *b2.add(i);
        s3 += av * *b3.add(i);
        i += 1;
    }
    (s0, s1, s2, s3)
}

/// AVX-512 counterpart of [`dot4_f32_avx2`]: 16-wide FMA, four shared-`a`
/// accumulators. On Skylake-SP this doubles the dot lanes versus AVX2 while the
/// 32-zmm register file absorbs the four input streams without spilling.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn dot4_f32_avx512(
    a: *const f32,
    b0: *const f32,
    b1: *const f32,
    b2: *const f32,
    b3: *const f32,
    len: usize,
) -> (f32, f32, f32, f32) {
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= len {
        let av = _mm512_loadu_ps(a.add(i));
        acc0 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b0.add(i)), acc0);
        acc1 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b1.add(i)), acc1);
        acc2 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b2.add(i)), acc2);
        acc3 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b3.add(i)), acc3);
        i += 16;
    }
    let mut s0 = _mm512_reduce_add_ps(acc0);
    let mut s1 = _mm512_reduce_add_ps(acc1);
    let mut s2 = _mm512_reduce_add_ps(acc2);
    let mut s3 = _mm512_reduce_add_ps(acc3);
    while i < len {
        let av = *a.add(i);
        s0 += av * *b0.add(i);
        s1 += av * *b1.add(i);
        s2 += av * *b2.add(i);
        s3 += av * *b3.add(i);
        i += 1;
    }
    (s0, s1, s2, s3)
}

/// AVX-512 counterpart of [`dot_f32_avx2`].
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn dot_f32_avx512(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0;
    while i + 32 <= len {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a.add(i)), _mm512_loadu_ps(b.add(i)), acc0);
        acc1 = _mm512_fmadd_ps(
            _mm512_loadu_ps(a.add(i + 16)),
            _mm512_loadu_ps(b.add(i + 16)),
            acc1,
        );
        i += 32;
    }
    while i + 16 <= len {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a.add(i)), _mm512_loadu_ps(b.add(i)), acc0);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    while i < len {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

#[inline]
fn dot_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            return unsafe { dot_f32_avx512(a.as_ptr(), b.as_ptr(), a.len()) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_avx2(a.as_ptr(), b.as_ptr(), a.len()) };
        }
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Fast batched Q4_K GEMM path: for each output row, walk each 144-byte block
/// once, decode it into a 256-element f32 scratch buffer, then for each batch
/// token compute `scratch · input_chunk` with AVX2 FMA. Decode cost is paid
/// once per row block; the dot product is paid per token but hits L1 since both
/// operands are tiny.
fn gemm_q4_k_decode_once(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows
        .saturating_mul(blocks_per_row)
        .saturating_mul(BLOCK_Q4_K_SIZE);
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe {
                gemm_q4_k_decode_once_avx2(
                    quantized_matrix,
                    rows,
                    cols,
                    inputs,
                    outputs,
                    batch,
                    blocks_per_row,
                )
            };
        }
    }

    let mut row_major = vec![0.0_f32; rows.saturating_mul(batch)];
    let compute_row = |row_idx: usize, partial: &mut [f32]| {
        let mut scratch = [0.0_f32; QK_K];
        let row_start = row_idx * blocks_per_row * BLOCK_Q4_K_SIZE;
        let row_blocks = &quantized_matrix[row_start..row_start + blocks_per_row * BLOCK_Q4_K_SIZE];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q4_K_SIZE).enumerate() {
            decode_q4_k_block(block, &mut scratch);
            let in_offset = block_idx * QK_K;
            for t in 0..batch {
                let v = &inputs[t * cols + in_offset..t * cols + in_offset + QK_K];
                partial[t] += dot_f32_fast(&scratch, v);
            }
        }
    };

    row_major
        .par_chunks_mut(batch)
        .enumerate()
        .for_each(|(row_idx, slice)| compute_row(row_idx, slice));

    for r in 0..rows {
        let src = &row_major[r * batch..(r + 1) * batch];
        for (t, &val) in src.iter().enumerate() {
            outputs[t * rows + r] = val;
        }
    }
    Ok(())
}

/// AVX2/FMA hot-path body of [`gemm_q4_k_decode_once`]. Hoists the runtime
/// CPU-feature check out of the (rows × blocks × batch) inner loop and lets
/// the compiler inline `decode_q4_k_group_avx2` + the dot kernel directly.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn gemm_q4_k_decode_once_avx2(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
    blocks_per_row: usize,
) -> Result<(), GemvError> {
    let qm_ptr_addr = quantized_matrix.as_ptr() as usize;
    let in_ptr_addr = inputs.as_ptr() as usize;
    let out_ptr_addr = outputs.as_mut_ptr() as usize;
    let row_stride_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;

    // The weight decode stays AVX2, but the per-token dot over the 256-element
    // f32 scratch runs 16-wide on AVX-512 hardware (Skylake-SP). Detected once;
    // the inner branch is perfectly predicted.
    let use_avx512 =
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");

    // Rows are processed in chunks to amortize rayon task dispatch overhead.
    const ROW_CHUNK: usize = 16;
    let process_row = |row_idx: usize, partial: &mut [f32]| {
        let qm_ptr = qm_ptr_addr as *const u8;
        let in_ptr = in_ptr_addr as *const f32;
        let mut scratch = [0.0_f32; QK_K];
        partial.fill(0.0);
        let row_base = unsafe { qm_ptr.add(row_idx * row_stride_bytes) };
        for block_idx in 0..blocks_per_row {
            // SAFETY: `row_base` points into the packed matrix row; each block is `BLOCK_Q4_K_SIZE`
            // bytes and `block_idx` is bounded by `blocks_per_row`.
            let block_ptr = unsafe { row_base.add(block_idx * BLOCK_Q4_K_SIZE) };
            let block = unsafe { std::slice::from_raw_parts(block_ptr, BLOCK_Q4_K_SIZE) };
            let d = f16_le_to_f32([block[0], block[1]]);
            let min = f16_le_to_f32([block[2], block[3]]);
            let scales = &block[4..16];
            let qs_ptr = unsafe { block_ptr.add(16) };
            for g in 0..8 {
                let (sc, m) = get_scale_min_k4(g, scales);
                let dl = d * sc as f32;
                let ml = min * m as f32;
                let pair = g / 2;
                unsafe {
                    decode_q4_k_group_avx2(
                        qs_ptr.add(pair * 32),
                        dl,
                        ml,
                        g & 1 != 0,
                        scratch.as_mut_ptr().add(g * 32),
                    );
                }
            }
            let in_offset_floats = block_idx * QK_K;
            // Process batch in groups of 4 — sweet spot for the 16 ymm
            // register file (1 scratch broadcast + 4 inputs + 4 accumulators
            // fits with headroom). dot8 was tried and regresses due to spill.
            let chunks = batch / 4;
            let tail = batch % 4;
            for ck in 0..chunks {
                let t0 = ck * 4;
                let v0 = unsafe { in_ptr.add((t0) * cols + in_offset_floats) };
                let v1 = unsafe { in_ptr.add((t0 + 1) * cols + in_offset_floats) };
                let v2 = unsafe { in_ptr.add((t0 + 2) * cols + in_offset_floats) };
                let v3 = unsafe { in_ptr.add((t0 + 3) * cols + in_offset_floats) };
                let (s0, s1, s2, s3) = if use_avx512 {
                    unsafe { dot4_f32_avx512(scratch.as_ptr(), v0, v1, v2, v3, QK_K) }
                } else {
                    unsafe { dot4_f32_avx2(scratch.as_ptr(), v0, v1, v2, v3, QK_K) }
                };
                partial[t0] += s0;
                partial[t0 + 1] += s1;
                partial[t0 + 2] += s2;
                partial[t0 + 3] += s3;
            }
            for ti in 0..tail {
                let t = chunks * 4 + ti;
                let v_ptr = unsafe { in_ptr.add(t * cols + in_offset_floats) };
                partial[t] += if use_avx512 {
                    unsafe { dot_f32_avx512(scratch.as_ptr(), v_ptr, QK_K) }
                } else {
                    unsafe { dot_f32_avx2(scratch.as_ptr(), v_ptr, QK_K) }
                };
            }
        }
    };

    // Keep per-row results in a row-major panel while workers are active.
    // Writing the caller's `[batch, rows]` layout directly from parallel row
    // workers causes cross-thread cache-line ping-pong on adjacent output rows,
    // which is especially painful for DFlash pp32. The final transpose is
    // linear and much cheaper than that false sharing.
    let mut row_major = vec![0.0_f32; rows.saturating_mul(batch)];
    row_major
        .par_chunks_mut(ROW_CHUNK * batch)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx * ROW_CHUNK;
            let end = (start + ROW_CHUNK).min(rows);
            for row in start..end {
                let local = row - start;
                let partial = &mut chunk[local * batch..(local + 1) * batch];
                process_row(row, partial);
            }
        });

    let out_ptr = out_ptr_addr as *mut f32;
    for r in 0..rows {
        let src = &row_major[r * batch..(r + 1) * batch];
        for (t, &val) in src.iter().enumerate() {
            unsafe { *out_ptr.add(t * rows + r) = val };
        }
    }
    Ok(())
}

/// Fast batched Q8_0 GEMM, same shape as [`gemm_q4_k_decode_once`].
fn gemm_q8_0_decode_once(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK8_0;
    let expected_matrix_len = rows
        .saturating_mul(blocks_per_row)
        .saturating_mul(BLOCK_Q8_0_SIZE);
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }

    let mut row_major = vec![0.0_f32; rows.saturating_mul(batch)];
    let compute_row = |row_idx: usize, partial: &mut [f32]| {
        let mut scratch = [0.0_f32; QK8_0];
        let row_start = row_idx * blocks_per_row * BLOCK_Q8_0_SIZE;
        let row_blocks = &quantized_matrix[row_start..row_start + blocks_per_row * BLOCK_Q8_0_SIZE];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q8_0_SIZE).enumerate() {
            decode_q8_0_block(block, &mut scratch);
            let in_offset = block_idx * QK8_0;
            for t in 0..batch {
                let v = &inputs[t * cols + in_offset..t * cols + in_offset + QK8_0];
                partial[t] += dot_f32_fast(&scratch, v);
            }
        }
    };

    row_major
        .par_chunks_mut(batch)
        .enumerate()
        .for_each(|(row_idx, slice)| compute_row(row_idx, slice));

    for r in 0..rows {
        let src = &row_major[r * batch..(r + 1) * batch];
        for (t, &val) in src.iter().enumerate() {
            outputs[t * rows + r] = val;
        }
    }
    Ok(())
}

/// Q6_K batched GEMM with decode-once optimization. Mirrors
/// `gemm_q4_k_decode_once` for 6-bit super-blocks: decode each 256-element
/// block to f32 once, then fan out a fast f32 dot against every batch token.
fn gemm_q6_k_decode_once(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows
        .saturating_mul(blocks_per_row)
        .saturating_mul(BLOCK_Q6_K_SIZE);
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }

    let mut row_major = vec![0.0_f32; rows.saturating_mul(batch)];
    let qm_ptr_addr = quantized_matrix.as_ptr() as usize;
    let in_ptr_addr = inputs.as_ptr() as usize;
    let row_stride_bytes = blocks_per_row * BLOCK_Q6_K_SIZE;

    const ROW_CHUNK: usize = 16;
    row_major
        .par_chunks_mut(ROW_CHUNK * batch)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let qm_ptr = qm_ptr_addr as *const u8;
            let in_ptr = in_ptr_addr as *const f32;
            let start = chunk_idx * ROW_CHUNK;
            let end = (start + ROW_CHUNK).min(rows);
            let use_avx512 =
                is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");
            let mut scratch = [0.0_f32; QK_K];
            for row in start..end {
                let local = row - start;
                let partial = &mut chunk[local * batch..(local + 1) * batch];
                partial.fill(0.0);
                let row_base = unsafe { qm_ptr.add(row * row_stride_bytes) };
                for block_idx in 0..blocks_per_row {
                    let block_ptr = unsafe { row_base.add(block_idx * BLOCK_Q6_K_SIZE) };
                    let block = unsafe { std::slice::from_raw_parts(block_ptr, BLOCK_Q6_K_SIZE) };
                    decode_q6_k_block(block, &mut scratch);
                    let in_offset_floats = block_idx * QK_K;
                    let chunks = batch / 4;
                    let tail = batch % 4;
                    for ck in 0..chunks {
                        let t0 = ck * 4;
                        let v0 = unsafe { in_ptr.add(t0 * cols + in_offset_floats) };
                        let v1 = unsafe { in_ptr.add((t0 + 1) * cols + in_offset_floats) };
                        let v2 = unsafe { in_ptr.add((t0 + 2) * cols + in_offset_floats) };
                        let v3 = unsafe { in_ptr.add((t0 + 3) * cols + in_offset_floats) };
                        let (s0, s1, s2, s3) = if use_avx512 {
                            unsafe { dot4_f32_avx512(scratch.as_ptr(), v0, v1, v2, v3, QK_K) }
                        } else {
                            unsafe { dot4_f32_avx2(scratch.as_ptr(), v0, v1, v2, v3, QK_K) }
                        };
                        partial[t0] += s0;
                        partial[t0 + 1] += s1;
                        partial[t0 + 2] += s2;
                        partial[t0 + 3] += s3;
                    }
                    for ti in 0..tail {
                        let t = chunks * 4 + ti;
                        let v_ptr = unsafe { in_ptr.add(t * cols + in_offset_floats) };
                        partial[t] += if use_avx512 {
                            unsafe { dot_f32_avx512(scratch.as_ptr(), v_ptr, QK_K) }
                        } else {
                            unsafe { dot_f32_avx2(scratch.as_ptr(), v_ptr, QK_K) }
                        };
                    }
                }
            }
        });

    for r in 0..rows {
        let src = &row_major[r * batch..(r + 1) * batch];
        for (t, &val) in src.iter().enumerate() {
            outputs[t * rows + r] = val;
        }
    }
    Ok(())
}

/// Decode a single 210-byte Q6_K block into 256 f32 values, matching
/// `q6_k_dot_scalar`'s semantics.
#[inline]
fn decode_q6_k_block(block: &[u8], out: &mut [f32]) {
    debug_assert!(out.len() >= QK_K);
    debug_assert!(block.len() >= BLOCK_Q6_K_SIZE);
    let d = f16_le_to_f32([block[208], block[209]]);
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let mut q_ptr = 0;
    for half in 0..2 {
        let scale_base = half * 8;
        let ql_base = half * 64;
        let qh_base = half * 32;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_base + l] & 0x0f) as i32 | (((qh[qh_base + l] & 3) as i32) << 4)) - 32;
            let q2 = ((ql[ql_base + l + 32] & 0x0f) as i32
                | ((((qh[qh_base + l] >> 2) & 3) as i32) << 4))
                - 32;
            let q3 =
                ((ql[ql_base + l] >> 4) as i32 | ((((qh[qh_base + l] >> 4) & 3) as i32) << 4)) - 32;
            let q4 = ((ql[ql_base + l + 32] >> 4) as i32
                | ((((qh[qh_base + l] >> 6) & 3) as i32) << 4))
                - 32;
            out[q_ptr + l] = d * sc[scale_base + is] as i8 as f32 * q1 as f32;
            out[q_ptr + 32 + l] = d * sc[scale_base + is + 2] as i8 as f32 * q2 as f32;
            out[q_ptr + 64 + l] = d * sc[scale_base + is + 4] as i8 as f32 * q3 as f32;
            out[q_ptr + 96 + l] = d * sc[scale_base + is + 6] as i8 as f32 * q4 as f32;
        }
        q_ptr += 128;
    }
}

fn gemm_iq1_s_decode_once(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_IQ1_S_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }

    let mut row_major = vec![0.0_f32; rows.saturating_mul(batch)];
    let compute_row = |row_idx: usize, partial: &mut [f32]| {
        let row_start = row_idx * blocks_per_row * BLOCK_IQ1_S_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + blocks_per_row * BLOCK_IQ1_S_SIZE];
        let mut scratch = [0.0_f32; QK_K];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_IQ1_S_SIZE).enumerate() {
            iq1s_dequantize_block(block, &mut scratch);
            let in_offset = block_idx * QK_K;
            for t in 0..batch {
                let v = &inputs[t * cols + in_offset..t * cols + in_offset + QK_K];
                partial[t] += crate::flash_attention::dot_product_f32(&scratch, v);
            }
        }
    };

    let total_ops = rows.saturating_mul(cols).saturating_mul(batch);
    if total_ops >= PARALLEL_GEMV_MIN_OPS {
        row_major
            .par_chunks_mut(batch)
            .enumerate()
            .for_each(|(row_idx, slice)| compute_row(row_idx, slice));
    } else {
        for (row_idx, slice) in row_major.chunks_mut(batch).enumerate() {
            compute_row(row_idx, slice);
        }
    }

    for r in 0..rows {
        let src = &row_major[r * batch..(r + 1) * batch];
        for (t, &val) in src.iter().enumerate() {
            outputs[t * rows + r] = val;
        }
    }
    Ok(())
}

fn gemm_iq1_m_decode_once(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_IQ1_M_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }

    let mut row_major = vec![0.0_f32; rows.saturating_mul(batch)];
    let compute_row = |row_idx: usize, partial: &mut [f32]| {
        let row_start = row_idx * blocks_per_row * BLOCK_IQ1_M_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + blocks_per_row * BLOCK_IQ1_M_SIZE];
        let mut scratch = [0.0_f32; QK_K];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_IQ1_M_SIZE).enumerate() {
            iq1m_dequantize_block(block, &mut scratch);
            let in_offset = block_idx * QK_K;
            for t in 0..batch {
                let v = &inputs[t * cols + in_offset..t * cols + in_offset + QK_K];
                partial[t] += crate::flash_attention::dot_product_f32(&scratch, v);
            }
        }
    };

    let total_ops = rows.saturating_mul(cols).saturating_mul(batch);
    if total_ops >= PARALLEL_GEMV_MIN_OPS {
        row_major
            .par_chunks_mut(batch)
            .enumerate()
            .for_each(|(row_idx, slice)| compute_row(row_idx, slice));
    } else {
        for (row_idx, slice) in row_major.chunks_mut(batch).enumerate() {
            compute_row(row_idx, slice);
        }
    }

    for r in 0..rows {
        let src = &row_major[r * batch..(r + 1) * batch];
        for (t, &val) in src.iter().enumerate() {
            outputs[t * rows + r] = val;
        }
    }
    Ok(())
}

fn gemm_nvfp4_decode_once(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_NVFP4;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_NVFP4_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }

    let mut row_major = vec![0.0_f32; rows.saturating_mul(batch)];
    let compute_row = |row_idx: usize, partial: &mut [f32]| {
        let row_start = row_idx * blocks_per_row * BLOCK_NVFP4_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + blocks_per_row * BLOCK_NVFP4_SIZE];
        let mut scratch = [0.0_f32; QK_NVFP4];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_NVFP4_SIZE).enumerate() {
            nvfp4_dequantize_block(block, &mut scratch);
            let in_offset = block_idx * QK_NVFP4;
            for t in 0..batch {
                let v = &inputs[t * cols + in_offset..t * cols + in_offset + QK_NVFP4];
                partial[t] += dot_f32_fast(&scratch, v);
            }
        }
    };

    if rows.saturating_mul(cols).saturating_mul(batch) >= PARALLEL_GEMV_MIN_OPS {
        row_major
            .par_chunks_mut(batch)
            .enumerate()
            .for_each(|(row_idx, slice)| compute_row(row_idx, slice));
    } else {
        for (row_idx, slice) in row_major.chunks_mut(batch).enumerate() {
            compute_row(row_idx, slice);
        }
    }

    for r in 0..rows {
        let src = &row_major[r * batch..(r + 1) * batch];
        for (t, &val) in src.iter().enumerate() {
            outputs[t * rows + r] = val;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gemm_k_quant_block(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
    k_block: usize,
    block_size: usize,
    dot_fn: fn(&[u8], &[f32]) -> f32,
    quantization: GgufQuantizationType,
) -> Result<(), GemvError> {
    let blocks_per_row = cols / k_block;
    let expected_matrix_len = rows
        .saturating_mul(blocks_per_row)
        .saturating_mul(block_size);
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    let _ = quantization;

    // Compute output into `[rows, batch]` panel layout so each parallel worker
    // owns one full output row and writes contiguously. We transpose into the
    // caller's `[batch, rows]` layout at the end. This is the generic
    // dot-per-(row, block, token) path used for Q2_K and Q6_K; Q4_K and Q8_0
    // take the faster decode-once-multiply-many path above.
    let mut row_major = vec![0.0_f32; rows.saturating_mul(batch)];
    let compute_row = |row_idx: usize, partial: &mut [f32]| {
        let row_start = row_idx * blocks_per_row * block_size;
        let row_blocks = &quantized_matrix[row_start..row_start + blocks_per_row * block_size];
        for (block_idx, block) in row_blocks.chunks_exact(block_size).enumerate() {
            let in_offset = block_idx * k_block;
            for t in 0..batch {
                let v = &inputs[t * cols + in_offset..t * cols + in_offset + k_block];
                partial[t] += dot_fn(block, v);
            }
        }
    };

    let total_ops = rows.saturating_mul(cols).saturating_mul(batch);
    if total_ops >= PARALLEL_GEMV_MIN_OPS {
        row_major
            .par_chunks_mut(batch)
            .enumerate()
            .for_each(|(row_idx, slice)| compute_row(row_idx, slice));
    } else {
        for (row_idx, slice) in row_major.chunks_mut(batch).enumerate() {
            compute_row(row_idx, slice);
        }
    }

    for r in 0..rows {
        let src = &row_major[r * batch..(r + 1) * batch];
        for (t, &val) in src.iter().enumerate() {
            outputs[t * rows + r] = val;
        }
    }
    Ok(())
}

/// Batched expert GEMV over a set of selected experts for one projection.
///
/// `matrix` holds `n_experts` contiguous row-major `[rows, cols]` expert weight
/// blocks. For each selected expert `selected[slot]` this writes
/// `output[slot*rows + r] = W_expert[r] · input_slot`. `input_stride == 0` means
/// every expert shares `inputs[..cols]` (gate / up projections); otherwise expert
/// `slot` uses `inputs[slot*input_stride .. slot*input_stride + cols]` (down
/// projection, where each expert has its own activation).
///
/// The whole thing runs as a single parallel region over all `(slot, row)` pairs,
/// which avoids the per-expert, per-projection rayon dispatch overhead that
/// dominates MoE decode (12 separate parallel calls per layer otherwise).
#[allow(clippy::too_many_arguments)]
pub fn gemv_quantized_experts_f32(
    quantization: GgufQuantizationType,
    matrix: &[u8],
    n_experts: usize,
    selected: &[usize],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    input_stride: usize,
    output: &mut [f32],
) -> Result<(), GemvError> {
    let n_sel = selected.len();
    if output.len() != n_sel * rows {
        return Err(GemvError::InvalidOutputLength {
            expected: n_sel * rows,
            actual: output.len(),
        });
    }
    if n_experts == 0 || matrix.is_empty() {
        return Err(GemvError::InvalidMatrixLength {
            expected: 1,
            actual: matrix.len(),
        });
    }
    let expert_bytes = matrix.len() / n_experts;
    let row_bytes = expert_bytes / rows.max(1);
    let shared = input_stride == 0;
    let input_for = |slot: usize| -> &[f32] {
        let base = if shared { 0 } else { slot * input_stride };
        &inputs[base..base + cols]
    };

    // Fast path: Q4_K × Q8_K AVX2. Quantize each distinct input to Q8_K once.
    if matches!(
        quantization,
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M
    ) && cols.is_multiple_of(QK_K)
        && q4_k_q8_k_avx2_available()
    {
        let blocks_per_row = cols / QK_K;
        let q8_stride = blocks_per_row * BLOCK_Q8_K_BYTES;
        let n_inputs = if shared { 1 } else { n_sel };
        let mut q8k = vec![0_u8; n_inputs * q8_stride];
        for s in 0..n_inputs {
            quantize_vector_q8_k_into(
                input_for(s),
                blocks_per_row,
                &mut q8k[s * q8_stride..(s + 1) * q8_stride],
            );
        }
        // 4-row custom kernel: shares the Q8_K input across rows and runs 4
        // independent accumulator chains to overlap DRAM latency. Chunks of 32
        // never span an expert slot when 32 divides `rows`. VNNI machines keep
        // their per-row VNNI kernel instead.
        let use_x4 = cfg!(any(target_arch = "x86", target_arch = "x86_64"))
            && !q4_k_q8_k_vnni_available()
            && rows.is_multiple_of(32);
        if use_x4 {
            run_output_chunks(output, GEMV_CHUNK_ROWS, |chunk_idx, out_chunk| {
                let matrix = crate::numa::local_slice(matrix);
                let i0 = chunk_idx * GEMV_CHUNK_ROWS;
                let slot = i0 / rows;
                let row0 = i0 % rows;
                let expert = selected[slot];
                let qs = if shared { 0 } else { slot };
                let q8 = &q8k[qs * q8_stride..(qs + 1) * q8_stride];
                // OXK opt-in (OXIDIZE_GEMV=oxk): same chunk, ×8 kernels.
                #[cfg(feature = "oxk")]
                if gemv_mode() == GemvMode::Oxk {
                    let start = expert * expert_bytes + row0 * row_bytes;
                    let end = start + out_chunk.len() * row_bytes;
                    oxidize_kernels::gemv_q4k_range(
                        &matrix[start..end],
                        blocks_per_row,
                        q8,
                        out_chunk,
                    );
                    return;
                }
                let mut r = 0;
                while r < out_chunk.len() {
                    if r + 4 <= out_chunk.len() {
                        let base = unsafe {
                            matrix
                                .as_ptr()
                                .add(expert * expert_bytes + (row0 + r) * row_bytes)
                        };
                        let mut quad = [0.0_f32; 4];
                        // Safety: avx2 verified by q4_k_q8_k_avx2_available();
                        // rows stay inside this expert because 32 | rows.
                        unsafe {
                            q4_k_q8_k_row_dot_x4_avx2(
                                base,
                                row_bytes,
                                blocks_per_row,
                                q8,
                                &mut quad,
                            )
                        };
                        out_chunk[r..r + 4].copy_from_slice(&quad);
                        r += 4;
                    } else {
                        let row_start = expert * expert_bytes + (row0 + r) * row_bytes;
                        let rowb = &matrix[row_start..row_start + row_bytes];
                        out_chunk[r] = unsafe { q4_k_q8_k_row_dot(rowb, blocks_per_row, q8) };
                        r += 1;
                    }
                }
            });
            return Ok(());
        }
        // with_min_len keeps rayon from splitting into per-row tasks; each row
        // dot is only ~1-3us, so fine splits drown in steal/join overhead.
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(i, out)| {
                let slot = i / rows;
                let row = i % rows;
                let expert = selected[slot];
                let row_start = expert * expert_bytes + row * row_bytes;
                let rowb = &matrix[row_start..row_start + row_bytes];
                let qs = if shared { 0 } else { slot };
                let q8 = &q8k[qs * q8_stride..(qs + 1) * q8_stride];
                // Safety: q4_k_q8_k_avx2_available() checked above; dispatcher picks
                // the VNNI kernel when the runtime supports it.
                *out = unsafe { q4_k_q8_k_row_dot(rowb, blocks_per_row, q8) };
            });
        return Ok(());
    }

    // Fast path: Q6_K x Q8_K integer kernel. Quantize each distinct input to
    // Q8_K once, then 4-row chunks share the input loads (same structure as
    // the Q4_K expert path).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if matches!(quantization, GgufQuantizationType::Q6_K)
        && cols.is_multiple_of(QK_K)
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        let blocks_per_row = cols / QK_K;
        let q8_stride = blocks_per_row * BLOCK_Q8_K_BYTES;
        let n_inputs = if shared { 1 } else { n_sel };
        let mut q8k = vec![0_u8; n_inputs * q8_stride];
        for s in 0..n_inputs {
            quantize_vector_q8_k_into(
                input_for(s),
                blocks_per_row,
                &mut q8k[s * q8_stride..(s + 1) * q8_stride],
            );
        }
        if rows.is_multiple_of(32) {
            run_output_chunks(output, GEMV_CHUNK_ROWS, |chunk_idx, out_chunk| {
                let matrix = crate::numa::local_slice(matrix);
                let i0 = chunk_idx * GEMV_CHUNK_ROWS;
                let slot = i0 / rows;
                let row0 = i0 % rows;
                let expert = selected[slot];
                let qs = if shared { 0 } else { slot };
                let q8 = &q8k[qs * q8_stride..(qs + 1) * q8_stride];
                let mut r = 0;
                while r < out_chunk.len() {
                    if r + 4 <= out_chunk.len() {
                        let base = unsafe {
                            matrix
                                .as_ptr()
                                .add(expert * expert_bytes + (row0 + r) * row_bytes)
                        };
                        let mut quad = [0.0_f32; 4];
                        // Safety: avx2+fma checked above; 32 | rows keeps
                        // the quad inside this expert's rows.
                        unsafe {
                            q6_k_q8_k_row_dot_x4_avx2(
                                base,
                                row_bytes,
                                blocks_per_row,
                                q8,
                                &mut quad,
                            )
                        };
                        out_chunk[r..r + 4].copy_from_slice(&quad);
                        r += 4;
                    } else {
                        let row_start = expert * expert_bytes + (row0 + r) * row_bytes;
                        let rowb = &matrix[row_start..row_start + row_bytes];
                        out_chunk[r] = unsafe { q6_k_q8_k_row_dot_avx2(rowb, blocks_per_row, q8) };
                        r += 1;
                    }
                }
            });
        } else {
            output
                .par_iter_mut()
                .with_min_len(32)
                .enumerate()
                .for_each(|(i, out)| {
                    let slot = i / rows;
                    let row = i % rows;
                    let expert = selected[slot];
                    let row_start = expert * expert_bytes + row * row_bytes;
                    let rowb = &matrix[row_start..row_start + row_bytes];
                    let qs = if shared { 0 } else { slot };
                    let q8 = &q8k[qs * q8_stride..(qs + 1) * q8_stride];
                    // Safety: avx2+fma checked above.
                    *out = unsafe { q6_k_q8_k_row_dot_avx2(rowb, blocks_per_row, q8) };
                });
        }
        return Ok(());
    }

    // Generic fallback: one parallel gemv per expert.
    for (slot, &expert) in selected.iter().enumerate() {
        let start = expert * expert_bytes;
        gemv_quantized_f32(
            quantization,
            &matrix[start..start + expert_bytes],
            rows,
            cols,
            input_for(slot),
            &mut output[slot * rows..(slot + 1) * rows],
        )?;
    }
    Ok(())
}

/// Fused gate+up expert GEMV: computes both MoE projections in ONE parallel
/// region (instead of two), halving the fork/join + steal overhead of the two
/// biggest per-layer dispatches during decode. `output` is `[2 * n_sel * rows]`
/// with the gate results in the first half and up results in the second.
///
/// Falls back to two [`gemv_quantized_experts_f32`] calls whenever the fused
/// fast-path conditions don't hold (non-Q4_K, rows not a multiple of 32, VNNI
/// machines, mismatched matrix sizes).
#[allow(clippy::too_many_arguments)]
pub fn gemv_quantized_experts_gate_up_f32(
    quantization: GgufQuantizationType,
    gate_matrix: &[u8],
    up_matrix: &[u8],
    n_experts: usize,
    selected: &[usize],
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let n_sel = selected.len();
    let half = n_sel * rows;
    if output.len() != 2 * half {
        return Err(GemvError::InvalidOutputLength {
            expected: 2 * half,
            actual: output.len(),
        });
    }
    let fused_ok = matches!(
        quantization,
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M
    ) && cols.is_multiple_of(QK_K)
        && q4_k_q8_k_avx2_available()
        && !q4_k_q8_k_vnni_available()
        && rows.is_multiple_of(32)
        && gate_matrix.len() == up_matrix.len()
        && n_experts > 0
        && !gate_matrix.is_empty();
    if !fused_ok {
        let (gate_out, up_out) = output.split_at_mut(half);
        gemv_quantized_experts_f32(
            quantization,
            gate_matrix,
            n_experts,
            selected,
            rows,
            cols,
            input,
            0,
            gate_out,
        )?;
        gemv_quantized_experts_f32(
            quantization,
            up_matrix,
            n_experts,
            selected,
            rows,
            cols,
            input,
            0,
            up_out,
        )?;
        return Ok(());
    }

    let expert_bytes = gate_matrix.len() / n_experts;
    let row_bytes = expert_bytes / rows.max(1);
    let blocks_per_row = cols / QK_K;
    let q8_stride = blocks_per_row * BLOCK_Q8_K_BYTES;
    let mut q8k = vec![0_u8; q8_stride];
    quantize_vector_q8_k_into(input, blocks_per_row, &mut q8k);
    let q8k = &q8k[..];

    // One region over both projections; 32 | rows guarantees a chunk never
    // spans a projection or expert-slot boundary.
    run_output_chunks(output, GEMV_CHUNK_ROWS, |chunk_idx, out_chunk| {
        let i0 = chunk_idx * GEMV_CHUNK_ROWS;
        let matrix = crate::numa::local_slice(if i0 < half { gate_matrix } else { up_matrix });
        let rem = i0 % half;
        let slot = rem / rows;
        let row0 = rem % rows;
        let expert = selected[slot];
        // OXK opt-in (OXIDIZE_GEMV=oxk): same chunk, ×8 kernels.
        #[cfg(feature = "oxk")]
        if gemv_mode() == GemvMode::Oxk {
            let start = expert * expert_bytes + row0 * row_bytes;
            let end = start + out_chunk.len() * row_bytes;
            oxidize_kernels::gemv_q4k_range(&matrix[start..end], blocks_per_row, q8k, out_chunk);
            return;
        }
        let mut r = 0;
        while r < out_chunk.len() {
            if r + 4 <= out_chunk.len() {
                let base = unsafe {
                    matrix
                        .as_ptr()
                        .add(expert * expert_bytes + (row0 + r) * row_bytes)
                };
                let mut quad = [0.0_f32; 4];
                // Safety: avx2 verified above; 32 | rows keeps the quad
                // inside this expert's rows.
                unsafe {
                    q4_k_q8_k_row_dot_x4_avx2(base, row_bytes, blocks_per_row, q8k, &mut quad)
                };
                out_chunk[r..r + 4].copy_from_slice(&quad);
                r += 4;
            } else {
                let row_start = expert * expert_bytes + (row0 + r) * row_bytes;
                let rowb = &matrix[row_start..row_start + row_bytes];
                out_chunk[r] = unsafe { q4_k_q8_k_row_dot(rowb, blocks_per_row, q8k) };
                r += 1;
            }
        }
    });
    Ok(())
}

/// Run `body(chunk_idx, out_chunk)` over `output` split into `chunk`-sized
/// pieces, dispatched through the persistent spin pool (decode-latency path).
fn run_output_chunks(output: &mut [f32], chunk: usize, body: impl Fn(usize, &mut [f32]) + Sync) {
    let len = output.len();
    let base = output.as_mut_ptr() as usize;
    let n_chunks = len.div_ceil(chunk);
    crate::spinpool::run_chunks(n_chunks, |ci| {
        let start = ci * chunk;
        let end = (start + chunk).min(len);
        // Safety: chunks are disjoint by construction and `output` outlives
        // the blocking run_chunks call.
        let slice =
            unsafe { std::slice::from_raw_parts_mut((base as *mut f32).add(start), end - start) };
        body(ci, slice);
    });
}

/// Per-shape GEMV profiling (`OXIDIZE_DECODE_PROFILE=1`): accumulates call
/// count, wall time, and bytes streamed per (quant, rows, cols) and prints a
/// summary at process exit. Attribution tool for decode wall time — the
/// achieved GB/s column shows which kernel/shape sits below the DRAM roof.
mod gemv_profile {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    type Table = Mutex<HashMap<(String, usize, usize), (u64, u64, u64)>>;
    static TABLE: OnceLock<Option<Table>> = OnceLock::new();

    fn table() -> Option<&'static Table> {
        TABLE
            .get_or_init(|| {
                if std::env::var("OXIDIZE_DECODE_PROFILE").is_ok_and(|v| v != "0") {
                    #[cfg(unix)]
                    unsafe {
                        libc::atexit(dump_at_exit);
                    }
                    Some(Mutex::new(HashMap::new()))
                } else {
                    None
                }
            })
            .as_ref()
    }

    #[cfg(unix)]
    extern "C" fn dump_at_exit() {
        dump();
    }

    pub fn enabled() -> bool {
        table().is_some()
    }

    pub fn record(label: String, rows: usize, cols: usize, bytes: usize, ns: u64) {
        if let Some(t) = table()
            && let Ok(mut map) = t.lock()
        {
            let e = map.entry((label, rows, cols)).or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += ns;
            e.2 += bytes as u64;
        }
    }

    pub fn dump() {
        let Some(t) = table() else { return };
        let Ok(map) = t.lock() else { return };
        let mut entries: Vec<_> = map.iter().collect();
        entries.sort_by_key(|(_, (_, ns, _))| std::cmp::Reverse(*ns));
        let total_ns: u64 = entries.iter().map(|(_, (_, ns, _))| ns).sum();
        eprintln!("gemv profile (total {:.1} ms):", total_ns as f64 / 1e6);
        for ((label, rows, cols), (count, ns, bytes)) in entries {
            eprintln!(
                "  {label:>8} {rows:>7}x{cols:<6} calls={count:<6} total={:>8.1}ms avg={:>7.1}us {:>6.1} GB/s",
                *ns as f64 / 1e6,
                *ns as f64 / 1e3 / *count as f64,
                *bytes as f64 / *ns as f64,
            );
        }
    }
}

/// Record a non-GEMV decode phase into the `OXIDIZE_DECODE_PROFILE` summary
/// (no-op when profiling is off). Returns whether profiling is enabled so
/// call sites can skip `Instant::now()` otherwise.
pub fn decode_profile_enabled() -> bool {
    gemv_profile::enabled()
}

pub fn decode_profile_record(label: &str, ns: u64) {
    gemv_profile::record(label.to_string(), 0, 0, 0, ns);
}

pub fn gemv_quantized_f32(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    if crate::gpu_dispatch::active_gpu().is_some() {
        return crate::gpu_dispatch::gemv_quantized(
            quantization,
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
        )
        .map_err(|err| GemvError::Cuda(err));
    }

    let profile_start = gemv_profile::enabled().then(std::time::Instant::now);
    let result = match quantization {
        GgufQuantizationType::Q8_0 => gemv_q8_0_f32_fused(quantized_matrix, cols, vector, output),
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M
            if cols.is_multiple_of(QK_K) && q4_k_q8_k_avx2_available() =>
        {
            gemv_q4_k_q8_k_fused(quantized_matrix, rows, cols, vector, output)
        }
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
            gemv_q4_k_f32_fused(quantized_matrix, rows, cols, vector, output)
        }
        GgufQuantizationType::Q2_K => {
            gemv_q2_k_f32_fused(quantized_matrix, rows, cols, vector, output)
        }
        GgufQuantizationType::Q6_K if cols.is_multiple_of(QK_K) && q4_k_q8_k_avx2_available() => {
            gemv_q6_k_q8_k_fused(quantized_matrix, rows, cols, vector, output)
        }
        GgufQuantizationType::Q6_K => {
            gemv_q6_k_f32_fused(quantized_matrix, rows, cols, vector, output)
        }
        GgufQuantizationType::IQ1_S if cols.is_multiple_of(QK_K) => {
            gemv_iq1_s_f32_fused(quantized_matrix, rows, cols, vector, output)
        }
        GgufQuantizationType::IQ1_M if cols.is_multiple_of(QK_K) => {
            gemv_iq1_m_f32_fused(quantized_matrix, rows, cols, vector, output)
        }
        GgufQuantizationType::NVFP4 if cols.is_multiple_of(QK_NVFP4) => {
            gemv_nvfp4_f32_fused(quantized_matrix, rows, cols, vector, output)
        }
        _ => Err(GemvError::UnsupportedQuantizationType { quantization }),
    };
    if let Some(start) = profile_start {
        gemv_profile::record(
            format!("{quantization:?}"),
            rows,
            cols,
            quantized_matrix.len(),
            start.elapsed().as_nanos() as u64,
        );
    }
    result
}

/// One matrix of a fused multi-GEMV region (see [`gemv_quantized_multi_f32`]).
pub struct GemvJob<'a> {
    pub quantization: GgufQuantizationType,
    pub matrix: &'a [u8],
    pub rows: usize,
    pub output: &'a mut [f32],
}

/// Run several quantized GEMVs that share one input vector as a SINGLE flat
/// parallel region. Token decode previously overlapped q/k/v and gate/up with
/// `rayon::join`, but nested parallel regions steal work from each other and
/// interleave the weight streams of different matrices on the same cores
/// (measured 19-21 GB/s vs 32+ GB/s for the same shape dispatched alone); with
/// the spin pool the losing join arm ran entirely serial. One flat region
/// keeps every worker on one contiguous weight range and quantizes the shared
/// input to Q8_K once.
///
/// Row results are bit-identical to [`gemv_quantized_f32`]: the same row-dot
/// kernels run in the same per-row order. Jobs whose quantization lacks the
/// integer Q8_K fast path on this CPU fall back to sequential
/// [`gemv_quantized_f32`] calls.
pub fn gemv_quantized_multi_f32(
    jobs: &mut [GemvJob<'_>],
    cols: usize,
    vector: &[f32],
) -> Result<(), GemvError> {
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    let fast = cols.is_multiple_of(QK_K)
        && q4_k_q8_k_avx2_available()
        && jobs.iter().all(|job| {
            matches!(
                job.quantization,
                GgufQuantizationType::Q4_K_S
                    | GgufQuantizationType::Q4_K_M
                    | GgufQuantizationType::Q6_K
            )
        });
    if !fast {
        for job in jobs.iter_mut() {
            gemv_quantized_f32(
                job.quantization,
                job.matrix,
                job.rows,
                cols,
                vector,
                job.output,
            )?;
        }
        return Ok(());
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unreachable!("fast multi-GEMV requires the x86 Q8_K kernels");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let blocks_per_row = cols / QK_K;
        for job in jobs.iter() {
            let block_size = match job.quantization {
                GgufQuantizationType::Q6_K => BLOCK_Q6_K_SIZE,
                _ => BLOCK_Q4_K_SIZE,
            };
            let expected = job.rows * blocks_per_row * block_size;
            if job.matrix.len() != expected {
                return Err(GemvError::InvalidMatrixLength {
                    expected,
                    actual: job.matrix.len(),
                });
            }
            if job.output.len() != job.rows {
                return Err(GemvError::InvalidOutputLength {
                    expected: job.rows,
                    actual: job.output.len(),
                });
            }
        }

        let profile_start = gemv_profile::enabled().then(std::time::Instant::now);
        let mut q8k = vec![0_u8; blocks_per_row * BLOCK_Q8_K_BYTES];
        quantize_vector_q8_k_into(vector, blocks_per_row, &mut q8k);

        // Flatten jobs into row chunks; chunk_starts[i] is the first global
        // chunk index of job i. Chunk sizes are byte-weighted per job (Q6_K
        // rows are 1.46x heavier than Q4_K) so the static block partition
        // over chunk indices stays balanced in BYTES when quantizations mix
        // within one region (q in Q4_K with k/v in Q6_K measurably skewed the
        // tail participants otherwise).
        let chunk_bytes_target = GEMV_CHUNK_ROWS * blocks_per_row * BLOCK_Q4_K_SIZE;
        let mut chunk_rows = Vec::with_capacity(jobs.len());
        let mut chunk_starts = Vec::with_capacity(jobs.len() + 1);
        let mut total_chunks = 0_usize;
        for job in jobs.iter() {
            let row_bytes = job.matrix.len() / job.rows.max(1);
            let rows_per_chunk = (chunk_bytes_target / row_bytes.max(1))
                .next_multiple_of(4)
                .clamp(4, GEMV_CHUNK_ROWS);
            chunk_starts.push(total_chunks);
            chunk_rows.push(rows_per_chunk);
            total_chunks += job.rows.div_ceil(rows_per_chunk);
        }
        chunk_starts.push(total_chunks);

        struct JobRef {
            quantization: GgufQuantizationType,
            matrix_ptr: usize,
            matrix_len: usize,
            rows: usize,
            out_ptr: usize,
        }
        let refs: Vec<JobRef> = jobs
            .iter_mut()
            .map(|job| JobRef {
                quantization: job.quantization,
                matrix_ptr: job.matrix.as_ptr() as usize,
                matrix_len: job.matrix.len(),
                rows: job.rows,
                out_ptr: job.output.as_mut_ptr() as usize,
            })
            .collect();
        let use_x4 = !q4_k_q8_k_vnni_available();
        let q8k = &q8k[..];
        let total_bytes: usize = refs.iter().map(|r| r.matrix_len).sum();
        let total_rows: usize = refs.iter().map(|r| r.rows).sum();

        crate::spinpool::run_chunks(total_chunks, |ci| {
            let job_idx = chunk_starts.partition_point(|&s| s <= ci) - 1;
            let job = &refs[job_idx];
            let job_chunk_rows = chunk_rows[job_idx];
            let row0 = (ci - chunk_starts[job_idx]) * job_chunk_rows;
            let nrows = job_chunk_rows.min(job.rows - row0);
            // Safety: chunks partition each job's rows disjointly, and the
            // matrices/outputs are caller borrows that outlive this region.
            let matrix =
                unsafe { std::slice::from_raw_parts(job.matrix_ptr as *const u8, job.matrix_len) };
            let matrix = crate::numa::local_slice(matrix);
            let out = unsafe {
                std::slice::from_raw_parts_mut((job.out_ptr as *mut f32).add(row0), nrows)
            };
            match job.quantization {
                GgufQuantizationType::Q6_K => {
                    let row_bytes = blocks_per_row * BLOCK_Q6_K_SIZE;
                    let mut r = 0;
                    while r < out.len() {
                        if use_x4 && r + 4 <= out.len() {
                            let base = unsafe { matrix.as_ptr().add((row0 + r) * row_bytes) };
                            let mut quad = [0.0_f32; 4];
                            // Safety: avx2+fma verified by the `fast` gate.
                            unsafe {
                                q6_k_q8_k_row_dot_x4_avx2(
                                    base,
                                    row_bytes,
                                    blocks_per_row,
                                    q8k,
                                    &mut quad,
                                )
                            };
                            out[r..r + 4].copy_from_slice(&quad);
                            r += 4;
                        } else {
                            let start = (row0 + r) * row_bytes;
                            let row = &matrix[start..start + row_bytes];
                            out[r] = unsafe { q6_k_q8_k_row_dot_avx2(row, blocks_per_row, q8k) };
                            r += 1;
                        }
                    }
                }
                _ => {
                    let row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;
                    #[cfg(feature = "oxk")]
                    let use_oxk = gemv_mode() == GemvMode::Oxk;
                    #[cfg(not(feature = "oxk"))]
                    let use_oxk = false;
                    if use_oxk {
                        #[cfg(feature = "oxk")]
                        {
                            let start = row0 * row_bytes;
                            oxidize_kernels::gemv_q4k_range(
                                &matrix[start..start + out.len() * row_bytes],
                                blocks_per_row,
                                q8k,
                                out,
                            );
                        }
                    } else {
                        let mut r = 0;
                        while r < out.len() {
                            if use_x4 && r + 4 <= out.len() {
                                let base = unsafe { matrix.as_ptr().add((row0 + r) * row_bytes) };
                                let mut quad = [0.0_f32; 4];
                                // Safety: avx2+fma verified by the `fast` gate.
                                unsafe {
                                    q4_k_q8_k_row_dot_x4_avx2(
                                        base,
                                        row_bytes,
                                        blocks_per_row,
                                        q8k,
                                        &mut quad,
                                    )
                                };
                                out[r..r + 4].copy_from_slice(&quad);
                                r += 4;
                            } else {
                                let start = (row0 + r) * row_bytes;
                                let row = &matrix[start..start + row_bytes];
                                out[r] = unsafe { q4_k_q8_k_row_dot(row, blocks_per_row, q8k) };
                                r += 1;
                            }
                        }
                    }
                }
            }
        });
        if let Some(start) = profile_start {
            gemv_profile::record(
                format!("fused{}", refs.len()),
                total_rows,
                cols,
                total_bytes,
                start.elapsed().as_nanos() as u64,
            );
        }
        Ok(())
    }
}

#[inline]
fn q4_k_q8_k_avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn q4_k_q8_k_vnni_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Which Q4_K GEMV implementation services the AVX2 decode hot path.
/// Selected once from `OXIDIZE_GEMV` (see the OXK migration plan): `auto`
/// (default) uses OXK when the `oxk` feature is compiled and this CPU supports
/// the kernel ISA, `legacy` keeps the tensor.rs intrinsics untouched, `oxk`
/// routes contiguous row ranges to the `oxidize-kernels` crate, and `shadow`
/// runs both and compares (dev/bench only). Without the `oxk` cargo feature
/// every value resolves to `Legacy`.
#[cfg_attr(not(feature = "oxk"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GemvMode {
    Legacy,
    #[cfg(feature = "oxk")]
    Oxk,
    #[cfg(feature = "oxk")]
    Shadow,
}

#[cfg_attr(not(feature = "oxk"), allow(dead_code))]
fn gemv_mode() -> GemvMode {
    static MODE: std::sync::OnceLock<GemvMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("OXIDIZE_GEMV").as_deref() {
        #[cfg(feature = "oxk")]
        Ok("oxk") => GemvMode::Oxk,
        #[cfg(feature = "oxk")]
        Ok("shadow") => GemvMode::Shadow,
        Ok("auto") | Ok("") | Err(_) => {
            #[cfg(feature = "oxk")]
            {
                if oxidize_kernels::oxk_avx2_available() {
                    GemvMode::Oxk
                } else {
                    GemvMode::Legacy
                }
            }
            #[cfg(not(feature = "oxk"))]
            {
                GemvMode::Legacy
            }
        }
        Ok("legacy") => GemvMode::Legacy,
        Ok(other) => {
            eprintln!(
                "OXIDIZE_GEMV={other} not available in this build (unknown value or \
                 'oxk' feature not compiled); falling back to legacy"
            );
            GemvMode::Legacy
        }
    })
}

/// Shadow mode: run the legacy range into `out`, the OXK range into a scratch
/// buffer, compare, and accumulate per-implementation wall time. Mismatches
/// beyond 1e-4 relative error and periodic timing summaries go to stderr.
#[cfg(feature = "oxk")]
fn shadow_q4k_range(
    rows: &[u8],
    blocks_per_row: usize,
    q8k: &[u8],
    out: &mut [f32],
    legacy: impl FnOnce(&mut [f32]),
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static LEGACY_NS: AtomicU64 = AtomicU64::new(0);
    static OXK_NS: AtomicU64 = AtomicU64::new(0);
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static MISMATCHES: AtomicU64 = AtomicU64::new(0);

    let t0 = std::time::Instant::now();
    legacy(out);
    let t1 = std::time::Instant::now();
    let mut scratch = vec![0.0_f32; out.len()];
    oxidize_kernels::gemv_q4k_range(rows, blocks_per_row, q8k, &mut scratch);
    let t2 = std::time::Instant::now();

    for (i, (l, o)) in out.iter().zip(scratch.iter()).enumerate() {
        let rel = (l - o).abs() / l.abs().max(1e-6);
        if rel > 1e-4 && MISMATCHES.fetch_add(1, Ordering::Relaxed) < 16 {
            eprintln!("[oxk-shadow] mismatch row {i}: legacy={l} oxk={o} rel={rel:.3e}");
        }
    }
    let legacy_ns = LEGACY_NS.fetch_add(t1.duration_since(t0).as_nanos() as u64, Ordering::Relaxed);
    let oxk_ns = OXK_NS.fetch_add(t2.duration_since(t1).as_nanos() as u64, Ordering::Relaxed);
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if calls.is_multiple_of(65_536) {
        eprintln!(
            "[oxk-shadow] {} ranges: legacy {:.3}s oxk {:.3}s (oxk = {:.1}% of legacy), mismatched rows {}",
            calls,
            legacy_ns as f64 / 1e9,
            oxk_ns as f64 / 1e9,
            oxk_ns as f64 / legacy_ns.max(1) as f64 * 100.0,
            MISMATCHES.load(Ordering::Relaxed),
        );
    }
}

/// Dispatch one Q4_K × Q8_K row dot to the best available kernel. VNNI is
/// preferred; AVX2 is the fallback. The caller must have verified
/// [`q4_k_q8_k_avx2_available`] (VNNI implies AVX2-class availability here).
#[inline]
unsafe fn q4_k_q8_k_row_dot(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if q4_k_q8_k_vnni_available() {
            return unsafe { q4_k_q8_k_row_dot_vnni(row, blocks_per_row, q8k) };
        }
    }
    unsafe { q4_k_q8_k_row_dot_avx2(row, blocks_per_row, q8k) }
}

/// Q6_K x Q8_K fused GEMV: quantizes the input once to Q8_K, then runs the
/// integer Q6_K kernel per row (4-row chunks share the input loads). Same
/// structure as [`gemv_q4_k_q8_k_fused`].
fn gemv_q6_k_q8_k_fused(
    weights: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    debug_assert!(cols.is_multiple_of(QK_K));
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows
        .saturating_mul(blocks_per_row)
        .saturating_mul(BLOCK_Q6_K_SIZE);
    if weights.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: weights.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }
    let mut q8k = vec![0_u8; blocks_per_row * BLOCK_Q8_K_BYTES];
    quantize_vector_q8_k_into(vector, blocks_per_row, &mut q8k);
    let row_bytes = blocks_per_row * BLOCK_Q6_K_SIZE;

    let run_range = |out_range: &mut [f32], row0: usize| {
        let weights = crate::numa::local_slice(weights);
        let mut r = 0;
        while r < out_range.len() {
            if r + 4 <= out_range.len() && row0 + r + 4 <= rows {
                let base = unsafe { weights.as_ptr().add((row0 + r) * row_bytes) };
                let mut quad = [0.0_f32; 4];
                // Safety: avx2 verified before dispatch; rows in range.
                unsafe {
                    q6_k_q8_k_row_dot_x4_avx2(base, row_bytes, blocks_per_row, &q8k, &mut quad)
                };
                out_range[r..r + 4].copy_from_slice(&quad);
                r += 4;
            } else {
                let row_start = (row0 + r) * row_bytes;
                let row = &weights[row_start..row_start + row_bytes];
                // Safety: avx2 verified before dispatch.
                out_range[r] = unsafe { q6_k_q8_k_row_dot_avx2(row, blocks_per_row, &q8k) };
                r += 1;
            }
        }
    };
    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        run_output_chunks(output, GEMV_CHUNK_ROWS, |chunk_idx, out_chunk| {
            run_range(out_chunk, chunk_idx * GEMV_CHUNK_ROWS)
        });
    } else {
        run_range(output, 0);
    }
    Ok(())
}

/// Q4_K × Q8_K fused GEMV. Quantizes the input vector to Q8_K (int8 + scale
/// per 256-element block, plus per-16 sums for the min correction) once, then
/// computes each output row using AVX2 `maddubs`/`madd` integer dot products
/// instead of fp32 FMA. This matches llama.cpp's `ggml_vec_dot_q4_K_q8_K`
/// strategy and is the main reason llama.cpp decode is fast on CPUs without
/// AVX-512.
fn gemv_q4_k_q8_k_fused(
    weights: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    debug_assert!(cols.is_multiple_of(QK_K));
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q4_K_SIZE;
    if weights.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: weights.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    // Quantize the input vector once into `blocks_per_row` Q8_K blocks of size
    // `BLOCK_Q8_K_BYTES` each. Layout matches llama.cpp's block_q8_K: f32 d,
    // then 256 int8 quants, then 16 int16 bsums.
    let mut q8k = vec![0_u8; blocks_per_row * BLOCK_Q8_K_BYTES];
    quantize_vector_q8_k_into(vector, blocks_per_row, &mut q8k);

    let row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;
    let compute_row = |row_idx: usize| -> f32 {
        // Prefetch the next row into L1 cache while the CPU processes this one.
        // The hardware prefetcher tracks sequential access but the 1440-byte stride
        // between rows is non-power-of-2 and may evade stride detection; explicit
        // prefetch hides DRAM latency for large matrices (measured ~5% benefit).
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if row_idx + 1 < rows {
            let next_row_ptr = weights
                .as_ptr()
                .wrapping_add((row_idx + 1) * row_bytes)
                .cast::<i8>();
            unsafe { _mm_prefetch::<{ _MM_HINT_T1 }>(next_row_ptr) };
        }
        let row_start = row_idx * row_bytes;
        let row = &weights[row_start..row_start + row_bytes];
        // Safety: q4_k_q8_k_avx2_available() was checked before dispatch;
        // dispatcher picks the VNNI kernel when the runtime supports it.
        unsafe { q4_k_q8_k_row_dot(row, blocks_per_row, &q8k) }
    };

    // 4-row custom kernel (AVX2 machines without VNNI): shares the Q8_K input
    // across 4 weight rows and keeps 4 independent accumulator chains in
    // flight so DRAM latency overlaps across row streams.
    let use_x4 =
        cfg!(any(target_arch = "x86", target_arch = "x86_64")) && !q4_k_q8_k_vnni_available();
    let run_range = |out_range: &mut [f32], row0: usize| {
        let weights = crate::numa::local_slice(weights);
        let legacy_range = |out_range: &mut [f32]| {
            let mut r = 0;
            while r < out_range.len() {
                if use_x4 && r + 4 <= out_range.len() && row0 + r + 4 <= rows {
                    let base = unsafe { weights.as_ptr().add((row0 + r) * row_bytes) };
                    let mut quad = [0.0_f32; 4];
                    // Safety: avx2+fma verified before dispatch; rows are in range.
                    unsafe {
                        q4_k_q8_k_row_dot_x4_avx2(base, row_bytes, blocks_per_row, &q8k, &mut quad)
                    };
                    out_range[r..r + 4].copy_from_slice(&quad);
                    r += 4;
                } else {
                    out_range[r] = compute_row(row0 + r);
                    r += 1;
                }
            }
        };
        // OXK dispatch choke point (single switch, OXIDIZE_GEMV): threading,
        // NUMA translation and Q8_K quantization above are shared by all modes.
        #[cfg(feature = "oxk")]
        {
            let start = row0 * row_bytes;
            let end = start + out_range.len() * row_bytes;
            match gemv_mode() {
                GemvMode::Oxk => {
                    oxidize_kernels::gemv_q4k_range(
                        &weights[start..end],
                        blocks_per_row,
                        &q8k,
                        out_range,
                    );
                    return;
                }
                GemvMode::Shadow => {
                    shadow_q4k_range(
                        &weights[start..end],
                        blocks_per_row,
                        &q8k,
                        out_range,
                        legacy_range,
                    );
                    return;
                }
                GemvMode::Legacy => {}
            }
        }
        legacy_range(out_range);
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        run_output_chunks(output, GEMV_CHUNK_ROWS, |chunk_idx, out_chunk| {
            run_range(out_chunk, chunk_idx * GEMV_CHUNK_ROWS)
        });
    } else {
        run_range(output, 0);
    }
    Ok(())
}

/// Batched Q4_K × Q8_K GEMM for prompt processing. Each input row is quantized
/// to Q8_K once, then rows are multiplied in small token chunks so packed Q4_K
/// weight bytes are reused across multiple prompt tokens.
fn gemm_q4_k_q8_k_fused(
    weights: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), GemvError> {
    debug_assert!(cols.is_multiple_of(QK_K));
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows
        .saturating_mul(blocks_per_row)
        .saturating_mul(BLOCK_Q4_K_SIZE);
    if weights.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: weights.len(),
        });
    }
    let expected_inputs = batch.saturating_mul(cols);
    if inputs.len() != expected_inputs {
        return Err(GemvError::InvalidVectorLength {
            expected: expected_inputs,
            actual: inputs.len(),
        });
    }
    let expected_outputs = batch.saturating_mul(rows);
    if outputs.len() != expected_outputs {
        return Err(GemvError::InvalidOutputLength {
            expected: expected_outputs,
            actual: outputs.len(),
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe {
                gemm_q4_k_q8_k_fused_avx2(
                    weights,
                    rows,
                    cols,
                    inputs,
                    outputs,
                    batch,
                    blocks_per_row,
                )
            };
        }
    }

    gemm_q4_k_decode_once(weights, rows, cols, inputs, outputs, batch)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn gemm_q4_k_q8_k_fused_avx2(
    weights: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
    blocks_per_row: usize,
) -> Result<(), GemvError> {
    let q8_stride = blocks_per_row * BLOCK_Q8_K_BYTES;
    if q8_stride == 0 {
        outputs.fill(0.0);
        return Ok(());
    }

    let mut q8_panel = vec![0_u8; batch.saturating_mul(q8_stride)];
    q8_panel
        .par_chunks_mut(q8_stride)
        .enumerate()
        .for_each(|(token, q8)| {
            let input = &inputs[token * cols..(token + 1) * cols];
            quantize_vector_q8_k_into(input, blocks_per_row, q8);
        });
    let q8_panel_slice = &q8_panel[..];

    const ROW_CHUNK: usize = 16;
    let row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;
    let mut row_major = vec![0.0_f32; rows.saturating_mul(batch)];

    row_major
        .par_chunks_mut(ROW_CHUNK * batch)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx * ROW_CHUNK;
            let end = (start + ROW_CHUNK).min(rows);
            for row_idx in start..end {
                let row_start = row_idx * row_bytes;
                let row = &weights[row_start..row_start + row_bytes];
                let local = row_idx - start;
                let partial = &mut chunk[local * batch..(local + 1) * batch];
                partial.fill(0.0);
                // Single kernel call processes all `batch` tokens; decodes q4
                // weight nibbles once per block.
                unsafe {
                    q4_k_q8_k_row_dot_chunk_avx2(
                        row,
                        blocks_per_row,
                        q8_panel_slice,
                        q8_stride,
                        0,
                        batch,
                        partial,
                    );
                }
            }
        });

    for token in 0..batch {
        let out = &mut outputs[token * rows..(token + 1) * rows];
        for row in 0..rows {
            out[row] = row_major[row * batch + token];
        }
    }
    let _ = q8_panel;
    Ok(())
}

/// Per-block Q8_K layout (matches llama.cpp's `block_q8_K`):
///   bytes 0..4   : f32 d (1/iscale)
///   bytes 4..260 : 256 int8 quants
///   bytes 260..292 : 16 int16 bsums (sum of int8 quants in groups of 16)
const BLOCK_Q8_K_BYTES: usize = 4 + 256 + 32;

/// Quantize `vector` (length `n_blocks * 256`) into `n_blocks` Q8_K blocks.
pub(crate) fn quantize_vector_q8_k_into(vector: &[f32], n_blocks: usize, out: &mut [u8]) {
    debug_assert_eq!(vector.len(), n_blocks * QK_K);
    debug_assert_eq!(out.len(), n_blocks * BLOCK_Q8_K_BYTES);
    for (b, block_in) in vector.chunks_exact(QK_K).enumerate().take(n_blocks) {
        let block_out = &mut out[b * BLOCK_Q8_K_BYTES..(b + 1) * BLOCK_Q8_K_BYTES];
        quantize_block_q8_k_scalar(block_in, block_out);
    }
}

fn quantize_block_q8_k_scalar(block_in: &[f32], block_out: &mut [u8]) {
    debug_assert_eq!(block_in.len(), QK_K);
    debug_assert_eq!(block_out.len(), BLOCK_Q8_K_BYTES);
    let mut amax = 0.0_f32;
    let mut max = 0.0_f32;
    for &v in block_in {
        let av = v.abs();
        if av > amax {
            amax = av;
            max = v;
        }
    }
    if amax == 0.0 {
        // d = 0, all qs = 0, bsums = 0.
        block_out[..4].copy_from_slice(&0.0_f32.to_le_bytes());
        for byte in &mut block_out[4..] {
            *byte = 0;
        }
        return;
    }
    // iscale = -128 / max (sign-preserving to keep symmetry with [-128, 127])
    let iscale = -128.0_f32 / max;
    let d = 1.0_f32 / iscale;
    block_out[..4].copy_from_slice(&d.to_le_bytes());
    let qs_off = 4;
    for (i, &v) in block_in.iter().enumerate() {
        let scaled = iscale * v;
        let q = scaled.round() as i32;
        let q = q.clamp(-128, 127) as i8;
        block_out[qs_off + i] = q as u8;
    }
    // bsums: 16 int16 sums, one per 16-element group.
    let bsums_off = qs_off + QK_K;
    for g in 0..(QK_K / 16) {
        let mut sum: i32 = 0;
        for i in 0..16 {
            sum += (block_out[qs_off + g * 16 + i] as i8) as i32;
        }
        let sum16 = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let lo = (sum16 as u16) as u8;
        let hi = ((sum16 as u16) >> 8) as u8;
        block_out[bsums_off + g * 2] = lo;
        block_out[bsums_off + g * 2 + 1] = hi;
    }
}

/// Per-row Q4_K × Q8_K dot product using AVX2 integer multiply-adds.
/// Returns the f32 dot product for one output row across all blocks in the row.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q4_k_q8_k_row_dot_avx2(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    let mask = _mm256_set1_epi8(0x0f);
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w_ptr = row.as_ptr().wrapping_add(block_idx * BLOCK_Q4_K_SIZE);
        let q8_ptr = q8k.as_ptr().wrapping_add(block_idx * BLOCK_Q8_K_BYTES);

        // Prefetch the weight stream ~4 blocks (576B) ahead: rows stream from
        // DRAM (often the remote NUMA node here) and the OOO window alone does
        // not hide that latency. 3 lines cover the 144B consumed per iteration.
        // Prefetch hints never fault, so running past the row end is safe.
        let ahead = w_ptr.wrapping_add(4 * BLOCK_Q4_K_SIZE).cast::<i8>();
        _mm_prefetch::<{ _MM_HINT_T0 }>(ahead);
        _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.wrapping_add(64));
        _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.wrapping_add(128));

        let d_w = f16_le_to_f32([unsafe { *w_ptr }, unsafe { *w_ptr.add(1) }]);
        let dmin_w = f16_le_to_f32([unsafe { *w_ptr.add(2) }, unsafe { *w_ptr.add(3) }]);
        let d_q8 = f32::from_le_bytes([
            unsafe { *q8_ptr },
            unsafe { *q8_ptr.add(1) },
            unsafe { *q8_ptr.add(2) },
            unsafe { *q8_ptr.add(3) },
        ]);
        let scales = unsafe { std::slice::from_raw_parts(w_ptr.add(4), 12) };
        let qs = unsafe { w_ptr.add(16) };
        let q8 = unsafe { q8_ptr.add(4) };
        let bsums = unsafe { q8_ptr.add(4 + QK_K) };

        // Single vector accumulator across all 8 sub-groups → 1 hsum per block.
        let mut vec_pos = _mm256_setzero_si256();
        let mut min_acc: i32 = 0;
        for gp in 0..4 {
            let g1 = gp * 2;
            let g2 = g1 + 1;
            let (s1, ms1) = get_scale_min_k4(g1, scales);
            let (s2, ms2) = get_scale_min_k4(g2, scales);
            let packed = unsafe { _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i) };
            let q4_low = _mm256_and_si256(packed, mask);
            let q4_high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            let q8_low = unsafe { _mm256_loadu_si256(q8.add(g1 * 32) as *const __m256i) };
            let q8_high = unsafe { _mm256_loadu_si256(q8.add(g2 * 32) as *const __m256i) };
            let p16_low = _mm256_maddubs_epi16(q4_low, q8_low);
            let p16_high = _mm256_maddubs_epi16(q4_high, q8_high);
            // madd(p16, set1_epi16(s)) == s * (p0 + p1) per i32 lane — identical
            // to madd(p16, ones) * s, but avoids the slow mullo_epi32 (10c lat).
            // No overflow: |p16| <= 2*15*127 = 3810, s <= 63 -> 240_030 << i32::MAX.
            let p32_low = _mm256_madd_epi16(p16_low, _mm256_set1_epi16(s1 as i16));
            let p32_high = _mm256_madd_epi16(p16_high, _mm256_set1_epi16(s2 as i16));
            vec_pos = _mm256_add_epi32(vec_pos, _mm256_add_epi32(p32_low, p32_high));

            let bs1 = unsafe { read_q8_k_bsum(bsums, g1 * 2) } as i32
                + unsafe { read_q8_k_bsum(bsums, g1 * 2 + 1) } as i32;
            let bs2 = unsafe { read_q8_k_bsum(bsums, g2 * 2) } as i32
                + unsafe { read_q8_k_bsum(bsums, g2 * 2 + 1) } as i32;
            min_acc += ms1 as i32 * bs1;
            min_acc += ms2 as i32 * bs2;
        }
        let pos_acc = unsafe { hsum_i32_avx2(vec_pos) };
        acc += d_w * d_q8 * pos_acc as f32 - dmin_w * d_q8 * min_acc as f32;
    }
    acc
}

/// Dot 4 consecutive weight rows against one shared Q8_K vector.
///
/// Per-row math is bit-identical to [`q4_k_q8_k_row_dot_avx2`] (same op
/// sequence and accumulation order); the win is structural: the Q8_K input
/// vectors and bsum pair-sums are loaded/computed once per block and reused by
/// all 4 rows, and the 4 scalar accumulators form independent dependency
/// chains so the out-of-order core can overlap DRAM (often remote-NUMA)
/// latency across rows instead of stalling on one row's stream.
///
/// # Safety
/// `rows_base` must point to 4 rows of `blocks_per_row` Q4_K blocks spaced
/// `row_bytes` apart; `q8k` must hold `blocks_per_row` Q8_K blocks. Caller
/// must have verified AVX2+FMA support.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q4_k_q8_k_row_dot_x4_avx2(
    rows_base: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    q8k: &[u8],
    out: &mut [f32; 4],
) {
    let mask = _mm256_set1_epi8(0x0f);
    let mut acc = [0.0_f32; 4];
    for block_idx in 0..blocks_per_row {
        let q8_ptr = q8k.as_ptr().add(block_idx * BLOCK_Q8_K_BYTES);
        let d_q8 = f32::from_le_bytes([*q8_ptr, *q8_ptr.add(1), *q8_ptr.add(2), *q8_ptr.add(3)]);
        let q8 = q8_ptr.add(4);
        let bsums = q8_ptr.add(4 + QK_K);

        // Shared across all 4 rows: the 8 q8 sub-group vectors and the
        // per-group-pair bsum sums (these depend only on the input vector).
        let q8v = [
            _mm256_loadu_si256(q8 as *const __m256i),
            _mm256_loadu_si256(q8.add(32) as *const __m256i),
            _mm256_loadu_si256(q8.add(64) as *const __m256i),
            _mm256_loadu_si256(q8.add(96) as *const __m256i),
            _mm256_loadu_si256(q8.add(128) as *const __m256i),
            _mm256_loadu_si256(q8.add(160) as *const __m256i),
            _mm256_loadu_si256(q8.add(192) as *const __m256i),
            _mm256_loadu_si256(q8.add(224) as *const __m256i),
        ];
        let mut bs = [0_i32; 8];
        for (g, b) in bs.iter_mut().enumerate() {
            *b = read_q8_k_bsum(bsums, g * 2) as i32 + read_q8_k_bsum(bsums, g * 2 + 1) as i32;
        }

        for (r, acc_r) in acc.iter_mut().enumerate() {
            let w_ptr = rows_base.add(r * row_bytes + block_idx * BLOCK_Q4_K_SIZE);
            // Same prefetch policy as the single-row kernel, per stream.
            let ahead = w_ptr.add(4 * BLOCK_Q4_K_SIZE).cast::<i8>();
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead);
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.add(64));
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.add(128));
            // For SHORT rows also sweep the NEXT quad's row r into L2, one
            // quad-time ahead: 10-block rows (1.4KB) restart the hardware
            // prefetcher every 22 cache lines, costing ~10% of DRAM bandwidth
            // on 2560-column matrices. Advancing one block per iteration, the
            // pointer covers the whole next row by quad end. Long rows keep
            // the prefetcher locked on their own — the extra reach only
            // pollutes L2 there.
            if blocks_per_row <= 16 {
                let next_quad = w_ptr.add(4 * row_bytes).cast::<i8>();
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad);
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad.add(64));
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad.add(128));
            } else {
                // Long rows: a second, deeper in-row sweep (T1, 16 blocks =
                // 2.3KB ahead) — the 576B T0 distance alone leaves the stream
                // ~8% under the short-row shapes once those got their sweep.
                let far = w_ptr.add(16 * BLOCK_Q4_K_SIZE).cast::<i8>();
                _mm_prefetch::<{ _MM_HINT_T1 }>(far);
                _mm_prefetch::<{ _MM_HINT_T1 }>(far.add(64));
                _mm_prefetch::<{ _MM_HINT_T1 }>(far.add(128));
            }

            let d_w = f16_le_to_f32([*w_ptr, *w_ptr.add(1)]);
            let dmin_w = f16_le_to_f32([*w_ptr.add(2), *w_ptr.add(3)]);
            let scales = std::slice::from_raw_parts(w_ptr.add(4), 12);
            let qs = w_ptr.add(16);

            let mut vec_pos = _mm256_setzero_si256();
            let mut min_acc: i32 = 0;
            for gp in 0..4 {
                let g1 = gp * 2;
                let g2 = g1 + 1;
                let (s1, ms1) = get_scale_min_k4(g1, scales);
                let (s2, ms2) = get_scale_min_k4(g2, scales);
                let packed = _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i);
                let q4_low = _mm256_and_si256(packed, mask);
                let q4_high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
                let p16_low = _mm256_maddubs_epi16(q4_low, q8v[g1]);
                let p16_high = _mm256_maddubs_epi16(q4_high, q8v[g2]);
                let p32_low = _mm256_madd_epi16(p16_low, _mm256_set1_epi16(s1 as i16));
                let p32_high = _mm256_madd_epi16(p16_high, _mm256_set1_epi16(s2 as i16));
                vec_pos = _mm256_add_epi32(vec_pos, _mm256_add_epi32(p32_low, p32_high));
                min_acc += ms1 as i32 * bs[g1];
                min_acc += ms2 as i32 * bs[g2];
            }
            let pos_acc = hsum_i32_avx2(vec_pos);
            *acc_r += d_w * d_q8 * pos_acc as f32 - dmin_w * d_q8 * min_acc as f32;
        }
    }
    *out = acc;
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn q4_k_q8_k_row_dot_x4_avx2(
    _rows_base: *const u8,
    _row_bytes: usize,
    _blocks_per_row: usize,
    _q8k: &[u8],
    _out: &mut [f32; 4],
) {
    unreachable!("x4 kernel is gated on x86 availability at call sites")
}

/// Integer Q6_K x Q8_K row dot (llama.cpp-style). Decodes 6-bit weights to
/// unsigned 0..63, runs `maddubs`/`madd` integer dot products against the
/// pre-quantized Q8_K input, and removes the implicit -32 offset analytically
/// via the Q8_K per-16 bsums: sum((q6u-32)*q8) = maddubs-sum - 32*bsum. This
/// replaces the f32 decode+FMA Q6_K kernel on the GEMV hot paths (~5x fewer
/// ops per byte). No overflow: maddubs pair <= 2*63*127 = 16_002 (i16),
/// madd with |scale| <= 127 -> ~4.1M per lane-pair (i32).
///
/// # Safety
/// Caller must verify AVX2; `row` holds `blocks_per_row` Q6_K blocks and
/// `q8k` the matching Q8_K blocks.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q6_k_q8_k_row_dot_avx2(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    let mask_low = _mm256_set1_epi8(0x0f);
    let mask_high = _mm256_set1_epi8(0x03);
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w_ptr = row.as_ptr().add(block_idx * BLOCK_Q6_K_SIZE);
        let q8_ptr = q8k.as_ptr().add(block_idx * BLOCK_Q8_K_BYTES);
        let d_q8 = f32::from_le_bytes([*q8_ptr, *q8_ptr.add(1), *q8_ptr.add(2), *q8_ptr.add(3)]);
        let q8 = q8_ptr.add(4);
        let bsums = q8_ptr.add(4 + QK_K);
        let d = f16_le_to_f32([*w_ptr.add(208), *w_ptr.add(209)]);
        let ql = w_ptr;
        let qh = w_ptr.add(128);
        let sc = std::slice::from_raw_parts(w_ptr.add(192) as *const i8, 16);

        let mut vec_pos = _mm256_setzero_si256();
        let mut min_acc: i32 = 0;
        for half in 0..2 {
            let s_base = half * 8;
            let v_base = half * 128;
            let ql_lo = _mm256_loadu_si256(ql.add(half * 64) as *const __m256i);
            let ql_hi = _mm256_loadu_si256(ql.add(half * 64 + 32) as *const __m256i);
            let qh_v = _mm256_loadu_si256(qh.add(half * 32) as *const __m256i);

            // Four 32-value groups per half; mapping mirrors q6_k_row_dot_avx2.
            let q1 = _mm256_or_si256(
                _mm256_and_si256(ql_lo, mask_low),
                _mm256_slli_epi16(_mm256_and_si256(qh_v, mask_high), 4),
            );
            let q2 = _mm256_or_si256(
                _mm256_and_si256(ql_hi, mask_low),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 2), mask_high), 4),
            );
            let q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), mask_low),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 4), mask_high), 4),
            );
            let q4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), mask_low),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 6), mask_high), 4),
            );

            for (g, qv) in [q1, q2, q3, q4].into_iter().enumerate() {
                let sa = sc[s_base + g * 2] as i16;
                let sb = sc[s_base + g * 2 + 1] as i16;
                let q8v = _mm256_loadu_si256(q8.add(v_base + g * 32) as *const __m256i);
                let p16 = _mm256_maddubs_epi16(qv, q8v);
                let scale_pair = _mm256_set_m128i(_mm_set1_epi16(sb), _mm_set1_epi16(sa));
                vec_pos = _mm256_add_epi32(vec_pos, _mm256_madd_epi16(p16, scale_pair));
                let g0 = half * 8 + g * 2;
                min_acc += sa as i32 * read_q8_k_bsum(bsums, g0) as i32;
                min_acc += sb as i32 * read_q8_k_bsum(bsums, g0 + 1) as i32;
            }
        }
        let pos = hsum_i32_avx2(vec_pos);
        acc += d * d_q8 * (pos - 32 * min_acc) as f32;
    }
    acc
}

/// 4-row variant of [`q6_k_q8_k_row_dot_avx2`]: shares the Q8_K loads and
/// keeps 4 independent accumulator chains in flight (same structure as
/// [`q4_k_q8_k_row_dot_x4_avx2`]).
///
/// # Safety
/// Same as the single-row kernel; `rows_base` must point at 4 rows spaced
/// `row_bytes` apart.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q6_k_q8_k_row_dot_x4_avx2(
    rows_base: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    q8k: &[u8],
    out: &mut [f32; 4],
) {
    let mask_low = _mm256_set1_epi8(0x0f);
    let mask_high = _mm256_set1_epi8(0x03);
    let mut acc = [0.0_f32; 4];
    for block_idx in 0..blocks_per_row {
        let q8_ptr = q8k.as_ptr().add(block_idx * BLOCK_Q8_K_BYTES);
        let d_q8 = f32::from_le_bytes([*q8_ptr, *q8_ptr.add(1), *q8_ptr.add(2), *q8_ptr.add(3)]);
        let q8 = q8_ptr.add(4);
        let bsums = q8_ptr.add(4 + QK_K);
        let mut bs = [0_i32; 16];
        for (g, b) in bs.iter_mut().enumerate() {
            *b = read_q8_k_bsum(bsums, g) as i32;
        }
        let q8v: [__m256i; 8] = [
            _mm256_loadu_si256(q8 as *const __m256i),
            _mm256_loadu_si256(q8.add(32) as *const __m256i),
            _mm256_loadu_si256(q8.add(64) as *const __m256i),
            _mm256_loadu_si256(q8.add(96) as *const __m256i),
            _mm256_loadu_si256(q8.add(128) as *const __m256i),
            _mm256_loadu_si256(q8.add(160) as *const __m256i),
            _mm256_loadu_si256(q8.add(192) as *const __m256i),
            _mm256_loadu_si256(q8.add(224) as *const __m256i),
        ];

        for (r, acc_r) in acc.iter_mut().enumerate() {
            let w_ptr = rows_base.add(r * row_bytes + block_idx * BLOCK_Q6_K_SIZE);
            let ahead = w_ptr.add(3 * BLOCK_Q6_K_SIZE).cast::<i8>();
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead);
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.add(64));
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.add(128));
            // Next-quad sweep for short rows, deeper in-row sweep for long
            // rows; see the Q4_K x4 kernel.
            if blocks_per_row <= 16 {
                let next_quad = w_ptr.add(4 * row_bytes).cast::<i8>();
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad);
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad.add(64));
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad.add(128));
            } else {
                let far = w_ptr.add(16 * BLOCK_Q6_K_SIZE).cast::<i8>();
                _mm_prefetch::<{ _MM_HINT_T1 }>(far);
                _mm_prefetch::<{ _MM_HINT_T1 }>(far.add(64));
                _mm_prefetch::<{ _MM_HINT_T1 }>(far.add(128));
            }

            let d = f16_le_to_f32([*w_ptr.add(208), *w_ptr.add(209)]);
            let ql = w_ptr;
            let qh = w_ptr.add(128);
            let sc = std::slice::from_raw_parts(w_ptr.add(192) as *const i8, 16);

            let mut vec_pos = _mm256_setzero_si256();
            let mut min_acc: i32 = 0;
            for half in 0..2 {
                let s_base = half * 8;
                let _v_base = half * 128;
                let ql_lo = _mm256_loadu_si256(ql.add(half * 64) as *const __m256i);
                let ql_hi = _mm256_loadu_si256(ql.add(half * 64 + 32) as *const __m256i);
                let qh_v = _mm256_loadu_si256(qh.add(half * 32) as *const __m256i);
                let q1 = _mm256_or_si256(
                    _mm256_and_si256(ql_lo, mask_low),
                    _mm256_slli_epi16(_mm256_and_si256(qh_v, mask_high), 4),
                );
                let q2 = _mm256_or_si256(
                    _mm256_and_si256(ql_hi, mask_low),
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 2), mask_high), 4),
                );
                let q3 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), mask_low),
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 4), mask_high), 4),
                );
                let q4 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), mask_low),
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 6), mask_high), 4),
                );
                for (g, qv) in [q1, q2, q3, q4].into_iter().enumerate() {
                    let sa = sc[s_base + g * 2] as i16;
                    let sb = sc[s_base + g * 2 + 1] as i16;
                    let p16 = _mm256_maddubs_epi16(qv, q8v[half * 4 + g]);
                    let scale_pair = _mm256_set_m128i(_mm_set1_epi16(sb), _mm_set1_epi16(sa));
                    vec_pos = _mm256_add_epi32(vec_pos, _mm256_madd_epi16(p16, scale_pair));
                    let g0 = half * 8 + g * 2;
                    min_acc += sa as i32 * bs[g0];
                    min_acc += sb as i32 * bs[g0 + 1];
                }
            }
            let pos = hsum_i32_avx2(vec_pos);
            *acc_r += d * d_q8 * (pos - 32 * min_acc) as f32;
        }
    }
    *out = acc;
}

/// AVX-512 VNNI variant of [`q4_k_q8_k_row_dot_avx2`]. Uses `_mm512_dpbusd_epi32`
/// to fuse the `maddubs` + `madd(ones)` int8→int32 reduction into a single
/// instruction, and processes the two scale sub-groups (g1, g2) of each `gp`
/// iteration together in one 512-bit lane group. Integer math is identical to
/// the AVX2 kernel; only the instruction sequence differs, so results match
/// bit-for-bit in the integer domain.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q4_k_q8_k_row_dot_vnni(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    let mask = _mm256_set1_epi8(0x0f);
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w_ptr = row.as_ptr().wrapping_add(block_idx * BLOCK_Q4_K_SIZE);
        let q8_ptr = q8k.as_ptr().wrapping_add(block_idx * BLOCK_Q8_K_BYTES);

        let d_w = f16_le_to_f32([*w_ptr, *w_ptr.add(1)]);
        let dmin_w = f16_le_to_f32([*w_ptr.add(2), *w_ptr.add(3)]);
        let d_q8 = f32::from_le_bytes([*q8_ptr, *q8_ptr.add(1), *q8_ptr.add(2), *q8_ptr.add(3)]);
        let scales = std::slice::from_raw_parts(w_ptr.add(4), 12);
        let qs = w_ptr.add(16);
        let q8 = q8_ptr.add(4);
        let bsums = q8_ptr.add(4 + QK_K);

        let mut vec_pos = _mm512_setzero_si512();
        let mut min_acc: i32 = 0;
        for gp in 0..4 {
            let g1 = gp * 2;
            let g2 = g1 + 1;
            let (s1, ms1) = get_scale_min_k4(g1, scales);
            let (s2, ms2) = get_scale_min_k4(g2, scales);
            let packed = _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i);
            // q4_low pairs with group g1's q8 (low half), q4_high with g2 (high half).
            let q4_low = _mm256_and_si256(packed, mask);
            let q4_high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            // g1 and g2 q8 quants are contiguous (g2 = g1 + 1) → one 512-bit load.
            let q8_512 = _mm512_loadu_si512(q8.add(g1 * 32) as *const __m512i);
            let q4_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q4_low), q4_high, 1);
            // dpbusd: unsigned q4 (0..15) × signed q8 → int32, 16 lanes.
            // Lanes 0..8 = group g1 (scale s1), lanes 8..16 = group g2 (scale s2).
            let prod = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_512, q8_512);
            let scale_v = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi32(s1 as i32)),
                _mm256_set1_epi32(s2 as i32),
                1,
            );
            vec_pos = _mm512_add_epi32(vec_pos, _mm512_mullo_epi32(prod, scale_v));

            let bs1 =
                read_q8_k_bsum(bsums, g1 * 2) as i32 + read_q8_k_bsum(bsums, g1 * 2 + 1) as i32;
            let bs2 =
                read_q8_k_bsum(bsums, g2 * 2) as i32 + read_q8_k_bsum(bsums, g2 * 2 + 1) as i32;
            min_acc += ms1 as i32 * bs1;
            min_acc += ms2 as i32 * bs2;
        }
        let pos_acc = _mm512_reduce_add_epi32(vec_pos);
        acc += d_w * d_q8 * pos_acc as f32 - dmin_w * d_q8 * min_acc as f32;
    }
    acc
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q4_k_q8_k_row_dot_chunk_avx2(
    row: &[u8],
    blocks_per_row: usize,
    q8_panel: &[u8],
    q8_stride: usize,
    token_start: usize,
    token_count: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), token_count);
    let mask = _mm256_set1_epi8(0x0f);

    for block_idx in 0..blocks_per_row {
        let w_ptr = row.as_ptr().wrapping_add(block_idx * BLOCK_Q4_K_SIZE);
        let d_w = f16_le_to_f32([unsafe { *w_ptr }, unsafe { *w_ptr.add(1) }]);
        let dmin_w = f16_le_to_f32([unsafe { *w_ptr.add(2) }, unsafe { *w_ptr.add(3) }]);
        let scales = unsafe { std::slice::from_raw_parts(w_ptr.add(4), 12) };
        let qs = unsafe { w_ptr.add(16) };

        // Decode all 8 (scale, min_scale) pairs once per block.
        let mut g_scales = [0_i32; 8];
        let mut g_min_scales = [0_i32; 8];
        for g in 0..8 {
            let (s, ms) = get_scale_min_k4(g, scales);
            g_scales[g] = s as i32;
            g_min_scales[g] = ms as i32;
        }

        // Pre-decode q4 nibbles for all 4 group-pairs once (shared across tokens).
        let mut q4_lo = [_mm256_setzero_si256(); 4];
        let mut q4_hi = [_mm256_setzero_si256(); 4];
        for gp in 0..4 {
            let packed = unsafe { _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i) };
            q4_lo[gp] = _mm256_and_si256(packed, mask);
            q4_hi[gp] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
        }

        // Broadcast scales as i16 for madd_epi16: madd(p16, set1_epi16(s)) ==
        // s * (p0 + p1) per i32 lane — identical to madd(p16, ones) * s but
        // avoids the slow mullo_epi32. No overflow: |p16| <= 3810, s <= 63.
        let s_v = [
            _mm256_set1_epi16(g_scales[0] as i16),
            _mm256_set1_epi16(g_scales[1] as i16),
            _mm256_set1_epi16(g_scales[2] as i16),
            _mm256_set1_epi16(g_scales[3] as i16),
            _mm256_set1_epi16(g_scales[4] as i16),
            _mm256_set1_epi16(g_scales[5] as i16),
            _mm256_set1_epi16(g_scales[6] as i16),
            _mm256_set1_epi16(g_scales[7] as i16),
        ];

        for (token, out_value) in out.iter_mut().enumerate().take(token_count) {
            let q8_base = q8_panel
                .as_ptr()
                .wrapping_add((token_start + token) * q8_stride + block_idx * BLOCK_Q8_K_BYTES);
            let d_q8 = f32::from_le_bytes([
                unsafe { *q8_base },
                unsafe { *q8_base.add(1) },
                unsafe { *q8_base.add(2) },
                unsafe { *q8_base.add(3) },
            ]);
            let q8 = unsafe { q8_base.add(4) };
            let bsums = unsafe { q8_base.add(4 + QK_K) };

            // Single vector accumulator across all 8 groups → 1 hsum per block.
            let mut vec_pos = _mm256_setzero_si256();
            for gp in 0..4 {
                let g1 = gp * 2;
                let g2 = g1 + 1;
                let q8_low = unsafe { _mm256_loadu_si256(q8.add(g1 * 32) as *const __m256i) };
                let q8_high = unsafe { _mm256_loadu_si256(q8.add(g2 * 32) as *const __m256i) };
                let p16_low = _mm256_maddubs_epi16(q4_lo[gp], q8_low);
                let p16_high = _mm256_maddubs_epi16(q4_hi[gp], q8_high);
                let scaled_low = _mm256_madd_epi16(p16_low, s_v[g1]);
                let scaled_high = _mm256_madd_epi16(p16_high, s_v[g2]);
                vec_pos = _mm256_add_epi32(vec_pos, _mm256_add_epi32(scaled_low, scaled_high));
            }
            let pos = unsafe { hsum_i32_avx2(vec_pos) };

            // Min correction: scalar over 8 groups is cheap, but use the
            // precomputed bsums directly as i16.
            let mut min: i32 = 0;
            for (g, min_scale) in g_min_scales.iter().enumerate() {
                let bs = unsafe { read_q8_k_bsum(bsums, g * 2) } as i32
                    + unsafe { read_q8_k_bsum(bsums, g * 2 + 1) } as i32;
                min += min_scale * bs;
            }

            let d_scale = d_w * d_q8;
            let dmin_scale = dmin_w * d_q8;
            *out_value += d_scale * pos as f32 - dmin_scale * min as f32;
        }
    }
}

#[inline]
unsafe fn read_q8_k_bsum(bsums: *const u8, index: usize) -> i16 {
    let ptr = unsafe { bsums.add(index * 2) };
    i16::from_le_bytes([unsafe { *ptr }, unsafe { *ptr.add(1) }])
}

/// Horizontal sum of 8 packed int32 values.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn hsum_i32_avx2(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let shuf = _mm_shuffle_epi32(sum128, 0b1110);
    let sum64 = _mm_add_epi32(sum128, shuf);
    let shuf2 = _mm_shuffle_epi32(sum64, 0b01);
    let sum32 = _mm_add_epi32(sum64, shuf2);
    _mm_cvtsi128_si32(sum32)
}

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

#[inline]
fn q4_k_value(block: &[u8], idx: usize) -> f32 {
    let d = f16_le_to_f32([block[0], block[1]]);
    let min = f16_le_to_f32([block[2], block[3]]);
    let scales = &block[4..16];
    let qs = &block[16..144];
    let group = idx / 32;
    let pair = group / 2;
    let (scale, min_scale) = get_scale_min_k4(group, scales);
    let q = if group.is_multiple_of(2) {
        qs[pair * 32 + (idx % 32)] & 0x0f
    } else {
        qs[pair * 32 + (idx % 32)] >> 4
    };
    d * scale as f32 * q as f32 - min * min_scale as f32
}

#[inline]
#[allow(dead_code)]
fn q4_k_dot(block: &[u8], vector: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { q4_k_dot_avx2(block, vector) };
        }
    }
    q4_k_dot_scalar(block, vector)
}

/// AVX2 + FMA dot product between a dequantized Q4_K block (256 weights) and a
/// 256-element vector slice. Dequantizes nibbles on the fly so the matrix is
/// read at 4-bit density — this is the decode hot path.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(dead_code)]
unsafe fn q4_k_dot_avx2(block: &[u8], vector: &[f32]) -> f32 {
    let d = f16_le_to_f32([block[0], block[1]]);
    let min = f16_le_to_f32([block[2], block[3]]);
    let scales = &block[4..16];
    let qs = block.as_ptr().wrapping_add(16);
    let mask = _mm_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_ps();
    for group_pair in 0..4 {
        let group1 = group_pair * 2;
        let group2 = group1 + 1;
        let (scale1, min_scale1) = get_scale_min_k4(group1, scales);
        let (scale2, min_scale2) = get_scale_min_k4(group2, scales);
        let d1 = _mm256_set1_ps(d * scale1 as f32);
        let min1 = _mm256_set1_ps(min * min_scale1 as f32);
        let d2 = _mm256_set1_ps(d * scale2 as f32);
        let min2 = _mm256_set1_ps(min * min_scale2 as f32);
        let q_base = group_pair * 32;
        let v_base = group_pair * 64;
        for l in (0..32).step_by(8) {
            let packed = unsafe { _mm_loadl_epi64(qs.add(q_base + l).cast::<__m128i>()) };
            let low_u8 = _mm_and_si128(packed, mask);
            let high_u8 = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
            let low = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low_u8));
            let high = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high_u8));
            // term = d * q - min
            let term1 = _mm256_fmsub_ps(low, d1, min1);
            let term2 = _mm256_fmsub_ps(high, d2, min2);
            let v1 = unsafe { _mm256_loadu_ps(vector.as_ptr().add(v_base + l)) };
            let v2 = unsafe { _mm256_loadu_ps(vector.as_ptr().add(v_base + 32 + l)) };
            acc = _mm256_fmadd_ps(term1, v1, acc);
            acc = _mm256_fmadd_ps(term2, v2, acc);
        }
    }
    // Horizontal sum of the 8 lanes.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[inline]
fn q4_k_dot_scalar(block: &[u8], vector: &[f32]) -> f32 {
    let d = f16_le_to_f32([block[0], block[1]]);
    let min = f16_le_to_f32([block[2], block[3]]);
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut sum = 0.0_f32;
    for group_pair in 0..4 {
        let group1 = group_pair * 2;
        let group2 = group1 + 1;
        let (scale1, min_scale1) = get_scale_min_k4(group1, scales);
        let (scale2, min_scale2) = get_scale_min_k4(group2, scales);
        let d1 = d * scale1 as f32;
        let min1 = min * min_scale1 as f32;
        let d2 = d * scale2 as f32;
        let min2 = min * min_scale2 as f32;
        let q_base = group_pair * 32;
        let v_base = group_pair * 64;
        for l in 0..32 {
            let packed = qs[q_base + l];
            sum += (d1 * (packed & 0x0f) as f32 - min1) * vector[v_base + l];
            sum += (d2 * (packed >> 4) as f32 - min2) * vector[v_base + 32 + l];
        }
    }
    sum
}

#[inline]
fn accumulate_q4_k_block(block: &[u8], factor: f32, output: &mut [f32]) {
    let d = f16_le_to_f32([block[0], block[1]]);
    let min = f16_le_to_f32([block[2], block[3]]);
    let scales = &block[4..16];
    let qs = &block[16..144];
    for group_pair in 0..4 {
        let group1 = group_pair * 2;
        let group2 = group1 + 1;
        let (scale1, min_scale1) = get_scale_min_k4(group1, scales);
        let (scale2, min_scale2) = get_scale_min_k4(group2, scales);
        let d1 = d * scale1 as f32;
        let min1 = min * min_scale1 as f32;
        let d2 = d * scale2 as f32;
        let min2 = min * min_scale2 as f32;
        let q_base = group_pair * 32;
        let out_base = group_pair * 64;
        for l in 0..32 {
            let packed = qs[q_base + l];
            output[out_base + l] += (d1 * (packed & 0x0f) as f32 - min1) * factor;
            output[out_base + 32 + l] += (d2 * (packed >> 4) as f32 - min2) * factor;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(dead_code)]
unsafe fn accumulate_q4_k_block_avx2(block: *const u8, factor: f32, output: *mut f32) {
    let d = f16_le_to_f32(unsafe { [*block, *block.add(1)] });
    let min = f16_le_to_f32(unsafe { [*block.add(2), *block.add(3)] });
    let scales = unsafe { std::slice::from_raw_parts(block.add(4), 12) };
    let qs = unsafe { block.add(16) };
    let mask = _mm_set1_epi8(0x0f);
    let factor_v = _mm256_set1_ps(factor);

    for group_pair in 0..4 {
        let group1 = group_pair * 2;
        let group2 = group1 + 1;
        let (scale1, min_scale1) = get_scale_min_k4(group1, scales);
        let (scale2, min_scale2) = get_scale_min_k4(group2, scales);
        let d1 = _mm256_set1_ps(d * scale1 as f32);
        let min1 = _mm256_set1_ps(min * min_scale1 as f32);
        let d2 = _mm256_set1_ps(d * scale2 as f32);
        let min2 = _mm256_set1_ps(min * min_scale2 as f32);
        let q_base = group_pair * 32;
        let out_base = group_pair * 64;
        for l in (0..32).step_by(8) {
            let packed = unsafe { _mm_loadl_epi64(qs.add(q_base + l).cast::<__m128i>()) };
            let low_u8 = _mm_and_si128(packed, mask);
            let high_u8 = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
            let low = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low_u8));
            let high = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high_u8));
            let vals1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(low, d1), min1), factor_v);
            let vals2 = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(high, d2), min2), factor_v);
            let out1 = unsafe { output.add(out_base + l) };
            let out2 = unsafe { output.add(out_base + 32 + l) };
            let cur1 = unsafe { _mm256_loadu_ps(out1) };
            let cur2 = unsafe { _mm256_loadu_ps(out2) };
            unsafe {
                _mm256_storeu_ps(out1, _mm256_add_ps(cur1, vals1));
                _mm256_storeu_ps(out2, _mm256_add_ps(cur2, vals2));
            }
        }
    }
}

fn gemv_q4_k_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q4_K_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let use_avx2 = false;

    let row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;
    let compute_row = |row_idx: usize| -> f32 {
        let row_start = row_idx * row_bytes;
        let row = &quantized_matrix[row_start..row_start + row_bytes];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if use_avx2 {
                return unsafe { q4_k_row_dot_avx2(row, blocks_per_row, vector) };
            }
        }
        let _ = use_avx2;
        let mut sum = 0.0_f32;
        for (block_idx, block) in row.chunks_exact(BLOCK_Q4_K_SIZE).enumerate() {
            let v_off = block_idx * QK_K;
            sum += q4_k_dot_scalar(block, &vector[v_off..v_off + QK_K]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

/// Whole-row Q4_K dot product. Accumulates over all `blocks_per_row` blocks
/// into four independent AVX2 registers (broken-up dependency chain so the
/// CPU can keep 4 FMA ops in flight per cycle on Zen 3/4) and does a single
/// horizontal reduce at the end.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q4_k_row_dot_avx2(row: &[u8], blocks_per_row: usize, vector: &[f32]) -> f32 {
    let mask = _mm_set1_epi8(0x0f);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    for block_idx in 0..blocks_per_row {
        let block_ptr = row.as_ptr().wrapping_add(block_idx * BLOCK_Q4_K_SIZE);
        let v_ptr = vector.as_ptr().wrapping_add(block_idx * QK_K);

        let d = f16_le_to_f32([unsafe { *block_ptr }, unsafe { *block_ptr.add(1) }]);
        let min = f16_le_to_f32([unsafe { *block_ptr.add(2) }, unsafe { *block_ptr.add(3) }]);
        let scales = unsafe { std::slice::from_raw_parts(block_ptr.add(4), 12) };
        let qs = unsafe { block_ptr.add(16) };

        for group_pair in 0..4 {
            let group1 = group_pair * 2;
            let group2 = group1 + 1;
            let (scale1, min_scale1) = get_scale_min_k4(group1, scales);
            let (scale2, min_scale2) = get_scale_min_k4(group2, scales);
            let d1 = _mm256_set1_ps(d * scale1 as f32);
            let min1 = _mm256_set1_ps(min * min_scale1 as f32);
            let d2 = _mm256_set1_ps(d * scale2 as f32);
            let min2 = _mm256_set1_ps(min * min_scale2 as f32);
            let q_base = group_pair * 32;
            let v_base = group_pair * 64;
            // Unroll inner loop to 4 lanes feeding 4 accumulators.
            let p0 = unsafe { _mm_loadl_epi64(qs.add(q_base).cast::<__m128i>()) };
            let p1 = unsafe { _mm_loadl_epi64(qs.add(q_base + 8).cast::<__m128i>()) };
            let p2 = unsafe { _mm_loadl_epi64(qs.add(q_base + 16).cast::<__m128i>()) };
            let p3 = unsafe { _mm_loadl_epi64(qs.add(q_base + 24).cast::<__m128i>()) };

            let l0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p0, mask)));
            let h0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(
                _mm_srli_epi16(p0, 4),
                mask,
            )));
            let l1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p1, mask)));
            let h1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(
                _mm_srli_epi16(p1, 4),
                mask,
            )));
            let l2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p2, mask)));
            let h2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(
                _mm_srli_epi16(p2, 4),
                mask,
            )));
            let l3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p3, mask)));
            let h3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(
                _mm_srli_epi16(p3, 4),
                mask,
            )));

            let v_l0 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base)) };
            let v_l1 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 8)) };
            let v_l2 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 16)) };
            let v_l3 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 24)) };
            let v_h0 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 32)) };
            let v_h1 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 40)) };
            let v_h2 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 48)) };
            let v_h3 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 56)) };

            let t_l0 = _mm256_fmsub_ps(l0, d1, min1);
            let t_l1 = _mm256_fmsub_ps(l1, d1, min1);
            let t_l2 = _mm256_fmsub_ps(l2, d1, min1);
            let t_l3 = _mm256_fmsub_ps(l3, d1, min1);
            let t_h0 = _mm256_fmsub_ps(h0, d2, min2);
            let t_h1 = _mm256_fmsub_ps(h1, d2, min2);
            let t_h2 = _mm256_fmsub_ps(h2, d2, min2);
            let t_h3 = _mm256_fmsub_ps(h3, d2, min2);

            acc0 = _mm256_fmadd_ps(t_l0, v_l0, acc0);
            acc1 = _mm256_fmadd_ps(t_l1, v_l1, acc1);
            acc2 = _mm256_fmadd_ps(t_l2, v_l2, acc2);
            acc3 = _mm256_fmadd_ps(t_l3, v_l3, acc3);
            acc0 = _mm256_fmadd_ps(t_h0, v_h0, acc0);
            acc1 = _mm256_fmadd_ps(t_h1, v_h1, acc1);
            acc2 = _mm256_fmadd_ps(t_h2, v_h2, acc2);
            acc3 = _mm256_fmadd_ps(t_h3, v_h3, acc3);
        }
    }

    let acc01 = _mm256_add_ps(acc0, acc1);
    let acc23 = _mm256_add_ps(acc2, acc3);
    let acc = _mm256_add_ps(acc01, acc23);
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[inline]
fn q2_k_dot_scalar(block: &[u8], vector: &[f32]) -> f32 {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = f16_le_to_f32([block[80], block[81]]);
    let min_v = f16_le_to_f32([block[82], block[83]]);
    let mut sum = 0.0_f32;
    let mut is: usize = 0;
    let mut weight_off: usize = 0;
    for outer in 0..2 {
        let qs_off = outer * 32;
        for _ in 0..4 {
            let sc1 = scales[is];
            is += 1;
            let sc2 = scales[is];
            is += 1;
            let shift = ((is / 2 - 1) % 4) * 2;
            let dl1 = d * (sc1 & 0x0F) as f32;
            let ml1 = min_v * (sc1 >> 4) as f32;
            let dl2 = d * (sc2 & 0x0F) as f32;
            let ml2 = min_v * (sc2 >> 4) as f32;
            for l in 0..16 {
                let q = ((qs[qs_off + l] >> shift) & 3) as f32;
                sum += (dl1 * q - ml1) * vector[weight_off + l];
            }
            for l in 0..16 {
                let q = ((qs[qs_off + 16 + l] >> shift) & 3) as f32;
                sum += (dl2 * q - ml2) * vector[weight_off + 16 + l];
            }
            weight_off += 32;
        }
    }
    sum
}

#[inline]
fn q2_k_dot(block: &[u8], vector: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { q2_k_dot_avx2(block, vector) };
        }
    }
    q2_k_dot_scalar(block, vector)
}

/// AVX2 + FMA dot product between a Q2_K block (256 weights, 2-bit packed)
/// and a 256-element f32 vector. Dequantizes 2-bit quants on the fly, mirroring
/// the Q4_K fast path so the matrix is read at ~2 bits/weight — this is the
/// decode hot path for Q2_K-quantized models.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q2_k_dot_avx2(block: &[u8], vector: &[f32]) -> f32 {
    let scales = &block[0..16];
    let qs_ptr = block.as_ptr().wrapping_add(16);
    let d = f16_le_to_f32([block[80], block[81]]);
    let min_v = f16_le_to_f32([block[82], block[83]]);
    let mask3 = _mm_set1_epi8(3);
    let mut acc = _mm256_setzero_ps();

    // Unroll the inner shift loop so _mm_srli_epi16 takes a const immediate.
    macro_rules! sub_block {
        ($shift:literal, $is_idx:expr, $qs_outer:expr, $v_outer:expr, $weight_idx:expr) => {{
            let sc1 = scales[$is_idx];
            let sc2 = scales[$is_idx + 1];
            let dl1 = _mm256_set1_ps(d * (sc1 & 0x0F) as f32);
            let ml1 = _mm256_set1_ps(min_v * (sc1 >> 4) as f32);
            let dl2 = _mm256_set1_ps(d * (sc2 & 0x0F) as f32);
            let ml2 = _mm256_set1_ps(min_v * (sc2 >> 4) as f32);
            // Lower 16 weights (two lanes of 8).
            for half in 0..2 {
                let lane_ptr = $qs_outer.add(half * 8);
                let q8 = _mm_loadl_epi64(lane_ptr as *const __m128i);
                let shifted = if $shift == 0 {
                    q8
                } else {
                    _mm_srli_epi16(q8, $shift)
                };
                let masked = _mm_and_si128(shifted, mask3);
                let f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(masked));
                let term = _mm256_fmsub_ps(f32x8, dl1, ml1);
                let v = _mm256_loadu_ps(vector.as_ptr().add($weight_idx + half * 8));
                acc = _mm256_fmadd_ps(term, v, acc);
            }
            // Upper 16 weights (two lanes of 8).
            for half in 0..2 {
                let lane_ptr = $qs_outer.add(16 + half * 8);
                let q8 = _mm_loadl_epi64(lane_ptr as *const __m128i);
                let shifted = if $shift == 0 {
                    q8
                } else {
                    _mm_srli_epi16(q8, $shift)
                };
                let masked = _mm_and_si128(shifted, mask3);
                let f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(masked));
                let term = _mm256_fmsub_ps(f32x8, dl2, ml2);
                let v = _mm256_loadu_ps(vector.as_ptr().add($weight_idx + 16 + half * 8));
                acc = _mm256_fmadd_ps(term, v, acc);
            }
            let _ = $v_outer; // silence unused (kept for symmetry)
        }};
    }

    // Outer 0: qs[0..32], weights 0..128.
    let qs0 = qs_ptr;
    sub_block!(0, 0, qs0, 0usize, 0usize);
    sub_block!(2, 2, qs0, 0usize, 32usize);
    sub_block!(4, 4, qs0, 0usize, 64usize);
    sub_block!(6, 6, qs0, 0usize, 96usize);
    // Outer 1: qs[32..64], weights 128..256.
    let qs1 = qs_ptr.add(32);
    sub_block!(0, 8, qs1, 0usize, 128usize);
    sub_block!(2, 10, qs1, 0usize, 160usize);
    sub_block!(4, 12, qs1, 0usize, 192usize);
    sub_block!(6, 14, qs1, 0usize, 224usize);

    // Horizontal sum.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

fn gemv_q2_k_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q2_K_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let compute_row = |row_idx: usize| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_Q2_K_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q2_K_SIZE)];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q2_K_SIZE).enumerate() {
            let vector_offset = block_idx * QK_K;
            sum += q2_k_dot(block, &vector[vector_offset..vector_offset + QK_K]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

#[inline]
fn q6_k_value(block: &[u8], idx: usize) -> f32 {
    let d = f16_le_to_f32([block[208], block[209]]);
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let half = idx / 128;
    let rem = idx % 128;
    let l = rem % 32;
    let group = rem / 32;
    // Each 128-weight half of the block consumes its own 64-byte ql window and
    // 32-byte qh window; advance into the second half for idx >= 128.
    let ql_base = half * 64;
    let qh_base = half * 32;
    let (q_low, q_high, scale_idx) = match group {
        0 => (ql[ql_base + l] & 0x0f, qh[qh_base + l] & 0x03, l / 16),
        1 => (
            ql[ql_base + l + 32] & 0x0f,
            (qh[qh_base + l] >> 2) & 0x03,
            l / 16 + 2,
        ),
        2 => (
            ql[ql_base + l] >> 4,
            (qh[qh_base + l] >> 4) & 0x03,
            l / 16 + 4,
        ),
        _ => (
            ql[ql_base + l + 32] >> 4,
            (qh[qh_base + l] >> 6) & 0x03,
            l / 16 + 6,
        ),
    };
    let scale = sc[scale_idx + half * 8] as i8 as f32;
    let q = ((q_low as i32) | ((q_high as i32) << 4)) - 32;
    d * scale * q as f32
}

#[inline]
fn q6_k_dot(block: &[u8], vector: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { q6_k_dot_avx2(block, vector) };
        }
    }
    q6_k_dot_scalar(block, vector)
}

#[inline]
fn q6_k_dot_scalar(block: &[u8], vector: &[f32]) -> f32 {
    let d = f16_le_to_f32([block[208], block[209]]);
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let mut sum = 0.0_f32;
    let mut q_ptr = 0;
    for half in 0..2 {
        let scale_base = half * 8;
        let ql_base = half * 64;
        let qh_base = half * 32;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_base + l] & 0x0f) as i32 | (((qh[qh_base + l] & 3) as i32) << 4)) - 32;
            let q2 = ((ql[ql_base + l + 32] & 0x0f) as i32
                | ((((qh[qh_base + l] >> 2) & 3) as i32) << 4))
                - 32;
            let q3 =
                ((ql[ql_base + l] >> 4) as i32 | ((((qh[qh_base + l] >> 4) & 3) as i32) << 4)) - 32;
            let q4 = ((ql[ql_base + l + 32] >> 4) as i32
                | ((((qh[qh_base + l] >> 6) & 3) as i32) << 4))
                - 32;
            sum += d * sc[scale_base + is] as i8 as f32 * q1 as f32 * vector[q_ptr + l];
            sum += d * sc[scale_base + is + 2] as i8 as f32 * q2 as f32 * vector[q_ptr + 32 + l];
            sum += d * sc[scale_base + is + 4] as i8 as f32 * q3 as f32 * vector[q_ptr + 64 + l];
            sum += d * sc[scale_base + is + 6] as i8 as f32 * q4 as f32 * vector[q_ptr + 96 + l];
        }
        q_ptr += 128;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q6_k_dot_avx2(block: &[u8], vector: &[f32]) -> f32 {
    let d = f16_le_to_f32([block[208], block[209]]);
    let ql = block.as_ptr();
    let qh = block.as_ptr().wrapping_add(128);
    let sc = &block[192..208];
    let mask_low = _mm_set1_epi8(0x0f);
    let mask_high = _mm_set1_epi8(0x03);
    let offset = _mm256_set1_ps(32.0);
    let mut acc = _mm256_setzero_ps();

    for half in 0..2 {
        let scale_base = half * 8;
        let v_base = half * 128;
        let ql_base = half * 64;
        let qh_base = half * 32;
        for l in (0..32).step_by(8) {
            let is = l / 16;
            let ql1 = unsafe { _mm_loadl_epi64(ql.add(ql_base + l).cast::<__m128i>()) };
            let ql2 = unsafe { _mm_loadl_epi64(ql.add(ql_base + l + 32).cast::<__m128i>()) };
            let qh_v = unsafe { _mm_loadl_epi64(qh.add(qh_base + l).cast::<__m128i>()) };

            let low1 = _mm_and_si128(ql1, mask_low);
            let low2 = _mm_and_si128(ql2, mask_low);
            let low3 = _mm_and_si128(_mm_srli_epi16(ql1, 4), mask_low);
            let low4 = _mm_and_si128(_mm_srli_epi16(ql2, 4), mask_low);
            let high1 = _mm_slli_epi16(_mm_and_si128(qh_v, mask_high), 4);
            let high2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 2), mask_high), 4);
            let high3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 4), mask_high), 4);
            let high4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 6), mask_high), 4);

            macro_rules! acc_group {
                ($low:expr, $high:expr, $scale_idx:expr, $vec_off:expr) => {{
                    let q_u8 = _mm_or_si128($low, $high);
                    let q = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_u8)), offset);
                    let scale = _mm256_set1_ps(d * sc[$scale_idx] as i8 as f32);
                    let w = _mm256_mul_ps(q, scale);
                    let v = unsafe { _mm256_loadu_ps(vector.as_ptr().add($vec_off)) };
                    acc = _mm256_fmadd_ps(w, v, acc);
                }};
            }

            acc_group!(low1, high1, scale_base + is, v_base + l);
            acc_group!(low2, high2, scale_base + is + 2, v_base + 32 + l);
            acc_group!(low3, high3, scale_base + is + 4, v_base + 64 + l);
            acc_group!(low4, high4, scale_base + is + 6, v_base + 96 + l);
        }
    }

    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[inline]
fn accumulate_q6_k_block(block: &[u8], factor: f32, output: &mut [f32]) {
    let d = f16_le_to_f32([block[208], block[209]]);
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let mut q_ptr = 0;
    for half in 0..2 {
        let scale_base = half * 8;
        let ql_base = half * 64;
        let qh_base = half * 32;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_base + l] & 0x0f) as i32 | (((qh[qh_base + l] & 3) as i32) << 4)) - 32;
            let q2 = ((ql[ql_base + l + 32] & 0x0f) as i32
                | ((((qh[qh_base + l] >> 2) & 3) as i32) << 4))
                - 32;
            let q3 =
                ((ql[ql_base + l] >> 4) as i32 | ((((qh[qh_base + l] >> 4) & 3) as i32) << 4)) - 32;
            let q4 = ((ql[ql_base + l + 32] >> 4) as i32
                | ((((qh[qh_base + l] >> 6) & 3) as i32) << 4))
                - 32;
            output[q_ptr + l] += d * sc[scale_base + is] as i8 as f32 * q1 as f32 * factor;
            output[q_ptr + 32 + l] += d * sc[scale_base + is + 2] as i8 as f32 * q2 as f32 * factor;
            output[q_ptr + 64 + l] += d * sc[scale_base + is + 4] as i8 as f32 * q3 as f32 * factor;
            output[q_ptr + 96 + l] += d * sc[scale_base + is + 6] as i8 as f32 * q4 as f32 * factor;
        }
        q_ptr += 128;
    }
}

// IQ1_S GEMV: decode-once approach.
// Each block is 50 bytes for 256 values. We dequantize each block to f32
// in a scratch buffer, then do a standard f32 dot product.
// This is slower than native IQ1_S dot products but requires no large
// lookup table and is correct.
const BLOCK_IQ1_S_SIZE: usize = 2 + 32 + 16; // ggml_half d + qs[32] + qh[16]
const IQ1S_DELTA: f32 = 0.125;

#[inline]
#[allow(clippy::needless_range_loop)]
fn iq1s_grid_decode(index: u16, out: &mut [i8; 8]) {
    let mut idx = index;
    for i in 0..8 {
        let bits = (idx & 3) as i8;
        out[i] = match bits {
            0 => -1,
            1 => 0,
            _ => 1,
        };
        idx >>= 2;
        if i == 3 {
            idx = index >> 8;
        }
    }
}

#[inline]
fn iq1s_dequantize_block(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), BLOCK_IQ1_S_SIZE);
    assert_eq!(out.len(), QK_K);
    let d = f16_le_to_f32([block[0], block[1]]);
    let qs = &block[2..34];
    let qh = &block[34..50];
    let qh_u16: &[u16] = unsafe { std::slice::from_raw_parts(qh.as_ptr() as *const u16, 16) };
    let mut out_ptr = 0_usize;
    let mut grid_vals = [0_i8; 8];
    for ib in 0..(QK_K / 32) {
        let dl = d * (2.0 * (((qh_u16[ib] >> 12) & 7) as f32) + 1.0);
        let delta = if qh_u16[ib] & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        for l in 0..4 {
            let grid_idx = (qs[l + ib * 4] as u16) | (((qh_u16[ib] >> (3 * l)) & 7) << 8);
            iq1s_grid_decode(grid_idx, &mut grid_vals);
            for j in 0..8 {
                out[out_ptr + j] = dl * (grid_vals[j] as f32 + delta);
            }
            out_ptr += 8;
        }
    }
}

#[inline]
fn ue4m3_to_f32(byte: u8) -> f32 {
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;
    if exp == 0 {
        (mant as f32) * 2.0_f32.powi(-9)
    } else {
        (1.0 + (mant as f32) / 8.0) * 2.0_f32.powi(exp as i32 - 7)
    }
}

#[inline]
#[allow(clippy::needless_range_loop)]
fn nvfp4_dequantize_block(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), BLOCK_NVFP4_SIZE);
    assert_eq!(out.len(), QK_NVFP4);
    let scales = &block[..QK_NVFP4 / QK_NVFP4_SUB];
    let qs = &block[QK_NVFP4 / QK_NVFP4_SUB..];
    for sub in 0..(QK_NVFP4 / QK_NVFP4_SUB) {
        let scale = ue4m3_to_f32(scales[sub]);
        let q_base = sub * (QK_NVFP4_SUB / 2);
        let out_base = sub * QK_NVFP4_SUB;
        for j in 0..(QK_NVFP4_SUB / 2) {
            let packed = qs[q_base + j];
            out[out_base + j] = scale * E2M1_DOUBLED_VALUES[(packed & 0x0f) as usize];
            out[out_base + j + QK_NVFP4_SUB / 2] =
                scale * E2M1_DOUBLED_VALUES[(packed >> 4) as usize];
        }
    }
}

fn gemv_nvfp4_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_NVFP4;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_NVFP4_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let compute_row = |row_idx: usize, scratch: &mut [f32; QK_NVFP4]| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_NVFP4_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + blocks_per_row * BLOCK_NVFP4_SIZE];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_NVFP4_SIZE).enumerate() {
            let v_off = block_idx * QK_NVFP4;
            nvfp4_dequantize_block(block, scratch);
            sum += dot_f32_fast(scratch, &vector[v_off..v_off + QK_NVFP4]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out)| {
                let mut scratch = [0.0_f32; QK_NVFP4];
                *out = compute_row(row_idx, &mut scratch);
            });
    } else {
        let mut scratch = [0.0_f32; QK_NVFP4];
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx, &mut scratch);
        }
    }
    Ok(())
}

fn gemv_iq1_s_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_IQ1_S_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let compute_row = |row_idx: usize, scratch: &mut [f32; QK_K]| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_IQ1_S_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_IQ1_S_SIZE)];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_IQ1_S_SIZE).enumerate() {
            let v_off = block_idx * QK_K;
            iq1s_dequantize_block(block, scratch);
            sum += crate::flash_attention::dot_product_f32(scratch, &vector[v_off..v_off + QK_K]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out)| {
                let mut scratch = [0.0_f32; QK_K];
                *out = compute_row(row_idx, &mut scratch);
            });
    } else {
        let mut scratch = [0.0_f32; QK_K];
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx, &mut scratch);
        }
    }
    Ok(())
}

// IQ1_M GEMV: similar decode-once approach.
const BLOCK_IQ1_M_SIZE: usize = 32 + 16 + 8; // qs[32] + qh[16] + scales[8]

#[inline]
fn iq1m_dequantize_block(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), BLOCK_IQ1_M_SIZE);
    assert_eq!(out.len(), QK_K);
    let qs = &block[0..32];
    let qh = &block[32..48];
    let scales = &block[48..56];
    let sc: &[u16] = unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u16, 4) };
    let scale_u16 =
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    let d = f16_bits_to_f32(scale_u16);
    let mut out_ptr = 0_usize;
    let mut grid_vals = [0_i8; 8];
    for ib in 0..(QK_K / 32) {
        let sc_ib = scales[ib / 2];
        let dl1 = d * (2.0 * (((sc_ib >> (6 * (ib % 2))) & 0x7) as f32) + 1.0);
        let dl2 = d * (2.0 * (((sc_ib >> (6 * (ib % 2) + 3)) & 0x7) as f32) + 1.0);
        let idx0 = qs[ib * 4] as u16 | ((qh[ib * 2] as u16) << 8 & 0x700);
        let idx1 = qs[ib * 4 + 1] as u16 | ((qh[ib * 2] as u16) << 4 & 0x700);
        let idx2 = qs[ib * 4 + 2] as u16 | ((qh[ib * 2 + 1] as u16) << 8 & 0x700);
        let idx3 = qs[ib * 4 + 3] as u16 | ((qh[ib * 2 + 1] as u16) << 4 & 0x700);
        let deltas = [
            if qh[ib * 2] & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if qh[ib * 2] & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if qh[ib * 2 + 1] & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if qh[ib * 2 + 1] & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
        ];
        iq1s_grid_decode(idx0, &mut grid_vals);
        for j in 0..8 {
            out[out_ptr + j] = dl1 * (grid_vals[j] as f32 + deltas[0]);
        }
        out_ptr += 8;
        iq1s_grid_decode(idx1, &mut grid_vals);
        for j in 0..8 {
            out[out_ptr + j] = dl1 * (grid_vals[j] as f32 + deltas[1]);
        }
        out_ptr += 8;
        iq1s_grid_decode(idx2, &mut grid_vals);
        for j in 0..8 {
            out[out_ptr + j] = dl2 * (grid_vals[j] as f32 + deltas[2]);
        }
        out_ptr += 8;
        iq1s_grid_decode(idx3, &mut grid_vals);
        for j in 0..8 {
            out[out_ptr + j] = dl2 * (grid_vals[j] as f32 + deltas[3]);
        }
        out_ptr += 8;
    }
}

fn gemv_iq1_m_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_IQ1_M_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let compute_row = |row_idx: usize, scratch: &mut [f32; QK_K]| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_IQ1_M_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_IQ1_M_SIZE)];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_IQ1_M_SIZE).enumerate() {
            let v_off = block_idx * QK_K;
            iq1m_dequantize_block(block, scratch);
            sum += crate::flash_attention::dot_product_f32(scratch, &vector[v_off..v_off + QK_K]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out)| {
                let mut scratch = [0.0_f32; QK_K];
                *out = compute_row(row_idx, &mut scratch);
            });
    } else {
        let mut scratch = [0.0_f32; QK_K];
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx, &mut scratch);
        }
    }
    Ok(())
}

fn gemv_q6_k_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q6_K_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let use_avx2 = false;

    let row_bytes = blocks_per_row * BLOCK_Q6_K_SIZE;
    let compute_row = |row_idx: usize| -> f32 {
        let row_start = row_idx * row_bytes;
        let row = &quantized_matrix[row_start..row_start + row_bytes];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if use_avx2 {
                return unsafe { q6_k_row_dot_avx2(row, blocks_per_row, vector) };
            }
        }
        let _ = use_avx2;
        let mut sum = 0.0_f32;
        for (block_idx, block) in row.chunks_exact(BLOCK_Q6_K_SIZE).enumerate() {
            let v_off = block_idx * QK_K;
            sum += q6_k_dot_scalar(block, &vector[v_off..v_off + QK_K]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

/// Whole-row Q6_K dot product. Same idea as `q4_k_row_dot_avx2`: one final
/// horizontal reduce instead of one per block.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q6_k_row_dot_avx2(row: &[u8], blocks_per_row: usize, vector: &[f32]) -> f32 {
    let mask_low = _mm_set1_epi8(0x0f);
    let mask_high = _mm_set1_epi8(0x03);
    let offset = _mm256_set1_ps(32.0);
    let mut acc = _mm256_setzero_ps();

    for block_idx in 0..blocks_per_row {
        let block_ptr = row.as_ptr().wrapping_add(block_idx * BLOCK_Q6_K_SIZE);
        let v_ptr = vector.as_ptr().wrapping_add(block_idx * QK_K);

        let d = f16_le_to_f32([unsafe { *block_ptr.add(208) }, unsafe {
            *block_ptr.add(209)
        }]);
        let ql = block_ptr;
        let qh = block_ptr.wrapping_add(128);
        let sc = unsafe { std::slice::from_raw_parts(block_ptr.add(192), 16) };

        for half in 0..2 {
            let scale_base = half * 8;
            let v_base = half * 128;
            let ql_base = half * 64;
            let qh_base = half * 32;
            for l in (0..32).step_by(8) {
                let is = l / 16;
                let ql1 = unsafe { _mm_loadl_epi64(ql.add(ql_base + l).cast::<__m128i>()) };
                let ql2 = unsafe { _mm_loadl_epi64(ql.add(ql_base + l + 32).cast::<__m128i>()) };
                let qh_v = unsafe { _mm_loadl_epi64(qh.add(qh_base + l).cast::<__m128i>()) };

                let low1 = _mm_and_si128(ql1, mask_low);
                let low2 = _mm_and_si128(ql2, mask_low);
                let low3 = _mm_and_si128(_mm_srli_epi16(ql1, 4), mask_low);
                let low4 = _mm_and_si128(_mm_srli_epi16(ql2, 4), mask_low);
                let high1 = _mm_slli_epi16(_mm_and_si128(qh_v, mask_high), 4);
                let high2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 2), mask_high), 4);
                let high3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 4), mask_high), 4);
                let high4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 6), mask_high), 4);

                macro_rules! acc_group {
                    ($low:expr, $high:expr, $scale_idx:expr, $vec_off:expr) => {{
                        let q_u8 = _mm_or_si128($low, $high);
                        let q =
                            _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_u8)), offset);
                        let scale = _mm256_set1_ps(d * sc[$scale_idx] as i8 as f32);
                        let w = _mm256_mul_ps(q, scale);
                        let v = unsafe { _mm256_loadu_ps(v_ptr.add($vec_off)) };
                        acc = _mm256_fmadd_ps(w, v, acc);
                    }};
                }

                acc_group!(low1, high1, scale_base + is, v_base + l);
                acc_group!(low2, high2, scale_base + is + 2, v_base + 32 + l);
                acc_group!(low3, high3, scale_base + is + 4, v_base + 64 + l);
                acc_group!(low4, high4, scale_base + is + 6, v_base + 96 + l);
            }
        }
    }

    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[allow(clippy::too_many_arguments, dead_code)]
fn gemv_qk_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
    block_size: usize,
    bits: usize,
    zero_point: f32,
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * block_size;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let compute_row = |row_idx: usize| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * block_size;
        let row_blocks = &quantized_matrix[row_start..row_start + (blocks_per_row * block_size)];
        for (block_idx, block) in row_blocks.chunks_exact(block_size).enumerate() {
            let d = f16_le_to_f32([block[0], block[1]]);
            let bitstream = &block[2..];
            let vector_offset = block_idx * QK_K;
            for idx in 0..QK_K {
                let q = extract_bits(bitstream, idx, bits) as f32;
                sum += (q - zero_point) * d * vector[vector_offset + idx];
            }
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

pub fn extract_bits(bitstream: &[u8], index: usize, bits: usize) -> u32 {
    let bit_offset = index * bits;
    let byte_index = bit_offset / 8;
    let shift = bit_offset % 8;

    let mut acc = 0_u32;
    for i in 0..4 {
        if let Some(byte) = bitstream.get(byte_index + i) {
            acc |= (*byte as u32) << (8 * i);
        }
    }

    (acc >> shift) & ((1_u32 << bits) - 1)
}

fn gemv_q8_0_f32_fused(
    quantized_matrix: &[u8],
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let rows = output.len();
    let blocks_per_row = cols / QK8_0;
    let compute_row = |row_idx: usize| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_Q8_0_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q8_0_SIZE)];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q8_0_SIZE).enumerate() {
            let vector_offset = block_idx * QK8_0;
            sum += q8_0_dot(block, &vector[vector_offset..vector_offset + QK8_0]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

#[inline]
fn q8_0_dot(block: &[u8], vector: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { q8_0_dot_avx2(block, vector) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { q8_0_dot_neon_aarch64(block, vector) };
        }
    }
    #[cfg(target_arch = "arm")]
    {
        if std::arch::is_arm_feature_detected!("neon") {
            return unsafe { q8_0_dot_neon_arm(block, vector) };
        }
    }
    q8_0_dot_scalar(block, vector)
}

#[inline]
fn q8_0_dot_scalar(block: &[u8], vector: &[f32]) -> f32 {
    let scale = f16_le_to_f32([block[0], block[1]]);
    block[2..]
        .iter()
        .zip(vector)
        .map(|(q, v)| (*q as i8) as f32 * scale * *v)
        .sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q8_0_dot_avx2(block: &[u8], vector: &[f32]) -> f32 {
    let scale = _mm256_set1_ps(f16_le_to_f32([block[0], block[1]]));
    let mut acc = _mm256_setzero_ps();
    for lane in 0..4 {
        let q8 = unsafe { _mm_loadl_epi64(block.as_ptr().add(2 + lane * 8).cast::<__m128i>()) };
        let q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
        let v = unsafe { _mm256_loadu_ps(vector.as_ptr().add(lane * 8)) };
        acc = _mm256_fmadd_ps(_mm256_mul_ps(q, scale), v, acc);
    }
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q8_0_dot_neon_aarch64(block: &[u8], vector: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let scale = vdupq_n_f32(f16_le_to_f32([block[0], block[1]]));
    let mut acc = vdupq_n_f32(0.0);
    for lane in 0..4 {
        let q8 = unsafe { vld1_s8(block.as_ptr().add(2 + lane * 8).cast::<i8>()) };
        let q16 = vmovl_s8(q8);
        let q_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
        let q_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
        let v_lo = unsafe { vld1q_f32(vector.as_ptr().add(lane * 8)) };
        let v_hi = unsafe { vld1q_f32(vector.as_ptr().add(lane * 8 + 4)) };
        acc = vfmaq_f32(acc, vmulq_f32(q_lo, scale), v_lo);
        acc = vfmaq_f32(acc, vmulq_f32(q_hi, scale), v_hi);
    }
    vaddvq_f32(acc)
}

#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn q8_0_dot_neon_arm(block: &[u8], vector: &[f32]) -> f32 {
    use std::arch::arm::*;

    let scale = vdupq_n_f32(f16_le_to_f32([block[0], block[1]]));
    let mut acc = vdupq_n_f32(0.0);
    for lane in 0..4 {
        let q8 = unsafe { vld1_s8(block.as_ptr().add(2 + lane * 8).cast::<i8>()) };
        let q16 = vmovl_s8(q8);
        let q_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
        let q_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
        let v_lo = unsafe { vld1q_f32(vector.as_ptr().add(lane * 8)) };
        let v_hi = unsafe { vld1q_f32(vector.as_ptr().add(lane * 8 + 4)) };
        acc = vmlaq_f32(acc, vmulq_f32(q_lo, scale), v_lo);
        acc = vmlaq_f32(acc, vmulq_f32(q_hi, scale), v_hi);
    }
    let pair = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    let pair = vpadd_f32(pair, pair);
    vget_lane_f32(pair, 0)
}

// Transposed GEMV functions for GGUF weight matrices stored as [input_dim, output_dim]
fn gemv_f32_transposed_cpu(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) {
    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_chunks_mut(TRANSPOSED_GEMV_COL_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * TRANSPOSED_GEMV_COL_CHUNK;
                out_chunk.fill(0.0);
                let col_end = col_start + out_chunk.len();
                for (row_values, &vi) in matrix.chunks_exact(cols).zip(vector.iter()).take(rows) {
                    let row_chunk = &row_values[col_start..col_end];
                    for (out, &weight) in out_chunk.iter_mut().zip(row_chunk) {
                        *out += weight * vi;
                    }
                }
            });
    } else {
        output.fill(0.0);
        for (row_values, &vi) in matrix.chunks_exact(cols).zip(vector.iter()).take(rows) {
            for (j, &weight) in row_values.iter().enumerate() {
                output[j] += weight * vi;
            }
        }
    }
}

pub fn gemv_f32_transposed(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let expected_matrix_len = rows.saturating_mul(cols);
    if matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: matrix.len(),
        });
    }
    if vector.len() != rows {
        return Err(GemvError::InvalidVectorLength {
            expected: rows,
            actual: vector.len(),
        });
    }
    if output.len() != cols {
        return Err(GemvError::InvalidOutputLength {
            expected: cols,
            actual: output.len(),
        });
    }
    #[cfg(feature = "cuda")]
    if crate::cuda::cuda_build_info().detected_at_build {
        return crate::cuda::gemv_f32_transposed_cuda(matrix, rows, cols, vector, output)
            .map_err(|err| GemvError::Cuda(format!("{err:?}")));
    }
    gemv_f32_transposed_cpu(matrix, rows, cols, vector, output);
    Ok(())
}

#[allow(clippy::too_many_arguments, dead_code)]
fn gemv_qk_f32_transposed(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
    block_size: usize,
    bits: usize,
    zero_point: f32,
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * block_size;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != rows {
        return Err(GemvError::InvalidVectorLength {
            expected: rows,
            actual: vector.len(),
        });
    }
    if output.len() != cols {
        return Err(GemvError::InvalidOutputLength {
            expected: cols,
            actual: output.len(),
        });
    }

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        let use_q4_avx512 = bits == 4 && q4_avx512_available();
        let use_q4_avx2 = bits == 4 && !use_q4_avx512 && q4_avx2_available();
        output
            .par_chunks_mut(TRANSPOSED_GEMV_COL_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * TRANSPOSED_GEMV_COL_CHUNK;
                let block_start = col_start / QK_K;
                let block_end = ((col_start + out_chunk.len()).div_ceil(QK_K)).min(blocks_per_row);
                out_chunk.fill(0.0);
                for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
                    if vi == 0.0 {
                        continue;
                    }
                    let row_start = row_idx * blocks_per_row * block_size;
                    let row_blocks =
                        &quantized_matrix[row_start..row_start + (blocks_per_row * block_size)];
                    for block_idx in block_start..block_end {
                        let block_offset = block_idx * block_size;
                        let block = &row_blocks[block_offset..block_offset + block_size];
                        let d = f16_le_to_f32([block[0], block[1]]);
                        let factor = d * vi;
                        let local_col = block_idx * QK_K - col_start;
                        let block_output_len = (out_chunk.len() - local_col).min(QK_K);
                        let out_block = &mut out_chunk[local_col..local_col + block_output_len];
                        if bits == 4 && zero_point == 8.0 && block_output_len == QK_K {
                            accumulate_q4_block(
                                &block[2..],
                                factor,
                                out_block,
                                use_q4_avx512,
                                use_q4_avx2,
                            );
                        } else {
                            let bitstream = &block[2..];
                            for (idx, out) in out_block.iter_mut().enumerate() {
                                let q = extract_bits(bitstream, idx, bits) as f32;
                                *out += (q - zero_point) * factor;
                            }
                        }
                    }
                }
            });
    } else {
        output.fill(0.0);
        for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
            let row_start = row_idx * blocks_per_row * block_size;
            let row_blocks =
                &quantized_matrix[row_start..row_start + (blocks_per_row * block_size)];
            for (block_idx, block) in row_blocks.chunks_exact(block_size).enumerate() {
                let d = f16_le_to_f32([block[0], block[1]]);
                let bitstream = &block[2..];
                let col_start = block_idx * QK_K;
                for idx in 0..QK_K {
                    if col_start + idx >= cols {
                        break;
                    }
                    let q = extract_bits(bitstream, idx, bits) as f32;
                    output[col_start + idx] += (q - zero_point) * d * vi;
                }
            }
        }
    }
    Ok(())
}

fn gemv_q4_k_f32_transposed(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q4_K_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != rows {
        return Err(GemvError::InvalidVectorLength {
            expected: rows,
            actual: vector.len(),
        });
    }
    if output.len() != cols {
        return Err(GemvError::InvalidOutputLength {
            expected: cols,
            actual: output.len(),
        });
    }

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_chunks_mut(TRANSPOSED_GEMV_COL_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * TRANSPOSED_GEMV_COL_CHUNK;
                let block_start = col_start / QK_K;
                let block_end = ((col_start + out_chunk.len()).div_ceil(QK_K)).min(blocks_per_row);
                out_chunk.fill(0.0);
                for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
                    if vi == 0.0 {
                        continue;
                    }
                    let row_start = row_idx * blocks_per_row * BLOCK_Q4_K_SIZE;
                    let row_blocks = &quantized_matrix
                        [row_start..row_start + (blocks_per_row * BLOCK_Q4_K_SIZE)];
                    for block_idx in block_start..block_end {
                        let block_offset = block_idx * BLOCK_Q4_K_SIZE;
                        let block = &row_blocks[block_offset..block_offset + BLOCK_Q4_K_SIZE];
                        let local_col = block_idx * QK_K - col_start;
                        let block_output_len = (out_chunk.len() - local_col).min(QK_K);
                        let out_block = &mut out_chunk[local_col..local_col + block_output_len];
                        if block_output_len == QK_K {
                            accumulate_q4_k_block(block, vi, out_block);
                        } else {
                            for (idx, out) in out_block.iter_mut().enumerate() {
                                *out += q4_k_value(block, idx) * vi;
                            }
                        }
                    }
                }
            });
    } else {
        output.fill(0.0);
        for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
            if vi == 0.0 {
                continue;
            }
            let row_start = row_idx * blocks_per_row * BLOCK_Q4_K_SIZE;
            let row_blocks =
                &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q4_K_SIZE)];
            for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q4_K_SIZE).enumerate() {
                let col_start = block_idx * QK_K;
                for idx in 0..QK_K {
                    if col_start + idx >= cols {
                        break;
                    }
                    output[col_start + idx] += q4_k_value(block, idx) * vi;
                }
            }
        }
    }
    Ok(())
}

fn gemv_q6_k_f32_transposed(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q6_K_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != rows {
        return Err(GemvError::InvalidVectorLength {
            expected: rows,
            actual: vector.len(),
        });
    }
    if output.len() != cols {
        return Err(GemvError::InvalidOutputLength {
            expected: cols,
            actual: output.len(),
        });
    }

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_chunks_mut(TRANSPOSED_GEMV_COL_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * TRANSPOSED_GEMV_COL_CHUNK;
                let block_start = col_start / QK_K;
                let block_end = ((col_start + out_chunk.len()).div_ceil(QK_K)).min(blocks_per_row);
                out_chunk.fill(0.0);
                for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
                    if vi == 0.0 {
                        continue;
                    }
                    let row_start = row_idx * blocks_per_row * BLOCK_Q6_K_SIZE;
                    let row_blocks = &quantized_matrix
                        [row_start..row_start + (blocks_per_row * BLOCK_Q6_K_SIZE)];
                    for block_idx in block_start..block_end {
                        let block_offset = block_idx * BLOCK_Q6_K_SIZE;
                        let block = &row_blocks[block_offset..block_offset + BLOCK_Q6_K_SIZE];
                        let local_col = block_idx * QK_K - col_start;
                        let block_output_len = (out_chunk.len() - local_col).min(QK_K);
                        let out_block = &mut out_chunk[local_col..local_col + block_output_len];
                        if block_output_len == QK_K {
                            accumulate_q6_k_block(block, vi, out_block);
                        } else {
                            for (idx, out) in out_block.iter_mut().enumerate() {
                                *out += q6_k_value(block, idx) * vi;
                            }
                        }
                    }
                }
            });
    } else {
        output.fill(0.0);
        for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
            if vi == 0.0 {
                continue;
            }
            let row_start = row_idx * blocks_per_row * BLOCK_Q6_K_SIZE;
            let row_blocks =
                &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q6_K_SIZE)];
            for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q6_K_SIZE).enumerate() {
                let col_start = block_idx * QK_K;
                for idx in 0..QK_K {
                    if col_start + idx >= cols {
                        break;
                    }
                    output[col_start + idx] += q6_k_value(block, idx) * vi;
                }
            }
        }
    }
    Ok(())
}

#[inline]
#[allow(dead_code)]
fn q4_avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
#[allow(dead_code)]
fn q4_avx512_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
#[allow(dead_code)]
fn accumulate_q4_block(
    bitstream: &[u8],
    factor: f32,
    output: &mut [f32],
    use_avx512: bool,
    use_avx2: bool,
) {
    debug_assert!(bitstream.len() >= QK_K / 2);
    debug_assert!(output.len() >= QK_K);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_avx512 {
        unsafe {
            accumulate_q4_block_avx512(bitstream.as_ptr(), factor, output.as_mut_ptr());
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_avx2 {
        unsafe {
            accumulate_q4_block_avx2(bitstream.as_ptr(), factor, output.as_mut_ptr());
        }
        return;
    }

    for (pair_idx, &packed) in bitstream.iter().take(QK_K / 2).enumerate() {
        let out_idx = pair_idx * 2;
        output[out_idx] += ((packed & 0x0F) as f32 - 8.0) * factor;
        output[out_idx + 1] += ((packed >> 4) as f32 - 8.0) * factor;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(dead_code)]
unsafe fn accumulate_q4_block_avx512(bitstream: *const u8, factor: f32, output: *mut f32) {
    let mask = _mm_set1_epi8(0x0F);
    let zero_point = _mm_set1_epi8(8);
    let factor = _mm512_set1_ps(factor);

    for byte_offset in (0..QK_K / 2).step_by(16) {
        let packed = unsafe { _mm_loadu_si128(bitstream.add(byte_offset).cast::<__m128i>()) };
        let low = _mm_and_si128(packed, mask);
        let high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
        let interleaved_lo = _mm_sub_epi8(_mm_unpacklo_epi8(low, high), zero_point);
        let interleaved_hi = _mm_sub_epi8(_mm_unpackhi_epi8(low, high), zero_point);

        let groups = [interleaved_lo, interleaved_hi];
        for (group_idx, group) in groups.into_iter().enumerate() {
            let q_i32 = _mm512_cvtepi8_epi32(group);
            let q_f32 = _mm512_cvtepi32_ps(q_i32);
            let out_ptr = unsafe { output.add(byte_offset * 2 + group_idx * 16) };
            let current = unsafe { _mm512_loadu_ps(out_ptr) };
            let updated = _mm512_add_ps(current, _mm512_mul_ps(q_f32, factor));
            unsafe { _mm512_storeu_ps(out_ptr, updated) };
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(dead_code)]
unsafe fn accumulate_q4_block_avx2(bitstream: *const u8, factor: f32, output: *mut f32) {
    let mask = _mm_set1_epi8(0x0F);
    let zero_point = _mm_set1_epi8(8);
    let factor = _mm256_set1_ps(factor);

    for byte_offset in (0..QK_K / 2).step_by(16) {
        let packed = unsafe { _mm_loadu_si128(bitstream.add(byte_offset).cast::<__m128i>()) };
        let low = _mm_and_si128(packed, mask);
        let high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
        let interleaved_lo = _mm_sub_epi8(_mm_unpacklo_epi8(low, high), zero_point);
        let interleaved_hi = _mm_sub_epi8(_mm_unpackhi_epi8(low, high), zero_point);

        let groups = [
            interleaved_lo,
            _mm_srli_si128(interleaved_lo, 8),
            interleaved_hi,
            _mm_srli_si128(interleaved_hi, 8),
        ];
        for (group_idx, group) in groups.into_iter().enumerate() {
            let q_i32 = _mm256_cvtepi8_epi32(group);
            let q_f32 = _mm256_cvtepi32_ps(q_i32);
            let out_ptr = unsafe { output.add(byte_offset * 2 + group_idx * 8) };
            let current = unsafe { _mm256_loadu_ps(out_ptr) };
            let updated = _mm256_add_ps(current, _mm256_mul_ps(q_f32, factor));
            unsafe { _mm256_storeu_ps(out_ptr, updated) };
        }
    }
}

fn gemv_q8_0_f32_transposed(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK8_0;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q8_0_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != rows {
        return Err(GemvError::InvalidVectorLength {
            expected: rows,
            actual: vector.len(),
        });
    }
    if output.len() != cols {
        return Err(GemvError::InvalidOutputLength {
            expected: cols,
            actual: output.len(),
        });
    }

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_chunks_mut(TRANSPOSED_GEMV_COL_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let col_start = chunk_idx * TRANSPOSED_GEMV_COL_CHUNK;
                let block_start = col_start / QK8_0;
                let block_end = ((col_start + out_chunk.len()).div_ceil(QK8_0)).min(blocks_per_row);
                out_chunk.fill(0.0);
                for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
                    if vi == 0.0 {
                        continue;
                    }
                    let row_start = row_idx * blocks_per_row * BLOCK_Q8_0_SIZE;
                    let row_blocks = &quantized_matrix
                        [row_start..row_start + (blocks_per_row * BLOCK_Q8_0_SIZE)];
                    for block_idx in block_start..block_end {
                        let block_offset = block_idx * BLOCK_Q8_0_SIZE;
                        let block = &row_blocks[block_offset..block_offset + BLOCK_Q8_0_SIZE];
                        let scale = f16_le_to_f32([block[0], block[1]]);
                        let factor = scale * vi;
                        let local_col = block_idx * QK8_0 - col_start;
                        let block_output_len = (out_chunk.len() - local_col).min(QK8_0);
                        let out_block = &mut out_chunk[local_col..local_col + block_output_len];
                        for (out, q_byte) in out_block.iter_mut().zip(block[2..].iter()) {
                            *out += (*q_byte as i8) as f32 * factor;
                        }
                    }
                }
            });
    } else {
        output.fill(0.0);
        for (row_idx, &vi) in vector.iter().enumerate().take(rows) {
            let row_start = row_idx * blocks_per_row * BLOCK_Q8_0_SIZE;
            let row_blocks =
                &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q8_0_SIZE)];
            for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q8_0_SIZE).enumerate() {
                let scale = f16_le_to_f32([block[0], block[1]]);
                let col_start = block_idx * QK8_0;
                for (idx, q_byte) in block[2..].iter().enumerate() {
                    if col_start + idx >= cols {
                        break;
                    }
                    output[col_start + idx] += (*q_byte as i8) as f32 * scale * vi;
                }
            }
        }
    }
    Ok(())
}

pub fn gemv_quantized_f32_transposed(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    match quantization {
        GgufQuantizationType::Q8_0 => {
            gemv_q8_0_f32_transposed(quantized_matrix, rows, cols, vector, output)
        }
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
            gemv_q4_k_f32_transposed(quantized_matrix, rows, cols, vector, output)
        }
        GgufQuantizationType::Q6_K => {
            gemv_q6_k_f32_transposed(quantized_matrix, rows, cols, vector, output)
        }
        _ => Err(GemvError::UnsupportedQuantizationType { quantization }),
    }
}

pub fn f16_le_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    f16_bits_to_f32(bits)
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;

    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut frac_norm = frac;
            let mut e = -14_i32;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                e -= 1;
            }
            frac_norm &= 0x03FF;
            (sign << 31) | (((e + 127) as u32) << 23) | (frac_norm << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F80_0000 | (frac << 13)
    } else {
        let e = exp as i32 - 15 + 127;
        (sign << 31) | ((e as u32) << 23) | (frac << 13)
    };

    f32::from_bits(f32_bits)
}

pub fn gemm_f32(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    output: &mut [f32],
) -> Result<(), GemmError> {
    let expected_left_len = rows.saturating_mul(shared_dim);
    if left_matrix.len() != expected_left_len {
        return Err(GemmError::InvalidLeftMatrixLength {
            expected: expected_left_len,
            actual: left_matrix.len(),
        });
    }

    let expected_right_len = shared_dim.saturating_mul(cols);
    if right_matrix.len() != expected_right_len {
        return Err(GemmError::InvalidRightMatrixLength {
            expected: expected_right_len,
            actual: right_matrix.len(),
        });
    }

    let expected_output_len = rows.saturating_mul(cols);
    if output.len() != expected_output_len {
        return Err(GemmError::InvalidOutputLength {
            expected: expected_output_len,
            actual: output.len(),
        });
    }

    #[cfg(feature = "cuda")]
    if crate::cuda::cuda_build_info().detected_at_build {
        return crate::cuda::gemm_f32_cuda(
            left_matrix,
            rows,
            shared_dim,
            right_matrix,
            cols,
            output,
        )
        .map_err(|err| GemmError::Cuda(format!("{err:?}")));
    }

    #[cfg(feature = "webgpu")]
    if crate::webgpu::should_use_webgpu_gemm(rows, shared_dim, cols) {
        crate::webgpu::validate_gemm_dims(
            left_matrix,
            rows,
            shared_dim,
            right_matrix,
            cols,
            output,
        )
        .map_err(|err| GemmError::WebGpu(format!("WebGPU GEMM validation failed: {err:?}")))?;
        gemm_f32_cpu(left_matrix, rows, shared_dim, right_matrix, cols, output);
        return Ok(());
    }

    #[cfg(feature = "metal")]
    if crate::metal::should_use_mps_gemm(rows, shared_dim, cols) {
        crate::metal::validate_gemm_dims(left_matrix, rows, shared_dim, right_matrix, cols, output)
            .map_err(|err| GemmError::Metal(format!("MPS GEMM validation failed: {err:?}")))?;
        gemm_f32_cpu(left_matrix, rows, shared_dim, right_matrix, cols, output);
        return Ok(());
    }

    let shard_count = choose_tensor_parallel_shard_count(shared_dim);
    if shard_count > 1 {
        gemm_f32_tensor_parallel(
            left_matrix,
            rows,
            shared_dim,
            right_matrix,
            cols,
            shard_count,
            output,
        )?;
        return Ok(());
    }

    gemm_f32_cpu(left_matrix, rows, shared_dim, right_matrix, cols, output);
    Ok(())
}

pub fn gemm_f32_tensor_parallel(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    shard_count: usize,
    output: &mut [f32],
) -> Result<(), GemmError> {
    validate_tensor_parallel_dims(left_matrix, rows, shared_dim, right_matrix, cols, output)?;
    if shard_count == 0 || !shared_dim.is_multiple_of(shard_count) {
        return Err(GemmError::InvalidTensorParallelShardCount {
            shared_dim,
            shard_count,
        });
    }
    if shard_count == 1 {
        gemm_f32_cpu(left_matrix, rows, shared_dim, right_matrix, cols, output);
        return Ok(());
    }

    let partials = std::thread::scope(|scope| {
        let mut jobs = Vec::with_capacity(shard_count);
        let chunk = shared_dim / shard_count;
        for shard_idx in 0..shard_count {
            let start_k = shard_idx * chunk;
            let end_k = start_k + chunk;
            jobs.push(scope.spawn(move || {
                let mut partial = vec![0.0_f32; rows * cols];
                for row in 0..rows {
                    let out_row = &mut partial[row * cols..(row + 1) * cols];
                    for k in start_k..end_k {
                        let left = left_matrix[row * shared_dim + k];
                        let right_row = &right_matrix[k * cols..(k + 1) * cols];
                        for (col, out_cell) in out_row.iter_mut().enumerate() {
                            *out_cell += left * right_row[col];
                        }
                    }
                }
                partial
            }));
        }
        jobs.into_iter()
            .map(|job| job.join().expect("tensor-parallel worker should not panic"))
            .collect::<Vec<_>>()
    });

    output.fill(0.0);
    for partial in partials {
        for (out, value) in output.iter_mut().zip(partial.iter()) {
            *out += *value;
        }
    }
    Ok(())
}

fn validate_tensor_parallel_dims(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    output: &[f32],
) -> Result<(), GemmError> {
    let expected_left_len = rows.saturating_mul(shared_dim);
    if left_matrix.len() != expected_left_len {
        return Err(GemmError::InvalidLeftMatrixLength {
            expected: expected_left_len,
            actual: left_matrix.len(),
        });
    }

    let expected_right_len = shared_dim.saturating_mul(cols);
    if right_matrix.len() != expected_right_len {
        return Err(GemmError::InvalidRightMatrixLength {
            expected: expected_right_len,
            actual: right_matrix.len(),
        });
    }

    let expected_output_len = rows.saturating_mul(cols);
    if output.len() != expected_output_len {
        return Err(GemmError::InvalidOutputLength {
            expected: expected_output_len,
            actual: output.len(),
        });
    }
    Ok(())
}

fn choose_tensor_parallel_shard_count(shared_dim: usize) -> usize {
    const MIN_SHARED_DIM_FOR_TP: usize = 1024;
    if shared_dim < MIN_SHARED_DIM_FOR_TP {
        return 1;
    }
    let max_threads = std::thread::available_parallelism().map_or(1, usize::from);
    let max_shards = max_threads.min(8).min(shared_dim);
    for shards in (2..=max_shards).rev() {
        if shared_dim.is_multiple_of(shards) {
            return shards;
        }
    }
    1
}

pub fn gemm_i8(
    left_matrix: &[i8],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[i8],
    cols: usize,
    output: &mut [i32],
) -> Result<(), GemmError> {
    let expected_left_len = rows.saturating_mul(shared_dim);
    if left_matrix.len() != expected_left_len {
        return Err(GemmError::InvalidLeftMatrixLength {
            expected: expected_left_len,
            actual: left_matrix.len(),
        });
    }

    let expected_right_len = shared_dim.saturating_mul(cols);
    if right_matrix.len() != expected_right_len {
        return Err(GemmError::InvalidRightMatrixLength {
            expected: expected_right_len,
            actual: right_matrix.len(),
        });
    }

    let expected_output_len = rows.saturating_mul(cols);
    if output.len() != expected_output_len {
        return Err(GemmError::InvalidOutputLength {
            expected: expected_output_len,
            actual: output.len(),
        });
    }

    gemm_i8_cpu(left_matrix, rows, shared_dim, right_matrix, cols, output);
    Ok(())
}

pub fn gemm_i4(
    left_matrix_packed: &[u8],
    rows: usize,
    shared_dim: usize,
    right_matrix_packed: &[u8],
    cols: usize,
    output: &mut [i32],
) -> Result<(), GemmError> {
    let expected_left_values = rows.saturating_mul(shared_dim);
    let expected_left_len = expected_left_values.div_ceil(2);
    if left_matrix_packed.len() != expected_left_len {
        return Err(GemmError::InvalidLeftMatrixLength {
            expected: expected_left_len,
            actual: left_matrix_packed.len(),
        });
    }

    let expected_right_values = shared_dim.saturating_mul(cols);
    let expected_right_len = expected_right_values.div_ceil(2);
    if right_matrix_packed.len() != expected_right_len {
        return Err(GemmError::InvalidRightMatrixLength {
            expected: expected_right_len,
            actual: right_matrix_packed.len(),
        });
    }

    let expected_output_len = rows.saturating_mul(cols);
    if output.len() != expected_output_len {
        return Err(GemmError::InvalidOutputLength {
            expected: expected_output_len,
            actual: output.len(),
        });
    }

    gemm_i4_cpu(
        left_matrix_packed,
        rows,
        shared_dim,
        right_matrix_packed,
        cols,
        output,
    );
    Ok(())
}

fn gemv_f32_cpu(matrix: &[f32], cols: usize, vector: &[f32], output: &mut [f32]) {
    // dot_f32_fast (AVX2 FMA, independent accumulators) rather than a scalar
    // iterator sum: LLVM cannot vectorize the f32 reduction (non-associative),
    // leaving a 4-cycle-latency serial FMA chain. The MoE router GEMV runs
    // through here every layer of every token — measured ~24 ms/token of
    // main-thread stall on Qwen3-30B before this change.
    let rows = output.len();
    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        matrix
            .par_chunks_exact(cols)
            .zip(output.par_iter_mut())
            .for_each(|(row_values, out)| {
                *out = dot_f32_fast(row_values, &vector[..cols]);
            });
    } else {
        for (row_values, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
            *out = dot_f32_fast(row_values, &vector[..cols]);
        }
    }
}

pub fn linear_activation_f32(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    activation: ActivationFn,
    output: &mut [f32],
) -> Result<(), LinearActivationError> {
    let expected_matrix_len = rows.saturating_mul(cols);
    if matrix.len() != expected_matrix_len {
        return Err(LinearActivationError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(LinearActivationError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(LinearActivationError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    for (row_values, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
        let linear = row_values
            .iter()
            .zip(vector.iter())
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        *out = activate(linear, activation);
    }

    Ok(())
}

fn activate(value: f32, activation: ActivationFn) -> f32 {
    match activation {
        ActivationFn::Relu => value.max(0.0),
        ActivationFn::Gelu => {
            let k = (2.0_f32 / std::f32::consts::PI).sqrt();
            0.5 * value * (1.0 + (k * (value + 0.044_715 * value.powi(3))).tanh())
        }
        ActivationFn::Silu => {
            let sigmoid = 1.0_f32 / (1.0 + (-value).exp());
            value * sigmoid
        }
    }
}

fn gemm_f32_cpu(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    output: &mut [f32],
) {
    let expected_right_len = shared_dim.saturating_mul(cols);
    let mut right_transposed = vec![0.0_f32; expected_right_len];
    for shared_idx in 0..shared_dim {
        let row_start = shared_idx * cols;
        let row_end = row_start + cols;
        let right_row = &right_matrix[row_start..row_end];
        for (col, value) in right_row.iter().enumerate() {
            right_transposed[col * shared_dim + shared_idx] = *value;
        }
    }

    for row in 0..rows {
        let left_row = &left_matrix[row * shared_dim..(row + 1) * shared_dim];
        let out_row = &mut output[row * cols..(row + 1) * cols];
        for (col, out_cell) in out_row.iter_mut().enumerate() {
            // `right_transposed` makes each column contiguous, so this is a
            // plain dot product — dispatch to the SIMD kernel (AVX-512/AVX2)
            // instead of a scalar loop fenced by a `black_box` "prefetch" that
            // also blocked autovectorization.
            let right_col = &right_transposed[col * shared_dim..(col + 1) * shared_dim];
            *out_cell = crate::flash_attention::dot_product_f32(left_row, right_col);
        }
    }
}

fn gemm_i8_cpu(
    left_matrix: &[i8],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[i8],
    cols: usize,
    output: &mut [i32],
) {
    let expected_right_len = shared_dim.saturating_mul(cols);
    let mut right_transposed = vec![0_i8; expected_right_len];
    for shared_idx in 0..shared_dim {
        let row_start = shared_idx * cols;
        let row_end = row_start + cols;
        let right_row = &right_matrix[row_start..row_end];
        for (col, value) in right_row.iter().enumerate() {
            right_transposed[col * shared_dim + shared_idx] = *value;
        }
    }

    for row in 0..rows {
        let left_row = &left_matrix[row * shared_dim..(row + 1) * shared_dim];
        let out_row = &mut output[row * cols..(row + 1) * cols];
        for (col, out_cell) in out_row.iter_mut().enumerate() {
            let right_col = &right_transposed[col * shared_dim..(col + 1) * shared_dim];
            *out_cell = left_row
                .iter()
                .zip(right_col.iter())
                .map(|(l, r)| i32::from(*l) * i32::from(*r))
                .sum();
        }
    }
}

fn gemm_i4_cpu(
    left_matrix_packed: &[u8],
    rows: usize,
    shared_dim: usize,
    right_matrix_packed: &[u8],
    cols: usize,
    output: &mut [i32],
) {
    for row in 0..rows {
        let out_row = &mut output[row * cols..(row + 1) * cols];
        for (col, out_cell) in out_row.iter_mut().enumerate() {
            let mut sum = 0_i32;
            for k in 0..shared_dim {
                let left_idx = row * shared_dim + k;
                let right_idx = k * cols + col;
                sum += unpack_i4(left_matrix_packed, left_idx)
                    * unpack_i4(right_matrix_packed, right_idx);
            }
            *out_cell = sum;
        }
    }
}

fn unpack_i4(packed: &[u8], value_index: usize) -> i32 {
    let byte = packed[value_index / 2];
    let nibble = if value_index.is_multiple_of(2) {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    };
    i32::from(nibble as i8 - 8)
}

pub fn scaled_dot_product_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    seq_len: usize,
    dim: usize,
    output: &mut [f32],
) -> Result<(), AttentionError> {
    if query.len() != dim {
        return Err(AttentionError::InvalidQueryLength {
            expected: dim,
            actual: query.len(),
        });
    }

    let expected_kv_len = seq_len.saturating_mul(dim);
    if key.len() != expected_kv_len {
        return Err(AttentionError::InvalidKeyLength {
            expected: expected_kv_len,
            actual: key.len(),
        });
    }
    if value.len() != expected_kv_len {
        return Err(AttentionError::InvalidValueLength {
            expected: expected_kv_len,
            actual: value.len(),
        });
    }
    if output.len() != dim {
        return Err(AttentionError::InvalidOutputLength {
            expected: dim,
            actual: output.len(),
        });
    }

    output.fill(0.0);
    if seq_len == 0 {
        return Ok(());
    }

    let scale = 1.0_f32 / (dim as f32).sqrt();
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0_f32;
    let mut token_offset = 0_usize;
    let mut block_acc = vec![0.0_f32; dim];
    let mut block_scores = vec![0.0_f32; FLASH_ATTENTION_BLOCK_TOKENS];

    while token_offset < seq_len {
        let block_len = (seq_len - token_offset).min(FLASH_ATTENTION_BLOCK_TOKENS);
        let block_start = token_offset * dim;
        let block_end = block_start + block_len * dim;
        let key_block = &key[block_start..block_end];
        let value_block = &value[block_start..block_end];

        let mut block_max = f32::NEG_INFINITY;
        for (idx, key_row) in key_block.chunks_exact(dim).enumerate() {
            let score = query
                .iter()
                .zip(key_row.iter())
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * scale;
            block_scores[idx] = score;
            block_max = block_max.max(score);
        }

        block_acc.fill(0.0);
        let mut block_sum = 0.0_f32;
        for (idx, value_row) in value_block.chunks_exact(dim).enumerate() {
            let score = block_scores[idx];
            let weight = (score - block_max).exp();
            block_sum += weight;
            for (acc, v) in block_acc.iter_mut().zip(value_row.iter()) {
                *acc += weight * v;
            }
        }

        let merged_max = running_max.max(block_max);
        let running_scale = (running_max - merged_max).exp();
        let block_scale = (block_max - merged_max).exp();
        for (out, acc) in output.iter_mut().zip(block_acc.iter()) {
            *out = *out * running_scale + acc * block_scale;
        }
        running_sum = running_sum * running_scale + block_sum * block_scale;
        running_max = merged_max;
        token_offset += block_len;
    }

    let inv_sum = 1.0_f32 / running_sum;
    for out in output.iter_mut() {
        *out *= inv_sum;
    }

    Ok(())
}

/// Apply rotary position embedding in NeoX/HF style (split-half), matching
/// llama.cpp's `GGML_ROPE_TYPE_NEOX` mode used by Llama, Mistral, Qwen, Gemma,
/// Phi and most other modern decoder-only transformers. Pairs index `i` with
/// index `head_dim/2 + i` rather than `2i` with `2i+1` (which is the GPT-J /
/// "norm" mode and produces garbage on NeoX-trained weights).
pub fn apply_rope_f32(
    input: &[f32],
    position: usize,
    head_dim: usize,
    theta: f32,
    output: &mut [f32],
) -> Result<(), RopeError> {
    if input.len() != head_dim {
        return Err(RopeError::InvalidInputLength {
            expected: head_dim,
            actual: input.len(),
        });
    }
    if output.len() != head_dim {
        return Err(RopeError::InvalidOutputLength {
            expected: head_dim,
            actual: output.len(),
        });
    }
    if !head_dim.is_multiple_of(2) {
        return Err(RopeError::OddHeadDim { head_dim });
    }

    if position == 0 {
        output.copy_from_slice(input);
        return Ok(());
    }

    let position_f = position as f32;
    let half_dim = head_dim / 2;
    let inv_head_dim = 1.0_f32 / head_dim as f32;
    // freq_i = theta^(-(2*i)/head_dim). Computing `powf` for every pair is
    // surprisingly expensive in prompt processing (tokens × heads × layers).
    // Use the geometric recurrence instead: freq_{i+1} = freq_i * base.
    let freq_multiplier = theta.powf(-2.0 * inv_head_dim);
    let mut freq = 1.0_f32;

    for i in 0..half_dim {
        let x0 = input[i];
        let x1 = input[half_dim + i];
        let angle = position_f * freq;
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        output[i] = x0 * cos_angle - x1 * sin_angle;
        output[half_dim + i] = x0 * sin_angle + x1 * cos_angle;
        freq *= freq_multiplier;
    }

    Ok(())
}

/// AVX2 SwiGLU kernel. Uses a bit-trick fast-exp approximation for sigmoid
/// (~1% error vs true sigmoid, negligible for inference quality).
///
/// sigmoid(x) ≈ 1/(1+exp(-x)) where exp(-x) ≈ 2^(-x·log₂e) via IEEE 754 cast.
/// Followed by one Newton-Raphson step on the reciprocal for better precision.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn swiglu_avx2(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = output.len();
    let chunks = n / 8;

    // SAFETY: All AVX2 intrinsics are unsafe; we are inside an `unsafe fn`
    // that is only called after runtime feature detection confirms AVX2+FMA.
    unsafe {
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let scale23 = _mm256_set1_ps((1u32 << 23) as f32);
        let _bias23 = _mm256_set1_ps((127u32 << 23) as f32);
        let one = _mm256_set1_ps(1.0_f32);
        let two = _mm256_set1_ps(2.0_f32);
        let zero = _mm256_setzero_ps();
        let clamp_hi = _mm256_set1_ps(12.0_f32);
        let clamp_lo = _mm256_set1_ps(-12.0_f32);

        for i in 0..chunks {
            let g = _mm256_loadu_ps(gate[i * 8..].as_ptr());
            let u = _mm256_loadu_ps(up[i * 8..].as_ptr());

            // neg_g clamped to [-12, 12] to keep 2^(neg_g·log2e) in valid i32 range
            let neg_g = _mm256_sub_ps(zero, g);
            let neg_g = _mm256_max_ps(_mm256_min_ps(neg_g, clamp_hi), clamp_lo);

            // exp(-g) ≈ 2^(neg_g·log₂e): multiply, add the float-domain bias 127·2²³,
            // then reinterpret the bits as f32 (standard Schraudolph approximation).
            let exp_arg = _mm256_fmadd_ps(neg_g, log2e, _mm256_set1_ps(127.0_f32));
            let exp_bits = _mm256_cvtps_epi32(_mm256_mul_ps(exp_arg, scale23));
            let exp_neg_g = _mm256_castsi256_ps(exp_bits);

            // sigmoid = 1 / (1 + exp(-g))
            let denom = _mm256_add_ps(one, exp_neg_g);
            // rcp + one Newton-Raphson step: r2 = r·(2 - denom·r)
            let rcp = _mm256_rcp_ps(denom);
            let sigmoid = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));

            // silu(g)·up = g·sigmoid(g)·up
            let result = _mm256_mul_ps(_mm256_mul_ps(g, sigmoid), u);
            _mm256_storeu_ps(output[i * 8..].as_mut_ptr(), result);
        }
    }

    // Scalar tail for remainder elements
    for i in (chunks * 8)..n {
        let g = gate[i];
        let sigmoid = 1.0_f32 / (1.0_f32 + (-g).exp());
        output[i] = g * sigmoid * up[i];
    }
}

/// SwiGLU written back into `gate`: `gate[i] = gate[i] * sigmoid(gate[i]) * up[i]`.
/// Uses an AVX2 inplace fast path when available to avoid scalar `exp()` overhead.
pub fn apply_swiglu_inplace_f32(gate: &mut [f32], up: &[f32]) {
    let n = gate.len().min(up.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: gate_ptr and up_ptr point to non-overlapping slices (distinct
        // allocations).  Within each iteration we load 8 gate values into a
        // register, compute silu, then store — no aliased concurrent access.
        unsafe { swiglu_avx2_inplace(gate.as_mut_ptr(), up.as_ptr(), n) };
        return;
    }
    for i in 0..n {
        let g = gate[i];
        let sigmoid = 1.0_f32 / (1.0_f32 + (-g).exp());
        gate[i] = g * sigmoid * up[i];
    }
}

/// AVX2 SwiGLU inplace: reads gate[i], computes silu(gate[i])*up[i], writes back.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn swiglu_avx2_inplace(g_ptr: *mut f32, u_ptr: *const f32, n: usize) {
    let chunks = n / 8;
    // SAFETY: All AVX2 intrinsics are unsafe; we are inside an `unsafe fn`
    // that is only called after runtime feature detection confirms AVX2+FMA.
    unsafe {
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let scale23 = _mm256_set1_ps((1u32 << 23) as f32);
        let one = _mm256_set1_ps(1.0_f32);
        let two = _mm256_set1_ps(2.0_f32);
        let zero = _mm256_setzero_ps();
        let clamp_hi = _mm256_set1_ps(12.0_f32);
        let clamp_lo = _mm256_set1_ps(-12.0_f32);

        for i in 0..chunks {
            let off = i * 8;
            let g = _mm256_loadu_ps(g_ptr.add(off));
            let u = _mm256_loadu_ps(u_ptr.add(off));

            let neg_g = _mm256_sub_ps(zero, g);
            let neg_g = _mm256_max_ps(_mm256_min_ps(neg_g, clamp_hi), clamp_lo);
            let exp_arg = _mm256_fmadd_ps(neg_g, log2e, _mm256_set1_ps(127.0_f32));
            let exp_bits = _mm256_cvtps_epi32(_mm256_mul_ps(exp_arg, scale23));
            let exp_neg_g = _mm256_castsi256_ps(exp_bits);
            let denom = _mm256_add_ps(one, exp_neg_g);
            let rcp = _mm256_rcp_ps(denom);
            let sigmoid = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));
            let result = _mm256_mul_ps(_mm256_mul_ps(g, sigmoid), u);
            _mm256_storeu_ps(g_ptr.add(off), result);
        }
        for i in (chunks * 8)..n {
            let g = *g_ptr.add(i);
            let sigmoid = 1.0_f32 / (1.0_f32 + (-g).exp());
            *g_ptr.add(i) = g * sigmoid * *u_ptr.add(i);
        }
    }
}

/// GeGLU written back into `gate`: `gate[i] = gelu(gate[i]) * up[i]` using the
/// tanh GELU approximation (matches Gemma's `gelu_pytorch_tanh` activation).
/// Scalar implementation — Gemma FFN width is modest and this is not the
/// dominant cost. Kept separate from SwiGLU so SiLU models are untouched.
pub fn apply_geglu_inplace_f32(gate: &mut [f32], up: &[f32]) {
    const K: f32 = 0.797_884_6_f32; // sqrt(2/pi)
    let n = gate.len().min(up.len());
    for i in 0..n {
        let g = gate[i];
        let gelu = 0.5 * g * (1.0 + (K * (g + 0.044_715 * g * g * g)).tanh());
        gate[i] = gelu * up[i];
    }
}

pub fn apply_swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<(), SwiGluError> {
    let expected_len = output.len();
    if gate.len() != expected_len {
        return Err(SwiGluError::InvalidGateLength {
            expected: expected_len,
            actual: gate.len(),
        });
    }
    if up.len() != expected_len {
        return Err(SwiGluError::InvalidUpLength {
            expected: expected_len,
            actual: up.len(),
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { swiglu_avx2(gate, up, output) };
        return Ok(());
    }

    for ((gate_value, up_value), out) in gate.iter().zip(up.iter()).zip(output.iter_mut()) {
        let sigmoid = 1.0_f32 / (1.0 + (-gate_value).exp());
        *out = gate_value * sigmoid * up_value;
    }

    Ok(())
}

pub fn rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    let hidden_dim = output.len();
    if hidden_dim == 0 {
        return Err(RmsNormError::ZeroDimension);
    }
    if input.len() != hidden_dim {
        return Err(RmsNormError::InvalidInputLength {
            expected: hidden_dim,
            actual: input.len(),
        });
    }
    if weight.len() != hidden_dim {
        return Err(RmsNormError::InvalidWeightLength {
            expected: hidden_dim,
            actual: weight.len(),
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { rms_norm_f32_avx2(input, weight, eps, output) };
            return Ok(());
        }
    }

    let sum_sq = input.iter().map(|value| value * value).sum::<f32>();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    for ((value, scale), out) in input.iter().zip(weight.iter()).zip(output.iter_mut()) {
        *out = value * inv_rms * scale;
    }
    Ok(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn rms_norm_f32_avx2(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let len = output.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 32 <= len {
        let v0 = _mm256_loadu_ps(input.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(input.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(input.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(input.as_ptr().add(i + 24));
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        acc2 = _mm256_fmadd_ps(v2, v2, acc2);
        acc3 = _mm256_fmadd_ps(v3, v3, acc3);
        i += 32;
    }
    while i + 8 <= len {
        let v = _mm256_loadu_ps(input.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(v, v, acc0);
        i += 8;
    }

    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let mut sum_sq = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));

    while i < len {
        let v = *input.get_unchecked(i);
        sum_sq += v * v;
        i += 1;
    }

    let inv_rms = 1.0_f32 / (sum_sq / len as f32 + eps).sqrt();
    let inv = _mm256_set1_ps(inv_rms);
    let mut j = 0;
    while j + 32 <= len {
        let v0 = _mm256_loadu_ps(input.as_ptr().add(j));
        let w0 = _mm256_loadu_ps(weight.as_ptr().add(j));
        let v1 = _mm256_loadu_ps(input.as_ptr().add(j + 8));
        let w1 = _mm256_loadu_ps(weight.as_ptr().add(j + 8));
        let v2 = _mm256_loadu_ps(input.as_ptr().add(j + 16));
        let w2 = _mm256_loadu_ps(weight.as_ptr().add(j + 16));
        let v3 = _mm256_loadu_ps(input.as_ptr().add(j + 24));
        let w3 = _mm256_loadu_ps(weight.as_ptr().add(j + 24));
        _mm256_storeu_ps(
            output.as_mut_ptr().add(j),
            _mm256_mul_ps(_mm256_mul_ps(v0, inv), w0),
        );
        _mm256_storeu_ps(
            output.as_mut_ptr().add(j + 8),
            _mm256_mul_ps(_mm256_mul_ps(v1, inv), w1),
        );
        _mm256_storeu_ps(
            output.as_mut_ptr().add(j + 16),
            _mm256_mul_ps(_mm256_mul_ps(v2, inv), w2),
        );
        _mm256_storeu_ps(
            output.as_mut_ptr().add(j + 24),
            _mm256_mul_ps(_mm256_mul_ps(v3, inv), w3),
        );
        j += 32;
    }
    while j + 8 <= len {
        let v = _mm256_loadu_ps(input.as_ptr().add(j));
        let w = _mm256_loadu_ps(weight.as_ptr().add(j));
        _mm256_storeu_ps(
            output.as_mut_ptr().add(j),
            _mm256_mul_ps(_mm256_mul_ps(v, inv), w),
        );
        j += 8;
    }
    while j < len {
        *output.get_unchecked_mut(j) = *input.get_unchecked(j) * inv_rms * *weight.get_unchecked(j);
        j += 1;
    }
}

/// Fused RMS-normalization + transposed GEMV for the attention Q projection.
/// Computes RMSNorm of `input`, then performs transposed GEMV using the
/// normalized vector as the GEMV input.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gemv_f32_transposed(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    matrix: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    if input.len() != rows {
        return Err(RmsNormError::InvalidInputLength {
            expected: rows,
            actual: input.len(),
        });
    }
    if weight.len() != rows {
        return Err(RmsNormError::InvalidWeightLength {
            expected: rows,
            actual: weight.len(),
        });
    }
    if matrix.len() != rows * cols {
        return Err(RmsNormError::InvalidOutputLength {
            expected: rows * cols,
            actual: matrix.len(),
        });
    }
    if output.len() != cols {
        return Err(RmsNormError::InvalidOutputLength {
            expected: cols,
            actual: output.len(),
        });
    }
    if rows == 0 {
        return Err(RmsNormError::ZeroDimension);
    }

    let sum_sq = input.iter().map(|v| v * v).sum::<f32>();
    let mean_sq = sum_sq / rows as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    output.fill(0.0);
    for (i, (value, scale)) in input.iter().zip(weight.iter()).enumerate() {
        let scaled = value * inv_rms * scale;
        let row = &matrix[i * cols..(i + 1) * cols];
        for (j, &mat_val) in row.iter().enumerate() {
            output[j] += mat_val * scaled;
        }
    }
    Ok(())
}

pub fn layer_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), LayerNormError> {
    let hidden_dim = output.len();
    if input.len() != hidden_dim {
        return Err(LayerNormError::InvalidInputLength {
            expected: hidden_dim,
            actual: input.len(),
        });
    }
    if weight.len() != hidden_dim {
        return Err(LayerNormError::InvalidWeightLength {
            expected: hidden_dim,
            actual: weight.len(),
        });
    }
    if bias.len() != hidden_dim {
        return Err(LayerNormError::InvalidBiasLength {
            expected: hidden_dim,
            actual: bias.len(),
        });
    }

    let mean = input.iter().sum::<f32>() / hidden_dim as f32;
    let variance = input
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / hidden_dim as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();

    for (((value, scale), shift), out) in input
        .iter()
        .zip(weight.iter())
        .zip(bias.iter())
        .zip(output.iter_mut())
    {
        *out = (value - mean) * inv_std * scale + shift;
    }
    Ok(())
}

pub fn softmax_f32(input: &[f32], output: &mut [f32]) -> Result<(), SoftmaxError> {
    let expected_len = output.len();
    if input.len() != expected_len {
        return Err(SoftmaxError::InvalidInputLength {
            expected: expected_len,
            actual: input.len(),
        });
    }

    let max_input = input.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let mut sum_exp = 0.0_f64;
    for (x, out) in input.iter().zip(output.iter_mut()) {
        let exp_value = (x - max_input).exp();
        *out = exp_value;
        sum_exp += exp_value as f64;
    }

    let inv_sum = (1.0_f64 / sum_exp) as f32;
    for out in output.iter_mut() {
        *out *= inv_sum;
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: DType,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, dtype: DType) -> Self {
        assert_eq!(
            shape.len(),
            strides.len(),
            "shape and strides must have the same rank"
        );
        Self {
            shape,
            strides,
            dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shape/thread/working-set microbenchmark for the Q4_K decode GEMV.
    /// Run with:
    ///   cargo test --release -p oxidize-core --lib -- --ignored --nocapture bench_q4k
    #[test]
    #[ignore]
    fn bench_q4k_gemv_shapes() {
        let shapes: [(usize, usize); 4] = [(9728, 2560), (2560, 9728), (4096, 2560), (1024, 2560)];
        for threads in [1usize, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for &(rows, cols) in &shapes {
                let bpr = cols / QK_K;
                let bytes = rows * bpr * BLOCK_Q4_K_SIZE;
                // 8 copies so the DRAM pass cannot sit in the 16MB L3.
                let copies = 8;
                let weights: Vec<u8> = (0..bytes * copies).map(|i| (i * 37 + 11) as u8).collect();
                let vector: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.001).sin()).collect();
                let mut output = vec![0.0_f32; rows];
                for (label, stride) in [("L3", 0usize), ("DRAM", bytes)] {
                    pool.install(|| {
                        for i in 0..copies {
                            let w = &weights[i * stride..i * stride + bytes];
                            gemv_q4_k_q8_k_fused(w, rows, cols, &vector, &mut output).unwrap();
                        }
                        let iters = 24;
                        let t0 = std::time::Instant::now();
                        for i in 0..iters {
                            let w = &weights[(i % copies) * stride..(i % copies) * stride + bytes];
                            gemv_q4_k_q8_k_fused(w, rows, cols, &vector, &mut output).unwrap();
                        }
                        let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
                        eprintln!(
                            "q4k {rows:>5}x{cols:<5} threads={threads} {label:>4}: {:>7.1}us {:>6.1} GB/s",
                            ns / 1e3,
                            bytes as f64 / ns
                        );
                    });
                }
            }
        }
    }

    /// The fused multi-matrix region must produce bit-identical rows to the
    /// sequential per-matrix GEMVs (same row kernels, same per-row order),
    /// including mixed Q4_K/Q6_K jobs and non-multiple-of-chunk tails.
    #[test]
    fn multi_gemv_matches_sequential_bitwise() {
        let cols = 2560;
        let bpr = cols / QK_K;
        let q4_rows = 96_usize;
        let q6_rows = 61_usize;
        let q4: Vec<u8> = (0..q4_rows * bpr * BLOCK_Q4_K_SIZE)
            .map(|i| (i * 31 + 7) as u8)
            .collect();
        let q6: Vec<u8> = (0..q6_rows * bpr * BLOCK_Q6_K_SIZE)
            .map(|i| (i * 17 + 3) as u8)
            .collect();
        let vector: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.01).sin()).collect();

        let mut seq_q4 = vec![0.0_f32; q4_rows];
        let mut seq_q6 = vec![0.0_f32; q6_rows];
        gemv_quantized_f32(
            GgufQuantizationType::Q4_K_M,
            &q4,
            q4_rows,
            cols,
            &vector,
            &mut seq_q4,
        )
        .unwrap();
        gemv_quantized_f32(
            GgufQuantizationType::Q6_K,
            &q6,
            q6_rows,
            cols,
            &vector,
            &mut seq_q6,
        )
        .unwrap();

        let mut multi_q4 = vec![0.0_f32; q4_rows];
        let mut multi_q6 = vec![0.0_f32; q6_rows];
        let mut jobs = [
            GemvJob {
                quantization: GgufQuantizationType::Q4_K_M,
                matrix: &q4,
                rows: q4_rows,
                output: &mut multi_q4,
            },
            GemvJob {
                quantization: GgufQuantizationType::Q6_K,
                matrix: &q6,
                rows: q6_rows,
                output: &mut multi_q6,
            },
        ];
        gemv_quantized_multi_f32(&mut jobs, cols, &vector).unwrap();

        for (i, (a, b)) in seq_q4.iter().zip(&multi_q4).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "q4 row {i}");
        }
        for (i, (a, b)) in seq_q6.iter().zip(&multi_q6).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "q6 row {i}");
        }
    }

    /// Tolerance for tests that compare CUDA (f16-intermediate) results against
    /// CPU references.  The GPU dequantizes to f16 before GEMV, so a small
    /// round-trip error (~0.01-0.5) is expected and acceptable.
    #[cfg(feature = "cuda")]
    const CUDA_TOL: f32 = 0.5;
    #[cfg(not(feature = "cuda"))]
    const CUDA_TOL: f32 = 1e-4;

    /// Gate A (OXK plan): the oxidize-kernels Q4_K row dots must match the
    /// legacy tensor.rs kernels bit-for-bit (same integer op sequence and f32
    /// combine order), and its Q8_K activation quantizer must be byte-equal.
    #[test]
    #[cfg(all(feature = "oxk", any(target_arch = "x86", target_arch = "x86_64")))]
    fn oxk_q4_k_kernels_match_legacy_exactly() {
        use crate::quantization::{quantize_scalar, quantized_size};
        if !q4_k_q8_k_avx2_available() {
            return;
        }
        let (rows, cols) = (24usize, 512usize);
        let blocks_per_row = cols / QK_K;
        let total = rows * cols;
        let mut bytes = vec![0u8; total * 4];
        for i in 0..total {
            let v = (((i * 31 + 7) % 211) as f32) / 53.0 - 2.0;
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let q_size = quantized_size(GgufQuantizationType::Q4_K_M, total).unwrap();
        let mut q = vec![0u8; q_size];
        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q4_K_M,
            &bytes,
            &mut q,
        )
        .unwrap();
        let input: Vec<f32> = (0..cols)
            .map(|i| (((i * 17 + 3) % 113) as f32) / 29.0 - 1.5)
            .collect();

        // Q8_K quantizer parity (byte-exact).
        let mut q8k_legacy = vec![0u8; blocks_per_row * BLOCK_Q8_K_BYTES];
        quantize_vector_q8_k_into(&input, blocks_per_row, &mut q8k_legacy);
        let mut q8k_oxk = vec![0u8; blocks_per_row * BLOCK_Q8_K_BYTES];
        oxidize_kernels::quantize_q8_k_into(&input, blocks_per_row, &mut q8k_oxk);
        assert_eq!(q8k_legacy, q8k_oxk, "Q8_K quantizer bytes differ");

        let row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;
        // Legacy single-row reference (AVX2 kernel, not VNNI, to pin the exact
        // instruction family OXK replicates; the two are bit-equal anyway).
        let legacy: Vec<f32> = (0..rows)
            .map(|r| unsafe {
                q4_k_q8_k_row_dot_avx2(
                    &q[r * row_bytes..(r + 1) * row_bytes],
                    blocks_per_row,
                    &q8k_legacy,
                )
            })
            .collect();

        // OXK scalar reference vs legacy AVX2: exact.
        for (r, &want) in legacy.iter().enumerate() {
            let got = oxidize_kernels::q4k_q8k_row_dot_scalar(
                &q[r * row_bytes..(r + 1) * row_bytes],
                blocks_per_row,
                &q8k_oxk,
            );
            assert_eq!(got.to_bits(), want.to_bits(), "oxk scalar row {r}");
        }

        // OXK x1 / x4 / x8 vs legacy: exact.
        for (r, &want) in legacy.iter().enumerate() {
            let got = unsafe {
                oxidize_kernels::q4k_q8k_row_dot_avx2(
                    &q[r * row_bytes..(r + 1) * row_bytes],
                    blocks_per_row,
                    &q8k_oxk,
                )
            };
            assert_eq!(got.to_bits(), want.to_bits(), "oxk x1 row {r}");
        }
        let mut quad = [0.0f32; 4];
        unsafe {
            oxidize_kernels::q4k_q8k_row_dot_x4_avx2(
                q.as_ptr(),
                row_bytes,
                blocks_per_row,
                &q8k_oxk,
                &mut quad,
            )
        };
        for (r, &got) in quad.iter().enumerate() {
            assert_eq!(got.to_bits(), legacy[r].to_bits(), "oxk x4 row {r}");
        }
        let mut octet = [0.0f32; 8];
        unsafe {
            oxidize_kernels::q4k_q8k_row_dot_x8_avx2(
                q.as_ptr(),
                row_bytes,
                blocks_per_row,
                &q8k_oxk,
                &mut octet,
            )
        };
        for (r, &got) in octet.iter().enumerate() {
            assert_eq!(got.to_bits(), legacy[r].to_bits(), "oxk x8 row {r}");
        }

        // Range helper over an x8+x4+x1 tail split (24 = 8+8+4+4 tails inside).
        let mut out = vec![0.0f32; rows];
        oxidize_kernels::gemv_q4k_range(&q, blocks_per_row, &q8k_oxk, &mut out);
        for (r, &got) in out.iter().enumerate() {
            assert_eq!(got.to_bits(), legacy[r].to_bits(), "oxk range row {r}");
        }
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn q4_k_x4_kernel_matches_single_row_paths() {
        fn approx_eq(a: f32, b: f32) -> bool {
            (a - b).abs() < 1e-4 || (a.is_nan() && b.is_nan())
        }

        use crate::quantization::{quantize_scalar, quantized_size};
        // rows multiple of 32 exercises the 4-row expert kernel; rows*cols
        // above PARALLEL_GEMV_MIN_OPS exercises the parallel x4 gemv path.
        let (n_experts, rows, cols) = (2usize, 64usize, 256usize);
        let total = n_experts * rows * cols;
        let mut bytes = vec![0u8; total * 4];
        for i in 0..total {
            let v = (((i * 13 + 11) % 103) as f32) / 51.0 - 1.0;
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let q_size = quantized_size(GgufQuantizationType::Q4_K_M, total).unwrap();
        let mut q = vec![0u8; q_size];
        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q4_K_M,
            &bytes,
            &mut q,
        )
        .unwrap();
        let input: Vec<f32> = (0..cols)
            .map(|i| (((i * 7 + 5) % 91) as f32) / 45.0 - 1.0)
            .collect();

        // Reference: per-row single dot via 1-row gemv calls.
        let expert_bytes = q.len() / n_experts;
        let row_bytes = expert_bytes / rows;
        let mut want = vec![0.0f32; n_experts * rows];
        for e in 0..n_experts {
            for r in 0..rows {
                let start = e * expert_bytes + r * row_bytes;
                let mut out = [0.0f32];
                gemv_quantized_f32(
                    GgufQuantizationType::Q4_K_M,
                    &q[start..start + row_bytes],
                    1,
                    cols,
                    &input,
                    &mut out,
                )
                .unwrap();
                want[e * rows + r] = out[0];
            }
        }

        // Expert path (x4 kernel when rows % 32 == 0). Exact equality: the x4
        // kernel performs the identical op sequence per row.
        let selected = [1usize, 0usize];
        let mut batched = vec![0.0f32; selected.len() * rows];
        gemv_quantized_experts_f32(
            GgufQuantizationType::Q4_K_M,
            &q,
            n_experts,
            &selected,
            rows,
            cols,
            &input,
            0,
            &mut batched,
        )
        .unwrap();
        for (slot, &e) in selected.iter().enumerate() {
            for r in 0..rows {
                let got = batched[slot * rows + r];
                let expected = want[e * rows + r];
                assert!(
                    approx_eq(got, expected),
                    "expert x4: slot {slot} expert {e} row {r}: got {got}, want {expected}"
                );
            }
        }

        // Fused gate+up region must match two separate expert GEMV calls
        // exactly (same kernel per row). Reuse the matrix for both halves.
        let mut fused = vec![0.0f32; 2 * selected.len() * rows];
        gemv_quantized_experts_gate_up_f32(
            GgufQuantizationType::Q4_K_M,
            &q,
            &q,
            n_experts,
            &selected,
            rows,
            cols,
            &input,
            &mut fused,
        )
        .unwrap();
        let half = selected.len() * rows;
        for j in 0..half {
            assert!(approx_eq(fused[j], batched[j]), "fused gate j {j}");
            assert!(approx_eq(fused[half + j], batched[j]), "fused up j {j}");
        }

        // Single-vector gemv path on one expert's matrix (serial branch also
        // routes through the x4 run_range helper).
        let mut gemv_out = vec![0.0f32; rows];
        gemv_quantized_f32(
            GgufQuantizationType::Q4_K_M,
            &q[..expert_bytes],
            rows,
            cols,
            &input,
            &mut gemv_out,
        )
        .unwrap();
        for r in 0..rows {
            assert!(approx_eq(gemv_out[r], want[r]), "gemv x4: row {r}");
        }
    }

    #[test]
    fn gemm_vs_gemv_bit_exact_probe() {
        use crate::quantization::{quantize_scalar, quantized_size};
        let (rows, cols, batch) = (64usize, 512usize, 4usize);
        let total = rows * cols;
        let mut bytes = vec![0u8; total * 4];
        for i in 0..total {
            let v = (((i * 37 + 13) % 211) as f32) / 105.0 - 1.0;
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let q_size = quantized_size(GgufQuantizationType::Q4_K_M, total).unwrap();
        let mut q = vec![0u8; q_size];
        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q4_K_M,
            &bytes,
            &mut q,
        )
        .unwrap();
        let mut inputs = vec![0.0f32; batch * cols];
        for (i, x) in inputs.iter_mut().enumerate() {
            *x = (((i * 19 + 7) % 113) as f32) / 56.0 - 1.0;
        }
        let mut gemm_out = vec![0.0f32; batch * rows];
        gemm_quantized_f32(
            GgufQuantizationType::Q4_K_M,
            &q,
            rows,
            cols,
            &inputs,
            &mut gemm_out,
            batch,
        )
        .unwrap();
        let mut mismatches = 0;
        for t in 0..batch {
            let mut gemv_out = vec![0.0f32; rows];
            gemv_quantized_f32(
                GgufQuantizationType::Q4_K_M,
                &q,
                rows,
                cols,
                &inputs[t * cols..(t + 1) * cols],
                &mut gemv_out,
            )
            .unwrap();
            for r in 0..rows {
                if gemm_out[t * rows + r].to_bits() != gemv_out[r].to_bits() {
                    if mismatches < 5 {
                        eprintln!(
                            "t={t} r={r}: gemm={} gemv={} diff={}",
                            gemm_out[t * rows + r],
                            gemv_out[r],
                            gemm_out[t * rows + r] - gemv_out[r]
                        );
                    }
                    mismatches += 1;
                }
            }
        }
        assert_eq!(
            mismatches,
            0,
            "{mismatches} bit mismatches of {}",
            batch * rows
        );
    }

    #[test]
    fn batched_expert_gemv_matches_per_expert_q4_k() {
        use crate::quantization::{quantize_scalar, quantized_size};
        let (n_experts, rows, cols) = (3usize, 4usize, 256usize);
        let total = n_experts * rows * cols;
        let mut bytes = vec![0u8; total * 4];
        for i in 0..total {
            let v = (((i * 31 + 7) % 101) as f32) / 50.0 - 1.0; // spread across [-1, 1]
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let q_size = quantized_size(GgufQuantizationType::Q4_K_M, total).unwrap();
        let mut q = vec![0u8; q_size];
        quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q4_K_M,
            &bytes,
            &mut q,
        )
        .unwrap();
        let expert_bytes = q.len() / n_experts;
        let selected = [2usize, 0usize];

        // Shared input (gate/up): input_stride = 0.
        let input: Vec<f32> = (0..cols)
            .map(|i| (((i * 17 + 3) % 97) as f32) / 48.0 - 1.0)
            .collect();
        let mut batched = vec![0.0f32; selected.len() * rows];
        gemv_quantized_experts_f32(
            GgufQuantizationType::Q4_K_M,
            &q,
            n_experts,
            &selected,
            rows,
            cols,
            &input,
            0,
            &mut batched,
        )
        .unwrap();
        for (slot, &e) in selected.iter().enumerate() {
            let mut want = vec![0.0f32; rows];
            gemv_quantized_f32(
                GgufQuantizationType::Q4_K_M,
                &q[e * expert_bytes..(e + 1) * expert_bytes],
                rows,
                cols,
                &input,
                &mut want,
            )
            .unwrap();
            for r in 0..rows {
                assert!(
                    (batched[slot * rows + r] - want[r]).abs() < 1e-4,
                    "shared slot {slot} e {e} row {r}: batched={} want={}",
                    batched[slot * rows + r],
                    want[r]
                );
            }
        }

        // Per-expert input (down): input_stride = cols.
        let mut inputs = vec![0.0f32; selected.len() * cols];
        for s in 0..selected.len() {
            for i in 0..cols {
                inputs[s * cols + i] = (((i * 23 + s * 5 + 1) % 89) as f32) / 44.0 - 1.0;
            }
        }
        let mut batched2 = vec![0.0f32; selected.len() * rows];
        gemv_quantized_experts_f32(
            GgufQuantizationType::Q4_K_M,
            &q,
            n_experts,
            &selected,
            rows,
            cols,
            &inputs,
            cols,
            &mut batched2,
        )
        .unwrap();
        for (slot, &e) in selected.iter().enumerate() {
            let mut want = vec![0.0f32; rows];
            gemv_quantized_f32(
                GgufQuantizationType::Q4_K_M,
                &q[e * expert_bytes..(e + 1) * expert_bytes],
                rows,
                cols,
                &inputs[slot * cols..(slot + 1) * cols],
                &mut want,
            )
            .unwrap();
            for r in 0..rows {
                assert!((batched2[slot * rows + r] - want[r]).abs() < CUDA_TOL);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn q2_k_dot_avx2_matches_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let mut block = vec![0u8; BLOCK_Q2_K_SIZE];
        let mut state: u32 = 0xCAFE_BABE;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u8
        };
        for b in block.iter_mut() {
            *b = next();
        }
        // Valid f16 d/dmin in finite range.
        block[80] = 0x00;
        block[81] = 0x3c; // d = 1.0
        block[82] = 0x00;
        block[83] = 0x38; // min = 0.5
        let vector: Vec<f32> = (0..QK_K).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let scalar = q2_k_dot_scalar(&block, &vector);
        let simd = unsafe { q2_k_dot_avx2(&block, &vector) };
        assert!(
            (scalar - simd).abs() <= 1e-3 * (1.0 + scalar.abs()),
            "AVX2 Q2_K dot ({simd}) diverged from scalar ({scalar})"
        );
    }

    #[test]
    fn q2_k_dot_scalar_matches_dequant_reference() {
        use crate::quantization::dequantize_q2_k_scalar;
        let mut block = vec![0u8; BLOCK_Q2_K_SIZE];
        let mut state: u32 = 0xDEAD_BEEF;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u8
        };
        for b in block.iter_mut() {
            *b = next();
        }
        block[80] = 0x00;
        block[81] = 0x3c;
        block[82] = 0x00;
        block[83] = 0x38;
        let vector: Vec<f32> = (0..QK_K).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();

        // Reference: full dequant then plain f32 dot.
        let mut dequant = vec![0.0_f32; QK_K];
        dequantize_q2_k_scalar(&block, &mut dequant).expect("dequant ok");
        let reference: f32 = dequant.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
        let fused = q2_k_dot_scalar(&block, &vector);
        assert!(
            (reference - fused).abs() <= 1e-4 * (1.0 + reference.abs()),
            "Q2_K fused dot ({fused}) diverged from dequant-then-dot reference ({reference})"
        );
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn nvfp4_gemv_and_gemm_match_dequant_reference() {
        let rows = 3;
        let blocks_per_row = 2;
        let cols = blocks_per_row * QK_NVFP4;
        let batch = 4;
        let mut weights = vec![0_u8; rows * blocks_per_row * BLOCK_NVFP4_SIZE];
        for (block_idx, block) in weights.chunks_exact_mut(BLOCK_NVFP4_SIZE).enumerate() {
            block[0] = 0x38;
            block[1] = 0x40;
            block[2] = 0x30;
            block[3] = 0x00;
            for (i, q) in block[4..].iter_mut().enumerate() {
                *q = (((i + block_idx) & 0x0f) | ((((15 - (i & 0x0f)) + block_idx) & 0x0f) << 4))
                    as u8;
            }
        }
        let inputs = (0..batch * cols)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03125)
            .collect::<Vec<_>>();

        let mut dequant = vec![0.0_f32; rows * cols];
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let src = (r * blocks_per_row + b) * BLOCK_NVFP4_SIZE;
                let dst = r * cols + b * QK_NVFP4;
                nvfp4_dequantize_block(
                    &weights[src..src + BLOCK_NVFP4_SIZE],
                    &mut dequant[dst..dst + QK_NVFP4],
                );
            }
        }

        let mut gemv = vec![0.0_f32; rows];
        gemv_quantized_f32(
            GgufQuantizationType::NVFP4,
            &weights,
            rows,
            cols,
            &inputs[..cols],
            &mut gemv,
        )
        .expect("nvfp4 gemv succeeds");
        for r in 0..rows {
            let expected: f32 = dequant[r * cols..(r + 1) * cols]
                .iter()
                .zip(inputs[..cols].iter())
                .map(|(a, b)| a * b)
                .sum();
            assert!((gemv[r] - expected).abs() <= 1e-5 * (1.0 + expected.abs()));
        }

        let mut gemm = vec![0.0_f32; batch * rows];
        gemm_quantized_f32(
            GgufQuantizationType::NVFP4,
            &weights,
            rows,
            cols,
            &inputs,
            &mut gemm,
            batch,
        )
        .expect("nvfp4 gemm succeeds");
        for token in 0..batch {
            for r in 0..rows {
                let expected: f32 = dequant[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(inputs[token * cols..(token + 1) * cols].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let actual = gemm[token * rows + r];
                assert!((actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()));
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn q4_k_dot_avx2_matches_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        // Deterministic pseudo-random Q4_K block (BLOCK_Q4_K_SIZE bytes) + vector.
        let mut block = vec![0u8; BLOCK_Q4_K_SIZE];
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u8
        };
        for b in block.iter_mut() {
            *b = next();
        }
        // Use small, valid f16 scale/min so values are finite.
        block[0] = 0x00;
        block[1] = 0x3c; // d = 1.0 in f16
        block[2] = 0x00;
        block[3] = 0x38; // min = 0.5 in f16
        let vector: Vec<f32> = (0..QK_K).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();

        let scalar = q4_k_dot_scalar(&block, &vector);
        let simd = unsafe { q4_k_dot_avx2(&block, &vector) };
        assert!(
            (scalar - simd).abs() <= 1e-3 * (1.0 + scalar.abs()),
            "AVX2 Q4_K dot ({simd}) diverged from scalar ({scalar})"
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn batched_q4_k_q8_k_gemm_matches_repeated_q4_k_gemv() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        let rows = 5;
        let blocks_per_row = 3;
        let cols = blocks_per_row * QK_K;
        let batch = 7;
        let mut weights = vec![0_u8; rows * blocks_per_row * BLOCK_Q4_K_SIZE];
        let mut state: u32 = 0xA11C_E55D;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u8
        };
        for block in weights.chunks_exact_mut(BLOCK_Q4_K_SIZE) {
            block[0] = 0x00;
            block[1] = 0x34; // d = 0.25
            block[2] = 0x00;
            block[3] = 0x28; // min = 0.03125
            for (i, scale) in block[4..16].iter_mut().enumerate() {
                *scale = ((i * 5 + 3) & 0x3f) as u8;
            }
            for byte in &mut block[16..] {
                *byte = next();
            }
        }
        let inputs = (0..batch * cols)
            .map(|i| ((i as f32 * 0.013).cos() * 0.75) + ((i % 7) as f32 - 3.0) * 0.03)
            .collect::<Vec<_>>();

        let mut batched = vec![0.0_f32; batch * rows];
        gemm_quantized_f32(
            GgufQuantizationType::Q4_K_M,
            &weights,
            rows,
            cols,
            &inputs,
            &mut batched,
            batch,
        )
        .expect("batched q4_k gemm should succeed");

        let mut expected = vec![0.0_f32; batch * rows];
        for token in 0..batch {
            gemv_quantized_f32(
                GgufQuantizationType::Q4_K_M,
                &weights,
                rows,
                cols,
                &inputs[token * cols..(token + 1) * cols],
                &mut expected[token * rows..(token + 1) * rows],
            )
            .expect("q4_k gemv should succeed");
        }

        for (actual, reference) in batched.iter().zip(expected.iter()) {
            assert!(
                (actual - reference).abs() <= CUDA_TOL * (1.0 + reference.abs()),
                "batched q4_k gemm value {actual} diverged from repeated gemv {reference}"
            );
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn q4_k_q8_k_row_dot_vnni_matches_avx2() {
        if !q4_k_q8_k_vnni_available() || !q4_k_q8_k_avx2_available() {
            return; // No VNNI hardware in this environment; nothing to compare.
        }

        let blocks_per_row = 4;
        let cols = blocks_per_row * QK_K;
        let mut row = vec![0_u8; blocks_per_row * BLOCK_Q4_K_SIZE];
        let mut state: u32 = 0x1357_9BDF;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u8
        };
        for block in row.chunks_exact_mut(BLOCK_Q4_K_SIZE) {
            block[0] = 0x00;
            block[1] = 0x34; // d = 0.25
            block[2] = 0x00;
            block[3] = 0x28; // min = 0.03125
            for (i, scale) in block[4..16].iter_mut().enumerate() {
                *scale = ((i * 7 + 1) & 0x3f) as u8;
            }
            for byte in &mut block[16..] {
                *byte = next();
            }
        }

        let vector = (0..cols)
            .map(|i| ((i as f32 * 0.021).sin() * 0.6) + ((i % 5) as f32 - 2.0) * 0.05)
            .collect::<Vec<_>>();
        let mut q8k = vec![0_u8; blocks_per_row * BLOCK_Q8_K_BYTES];
        quantize_vector_q8_k_into(&vector, blocks_per_row, &mut q8k);

        let avx2 = unsafe { q4_k_q8_k_row_dot_avx2(&row, blocks_per_row, &q8k) };
        let vnni = unsafe { q4_k_q8_k_row_dot_vnni(&row, blocks_per_row, &q8k) };
        assert!(
            (avx2 - vnni).abs() <= 1e-4 * (1.0 + avx2.abs()),
            "VNNI row dot {vnni} diverged from AVX2 {avx2}"
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn q6_k_dot_avx2_matches_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let mut block = vec![0u8; BLOCK_Q6_K_SIZE];
        let mut state: u32 = 0xBADC_0FFE;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u8
        };
        for b in block.iter_mut() {
            *b = next();
        }
        block[208] = 0x00;
        block[209] = 0x3c; // d = 1.0
        let vector: Vec<f32> = (0..QK_K).map(|i| ((i % 19) as f32 - 9.0) * 0.07).collect();

        let scalar = q6_k_dot_scalar(&block, &vector);
        let simd = unsafe { q6_k_dot_avx2(&block, &vector) };
        assert!(
            (scalar - simd).abs() <= 1e-3 * (1.0 + scalar.abs()),
            "AVX2 Q6_K dot ({simd}) diverged from scalar ({scalar})"
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn q8_0_dot_avx2_matches_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let mut block = vec![0u8; BLOCK_Q8_0_SIZE];
        block[0] = 0x00;
        block[1] = 0x3c; // scale = 1.0
        for (i, q) in block[2..].iter_mut().enumerate() {
            *q = (i as i8 - 16) as u8;
        }
        let vector: Vec<f32> = (0..QK8_0).map(|i| ((i % 7) as f32 - 3.0) * 0.13).collect();

        let scalar = q8_0_dot_scalar(&block, &vector);
        let simd = unsafe { q8_0_dot_avx2(&block, &vector) };
        assert!(
            (scalar - simd).abs() <= 1e-4 * (1.0 + scalar.abs()),
            "AVX2 Q8_0 dot ({simd}) diverged from scalar ({scalar})"
        );
    }

    fn reference_attention(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        seq_len: usize,
        dim: usize,
        output: &mut [f32],
    ) {
        let scale = 1.0_f32 / (dim as f32).sqrt();
        let mut scores = vec![0.0_f32; seq_len];

        for (idx, key_row) in key.chunks_exact(dim).enumerate() {
            let dot = query
                .iter()
                .zip(key_row.iter())
                .map(|(q, k)| q * k)
                .sum::<f32>();
            scores[idx] = dot * scale;
        }

        let max_score = scores.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        for score in &mut scores {
            *score = (*score - max_score).exp();
        }
        let sum_exp = scores.iter().sum::<f32>();
        for score in &mut scores {
            *score /= sum_exp;
        }

        output.fill(0.0);
        for (weight, value_row) in scores.iter().zip(value.chunks_exact(dim)) {
            for (out, v) in output.iter_mut().zip(value_row.iter()) {
                *out += weight * v;
            }
        }
    }

    #[test]
    fn creates_tensor_with_shape_strides_and_dtype() {
        let tensor = Tensor::new(vec![4, 8], vec![8, 1], DType::F32);

        assert_eq!(tensor.shape, vec![4, 8]);
        assert_eq!(tensor.strides, vec![8, 1]);
        assert_eq!(tensor.dtype, DType::F32);
    }

    #[test]
    #[should_panic(expected = "shape and strides must have the same rank")]
    fn rejects_mismatched_shape_and_strides() {
        let _ = Tensor::new(vec![2, 3], vec![3], DType::I8);
    }

    #[test]
    fn gemv_multiplies_matrix_and_vector() {
        let matrix = vec![
            1.0_f32, 2.0, 3.0, //
            4.0, 5.0, 6.0,
        ];
        let vector = vec![0.5_f32, -1.0, 2.0];
        let mut output = vec![0.0_f32; 2];

        gemv_f32(&matrix, 2, 3, &vector, &mut output).expect("gemv should succeed");

        assert!((output[0] - 4.5).abs() < 1e-6);
        assert!((output[1] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn gemv_rejects_invalid_input_shapes() {
        let mut output = vec![0.0_f32; 2];
        let err = gemv_f32(&[1.0_f32, 2.0, 3.0], 2, 2, &[1.0, 2.0], &mut output)
            .expect_err("matrix length mismatch should fail");
        assert!(matches!(err, GemvError::InvalidMatrixLength { .. }));

        let matrix = vec![1.0_f32, 2.0, 3.0, 4.0];
        let err = gemv_f32(&matrix, 2, 2, &[1.0_f32], &mut output)
            .expect_err("vector length mismatch should fail");
        assert!(matches!(err, GemvError::InvalidVectorLength { .. }));

        let mut short_output = vec![0.0_f32; 1];
        let err = gemv_f32(&matrix, 2, 2, &[1.0_f32, 1.0], &mut short_output)
            .expect_err("output length mismatch should fail");
        assert!(matches!(err, GemvError::InvalidOutputLength { .. }));
    }

    #[test]
    fn quantized_q8_0_gemv_matches_f32_reference() {
        let rows = 2;
        let cols = 32;
        let matrix = (0..rows * cols)
            .map(|i| (i as f32 * 0.125) - 2.0)
            .collect::<Vec<_>>();
        let vector = (0..cols)
            .map(|i| (i as f32 * 0.05) - 0.6)
            .collect::<Vec<_>>();

        let mut matrix_bytes = Vec::with_capacity(matrix.len() * 4);
        for value in &matrix {
            matrix_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let q8_bytes_len =
            crate::quantization::quantized_size(GgufQuantizationType::Q8_0, matrix.len())
                .expect("quantized size is known");
        let mut quantized_matrix = vec![0_u8; q8_bytes_len];
        crate::quantization::quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q8_0,
            &matrix_bytes,
            &mut quantized_matrix,
        )
        .expect("q8_0 quantization should succeed");

        let mut quantized_out = vec![0.0_f32; rows];
        gemv_quantized_f32(
            GgufQuantizationType::Q8_0,
            &quantized_matrix,
            rows,
            cols,
            &vector,
            &mut quantized_out,
        )
        .expect("quantized gemv should succeed");

        let mut dequantized = vec![0.0_f32; rows * cols];
        crate::quantization::dequantize_scalar(
            GgufQuantizationType::Q8_0,
            &quantized_matrix,
            &mut dequantized,
        )
        .expect("dequantization should succeed");
        let mut reference = vec![0.0_f32; rows];
        gemv_f32(&dequantized, rows, cols, &vector, &mut reference)
            .expect("f32 gemv should succeed");

        for (lhs, rhs) in quantized_out.iter().zip(reference.iter()) {
            assert!((lhs - rhs).abs() < CUDA_TOL);
        }
    }

    #[test]
    fn fused_q8_0_gemv_matches_dequantize_then_gemv_reference() {
        let rows = 3;
        let cols = QK8_0 * 2;
        let matrix = (0..rows * cols)
            .map(|i| ((i as f32 * 0.03125).sin() * 6.0) - 1.0)
            .collect::<Vec<_>>();
        let vector = (0..cols)
            .map(|i| ((i as f32 * 0.11).cos() * 0.5) + 0.25)
            .collect::<Vec<_>>();

        let mut matrix_bytes = Vec::with_capacity(matrix.len() * 4);
        for value in &matrix {
            matrix_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let q8_bytes_len =
            crate::quantization::quantized_size(GgufQuantizationType::Q8_0, matrix.len())
                .expect("quantized size is known");
        let mut quantized_matrix = vec![0_u8; q8_bytes_len];
        crate::quantization::quantize_scalar(
            GgufQuantizationType::F32,
            GgufQuantizationType::Q8_0,
            &matrix_bytes,
            &mut quantized_matrix,
        )
        .expect("q8_0 quantization should succeed");

        let mut fused_out = vec![0.0_f32; rows];
        gemv_quantized_f32(
            GgufQuantizationType::Q8_0,
            &quantized_matrix,
            rows,
            cols,
            &vector,
            &mut fused_out,
        )
        .expect("fused q8_0 gemv should succeed");

        let mut dequantized = vec![0.0_f32; rows * cols];
        crate::quantization::dequantize_scalar(
            GgufQuantizationType::Q8_0,
            &quantized_matrix,
            &mut dequantized,
        )
        .expect("dequantization should succeed");
        let mut reference = vec![0.0_f32; rows];
        gemv_f32(&dequantized, rows, cols, &vector, &mut reference)
            .expect("f32 gemv should succeed");

        for (lhs, rhs) in fused_out.iter().zip(reference.iter()) {
            assert!((lhs - rhs).abs() < CUDA_TOL);
        }
    }

    #[test]
    fn gemm_multiplies_matrices() {
        let left = vec![
            1.0_f32, 2.0, 3.0, //
            4.0, 5.0, 6.0,
        ];
        let right = vec![
            7.0_f32, 8.0, //
            9.0, 10.0, //
            11.0, 12.0,
        ];
        let mut output = vec![0.0_f32; 4];

        gemm_f32(&left, 2, 3, &right, 2, &mut output).expect("gemm should succeed");

        let expected = [58.0_f32, 64.0, 139.0, 154.0];
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn gemm_rejects_invalid_input_shapes() {
        let right = vec![
            1.0_f32, 2.0, //
            3.0, 4.0, //
            5.0, 6.0,
        ];
        let mut output = vec![0.0_f32; 4];

        let err = gemm_f32(&[1.0_f32, 2.0, 3.0], 2, 3, &right, 2, &mut output)
            .expect_err("left matrix length mismatch should fail");
        assert!(matches!(err, GemmError::InvalidLeftMatrixLength { .. }));

        let left = vec![
            1.0_f32, 2.0, 3.0, //
            4.0, 5.0, 6.0,
        ];
        let err = gemm_f32(&left, 2, 3, &[1.0_f32, 2.0, 3.0], 2, &mut output)
            .expect_err("right matrix length mismatch should fail");
        assert!(matches!(err, GemmError::InvalidRightMatrixLength { .. }));

        let mut short_output = vec![0.0_f32; 3];
        let err = gemm_f32(&left, 2, 3, &right, 2, &mut short_output)
            .expect_err("output length mismatch should fail");
        assert!(matches!(err, GemmError::InvalidOutputLength { .. }));
    }

    #[test]
    fn gemm_multiplies_non_square_matrices() {
        let left = vec![
            1.0_f32, 2.0, //
            3.0, 4.0, //
            5.0, 6.0,
        ];
        let right = vec![
            7.0_f32, 8.0, 9.0, 10.0, //
            11.0, 12.0, 13.0, 14.0,
        ];
        let mut output = vec![0.0_f32; 12];

        gemm_f32(&left, 3, 2, &right, 4, &mut output).expect("gemm should succeed");

        let expected = [
            29.0_f32, 32.0, 35.0, 38.0, //
            65.0, 72.0, 79.0, 86.0, //
            101.0, 112.0, 123.0, 134.0,
        ];
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn tensor_parallel_gemm_matches_cpu_reference() {
        let rows = 4;
        let shared_dim = 8;
        let cols = 3;
        let left = (0..rows * shared_dim)
            .map(|i| (i as f32 * 0.125) - 1.0)
            .collect::<Vec<_>>();
        let right = (0..shared_dim * cols)
            .map(|i| (i as f32 * 0.05) + 0.25)
            .collect::<Vec<_>>();

        let mut expected = vec![0.0_f32; rows * cols];
        gemm_f32_cpu(&left, rows, shared_dim, &right, cols, &mut expected);

        let mut actual = vec![0.0_f32; rows * cols];
        gemm_f32_tensor_parallel(&left, rows, shared_dim, &right, cols, 4, &mut actual)
            .expect("tensor parallel gemm should succeed");

        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn tensor_parallel_gemm_rejects_invalid_shard_count() {
        let left = vec![1.0_f32; 6];
        let right = vec![1.0_f32; 6];
        let mut output = vec![0.0_f32; 4];
        let err = gemm_f32_tensor_parallel(&left, 2, 3, &right, 2, 2, &mut output)
            .expect_err("non-divisible shard count should fail");
        assert_eq!(
            err,
            GemmError::InvalidTensorParallelShardCount {
                shared_dim: 3,
                shard_count: 2
            }
        );
    }

    #[test]
    fn gemm_i8_multiplies_matrices() {
        let left = vec![
            1_i8, -2, 3, //
            4, 0, -1,
        ];
        let right = vec![
            2_i8, -3, //
            1, 5, //
            -4, 2,
        ];
        let mut output = vec![0_i32; 4];

        gemm_i8(&left, 2, 3, &right, 2, &mut output).expect("int8 gemm should succeed");

        assert_eq!(output, vec![-12, -7, 12, -14]);
    }

    #[test]
    fn gemm_i8_rejects_invalid_shapes() {
        let right = vec![
            1_i8, 2, //
            3, 4, //
            5, 6,
        ];
        let mut output = vec![0_i32; 4];

        let err = gemm_i8(&[1_i8, 2, 3], 2, 3, &right, 2, &mut output)
            .expect_err("left matrix length mismatch should fail");
        assert!(matches!(err, GemmError::InvalidLeftMatrixLength { .. }));

        let left = vec![
            1_i8, 2, 3, //
            4, 5, 6,
        ];
        let err = gemm_i8(&left, 2, 3, &[1_i8, 2, 3], 2, &mut output)
            .expect_err("right matrix length mismatch should fail");
        assert!(matches!(err, GemmError::InvalidRightMatrixLength { .. }));

        let mut short_output = vec![0_i32; 3];
        let err = gemm_i8(&left, 2, 3, &right, 2, &mut short_output)
            .expect_err("output length mismatch should fail");
        assert!(matches!(err, GemmError::InvalidOutputLength { .. }));
    }

    #[test]
    fn gemm_i4_multiplies_packed_matrices() {
        let left_values = [1_i8, -2, 3, 4, 0, -1];
        let right_values = [2_i8, -3, 1, 5, -4, 2];
        let left = pack_i4_values(&left_values);
        let right = pack_i4_values(&right_values);
        let mut output = vec![0_i32; 4];

        gemm_i4(&left, 2, 3, &right, 2, &mut output).expect("int4 gemm should succeed");

        assert_eq!(output, vec![-12, -7, 12, -14]);
    }

    #[test]
    fn gemm_i4_supports_odd_value_count() {
        let left_values = [1_i8, -2, 3];
        let right_values = [2_i8, 4, -1];
        let left = pack_i4_values(&left_values);
        let right = pack_i4_values(&right_values);
        let mut output = vec![0_i32; 1];

        gemm_i4(&left, 1, 3, &right, 1, &mut output)
            .expect("odd shared dim int4 gemm should succeed");

        assert_eq!(output, vec![-9]);
    }

    #[test]
    fn gemm_i4_rejects_invalid_shapes() {
        let right = pack_i4_values(&[1_i8, 2, 3, 4, 5, 6]);
        let mut output = vec![0_i32; 4];

        let err = gemm_i4(&pack_i4_values(&[1_i8, 2, 3]), 2, 3, &right, 2, &mut output)
            .expect_err("left matrix byte length mismatch should fail");
        assert!(matches!(err, GemmError::InvalidLeftMatrixLength { .. }));

        let left = pack_i4_values(&[1_i8, 2, 3, 4, 5, 6]);
        let err = gemm_i4(&left, 2, 3, &pack_i4_values(&[1_i8, 2, 3]), 2, &mut output)
            .expect_err("right matrix byte length mismatch should fail");
        assert!(matches!(err, GemmError::InvalidRightMatrixLength { .. }));

        let mut short_output = vec![0_i32; 3];
        let err = gemm_i4(&left, 2, 3, &right, 2, &mut short_output)
            .expect_err("output length mismatch should fail");
        assert!(matches!(err, GemmError::InvalidOutputLength { .. }));
    }

    fn pack_i4_values(values: &[i8]) -> Vec<u8> {
        let mut packed = Vec::with_capacity(values.len().div_ceil(2));
        for chunk in values.chunks(2) {
            let low = ((chunk[0] + 8) as u8) & 0x0F;
            let high = if chunk.len() == 2 {
                ((chunk[1] + 8) as u8) & 0x0F
            } else {
                0
            };
            packed.push(low | (high << 4));
        }
        packed
    }

    #[test]
    fn scaled_dot_product_attention_computes_weighted_output() {
        let query = vec![1.0_f32, 0.0];
        let key = vec![
            1.0_f32, 0.0, //
            0.0, 1.0,
        ];
        let value = vec![
            10.0_f32, 0.0, //
            0.0, 20.0,
        ];
        let mut output = vec![0.0_f32; 2];

        scaled_dot_product_attention_f32(&query, &key, &value, 2, 2, &mut output)
            .expect("attention should succeed");

        assert!((output[0] - 6.697615).abs() < 1e-5);
        assert!((output[1] - 6.604_77).abs() < 1e-5);
    }

    #[test]
    fn scaled_dot_product_attention_rejects_invalid_shapes() {
        let mut output = vec![0.0_f32; 2];
        let err = scaled_dot_product_attention_f32(
            &[1.0_f32],
            &[1.0_f32, 0.0],
            &[1.0_f32, 0.0],
            1,
            2,
            &mut output,
        )
        .expect_err("query length mismatch should fail");
        assert!(matches!(err, AttentionError::InvalidQueryLength { .. }));

        let err = scaled_dot_product_attention_f32(
            &[1.0_f32, 0.0],
            &[1.0_f32, 0.0, 1.0],
            &[1.0_f32, 0.0, 1.0, 0.0],
            2,
            2,
            &mut output,
        )
        .expect_err("key length mismatch should fail");
        assert!(matches!(err, AttentionError::InvalidKeyLength { .. }));

        let err = scaled_dot_product_attention_f32(
            &[1.0_f32, 0.0],
            &[1.0_f32, 0.0, 1.0, 0.0],
            &[1.0_f32, 0.0, 1.0],
            2,
            2,
            &mut output,
        )
        .expect_err("value length mismatch should fail");
        assert!(matches!(err, AttentionError::InvalidValueLength { .. }));

        let mut short_output = vec![0.0_f32; 1];
        let err = scaled_dot_product_attention_f32(
            &[1.0_f32, 0.0],
            &[1.0_f32, 0.0, 1.0, 0.0],
            &[1.0_f32, 0.0, 1.0, 0.0],
            2,
            2,
            &mut short_output,
        )
        .expect_err("output length mismatch should fail");
        assert!(matches!(err, AttentionError::InvalidOutputLength { .. }));
    }

    #[test]
    fn scaled_dot_product_attention_matches_reference_softmax() {
        let dim = 4;
        let seq_len = 6;
        let query = vec![0.2_f32, -0.4, 1.2, 0.8];
        let key = vec![
            0.1_f32, 0.4, -0.2, 1.0, //
            0.5, -0.7, 0.3, 0.0, //
            -1.0, 0.1, 0.9, 0.2, //
            0.7, 0.6, -0.8, 0.3, //
            -0.4, 1.3, 0.2, -0.6, //
            0.8, -0.2, -0.5, 0.4,
        ];
        let value = vec![
            1.0_f32, 0.0, 2.0, -1.0, //
            0.3, 1.5, -0.2, 0.7, //
            -1.1, 0.6, 0.2, 0.4, //
            0.9, -0.8, 1.0, 0.5, //
            0.0, 0.2, -0.4, 1.2, //
            -0.7, 0.3, 0.8, -0.9,
        ];
        let mut actual = vec![0.0_f32; dim];
        let mut expected = vec![0.0_f32; dim];

        scaled_dot_product_attention_f32(&query, &key, &value, seq_len, dim, &mut actual)
            .expect("attention should succeed");
        reference_attention(&query, &key, &value, seq_len, dim, &mut expected);

        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn scaled_dot_product_attention_matches_pytorch_reference() {
        let query = vec![0.3_f32, -0.8, 1.1, 0.2];
        let key = vec![
            0.4_f32, -0.7, 0.9, -0.1, //
            -1.2, 0.3, 0.5, 0.8, //
            0.6, 1.0, -0.4, 0.2, //
            0.1, -0.2, 0.7, 1.3, //
            -0.5, 0.9, -1.1, 0.4,
        ];
        let value = vec![
            1.0_f32, -0.5, 0.3, 0.9, //
            -0.7, 1.2, 0.4, -1.1, //
            0.6, 0.8, -0.2, 0.5, //
            1.4, -0.9, 1.1, -0.3, //
            -0.4, 0.2, -1.3, 0.7,
        ];
        let expected = [0.704_716_4_f32, -0.158_690_65, 0.412_106_54, 0.145_940_9];
        let mut actual = vec![0.0_f32; 4];

        scaled_dot_product_attention_f32(&query, &key, &value, 5, 4, &mut actual)
            .expect("attention should succeed");

        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn scaled_dot_product_attention_is_stable_for_large_logits() {
        let dim = 2;
        let seq_len = 3;
        let query = vec![10_000.0_f32, -10_000.0];
        let key = vec![
            1.0_f32, -1.0, //
            -1.0, 1.0, //
            0.5, -0.5,
        ];
        let value = vec![
            2.0_f32, 4.0, //
            -3.0, 5.0, //
            7.0, -8.0,
        ];
        let mut output = vec![0.0_f32; dim];

        scaled_dot_product_attention_f32(&query, &key, &value, seq_len, dim, &mut output)
            .expect("attention should succeed");

        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn scaled_dot_product_attention_matches_reference_across_flash_blocks() {
        let dim = 8;
        let seq_len = FLASH_ATTENTION_BLOCK_TOKENS * 2 + 7;
        let query = (0..dim)
            .map(|i| (i as f32 * 0.13).sin() - 0.25)
            .collect::<Vec<_>>();
        let key = (0..seq_len * dim)
            .map(|i| ((i as f32 * 0.017).cos() * 1.3) - 0.2)
            .collect::<Vec<_>>();
        let value = (0..seq_len * dim)
            .map(|i| ((i as f32 * 0.031).sin() * 0.9) + 0.1)
            .collect::<Vec<_>>();

        let mut actual = vec![0.0_f32; dim];
        let mut expected = vec![0.0_f32; dim];
        scaled_dot_product_attention_f32(&query, &key, &value, seq_len, dim, &mut actual)
            .expect("flash-style attention should succeed");
        reference_attention(&query, &key, &value, seq_len, dim, &mut expected);

        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }

    #[test]
    fn scaled_dot_product_attention_matches_reference_for_long_contexts() {
        let dim = 4;
        let seq_len = FLASH_ATTENTION_BLOCK_TOKENS * 64;
        let query = (0..dim)
            .map(|i| ((i as f32 * 0.11).sin() * 0.7) - 0.1)
            .collect::<Vec<_>>();
        let key = (0..seq_len * dim)
            .map(|i| ((i as f32 * 0.007).cos() * 0.8) + 0.05)
            .collect::<Vec<_>>();
        let value = (0..seq_len * dim)
            .map(|i| ((i as f32 * 0.013).sin() * 1.1) - 0.2)
            .collect::<Vec<_>>();

        let mut actual = vec![0.0_f32; dim];
        let mut expected = vec![0.0_f32; dim];
        scaled_dot_product_attention_f32(&query, &key, &value, seq_len, dim, &mut actual)
            .expect("flash attention should support long contexts");
        reference_attention(&query, &key, &value, seq_len, dim, &mut expected);

        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }

    #[test]
    fn apply_rope_rotates_each_pair_by_position_dependent_angle() {
        let input = vec![1.0_f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0_f32; 4];

        apply_rope_f32(&input, 1, 4, 10_000.0, &mut output).expect("rope should succeed");

        assert!((output[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((output[2] - 1.0_f32.sin()).abs() < 1e-6);

        let angle_1 = 1.0_f32 / 100.0;
        assert!((output[1] + angle_1.sin()).abs() < 1e-6);
        assert!((output[3] - angle_1.cos()).abs() < 1e-6);
    }

    #[test]
    fn apply_rope_position_zero_is_identity() {
        let input = vec![0.25_f32, -0.75, 1.5, 2.0];
        let mut output = vec![0.0_f32; 4];

        apply_rope_f32(&input, 0, 4, 10_000.0, &mut output).expect("rope should succeed");

        for (actual, expected) in output.iter().zip(input.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn apply_rope_rejects_invalid_shapes() {
        let mut output = vec![0.0_f32; 4];
        let err = apply_rope_f32(&[1.0_f32, 2.0], 1, 4, 10_000.0, &mut output)
            .expect_err("input length mismatch should fail");
        assert!(matches!(err, RopeError::InvalidInputLength { .. }));

        let mut short_output = vec![0.0_f32; 2];
        let err = apply_rope_f32(&[1.0_f32, 2.0, 3.0, 4.0], 1, 4, 10_000.0, &mut short_output)
            .expect_err("output length mismatch should fail");
        assert!(matches!(err, RopeError::InvalidOutputLength { .. }));

        let mut odd_output = vec![0.0_f32; 3];
        let err = apply_rope_f32(&[1.0_f32, 2.0, 3.0], 1, 3, 10_000.0, &mut odd_output)
            .expect_err("odd head dimension should fail");
        assert!(matches!(err, RopeError::OddHeadDim { .. }));
    }

    #[test]
    fn swiglu_applies_silu_gate_times_up_projection() {
        let gate = [0.0_f32, 2.0, -2.0];
        let up = [1.0_f32, 3.0, -4.0];
        let mut output = [0.0_f32; 3];

        apply_swiglu_f32(&gate, &up, &mut output).expect("swiglu should succeed");

        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 5.284_782_4).abs() < 1e-6);
        assert!((output[2] - 0.953_623_35).abs() < 1e-6);
    }

    #[test]
    fn swiglu_rejects_mismatched_input_lengths() {
        let mut output = [0.0_f32; 2];

        let gate_err = apply_swiglu_f32(&[1.0_f32], &[1.0_f32, 2.0], &mut output)
            .expect_err("gate length mismatch should fail");
        assert!(matches!(gate_err, SwiGluError::InvalidGateLength { .. }));

        let up_err = apply_swiglu_f32(&[1.0_f32, 2.0], &[1.0_f32], &mut output)
            .expect_err("up length mismatch should fail");
        assert!(matches!(up_err, SwiGluError::InvalidUpLength { .. }));
    }

    #[test]
    fn linear_activation_fuses_gemv_and_relu() {
        let matrix = [
            1.0_f32, -2.0, 3.0, //
            -4.0, 5.0, -6.0,
        ];
        let vector = [0.5_f32, 2.0, -1.0];
        let mut output = [0.0_f32; 2];

        linear_activation_f32(&matrix, 2, 3, &vector, ActivationFn::Relu, &mut output)
            .expect("fused linear+relu should succeed");

        let expected_linear = [-6.5_f32, 14.0];
        let expected = [expected_linear[0].max(0.0), expected_linear[1].max(0.0)];
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn linear_activation_matches_reference_gelu_and_silu() {
        let matrix = vec![
            0.2_f32, -0.4, 1.1, -0.9, //
            1.4, 0.3, -0.7, 0.5, //
            -0.8, 0.6, 0.9, -1.2,
        ];
        let vector = vec![0.3_f32, -0.2, 1.7, 0.4];
        let mut linear = vec![0.0_f32; 3];
        gemv_f32(&matrix, 3, 4, &vector, &mut linear).expect("gemv should succeed");

        for activation in [ActivationFn::Gelu, ActivationFn::Silu] {
            let mut fused = vec![0.0_f32; 3];
            linear_activation_f32(&matrix, 3, 4, &vector, activation, &mut fused)
                .expect("fused linear activation should succeed");
            let expected = linear
                .iter()
                .map(|value| activate(*value, activation))
                .collect::<Vec<_>>();
            for (actual, expected) in fused.iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn linear_activation_rejects_invalid_shapes() {
        let matrix = [1.0_f32, 2.0, 3.0, 4.0];
        let vector = [1.0_f32, 2.0];
        let mut output = [0.0_f32; 2];

        let matrix_err = linear_activation_f32(
            &[1.0_f32, 2.0, 3.0],
            2,
            2,
            &vector,
            ActivationFn::Relu,
            &mut output,
        )
        .expect_err("matrix length mismatch should fail");
        assert!(matches!(
            matrix_err,
            LinearActivationError::InvalidMatrixLength { .. }
        ));

        let vector_err =
            linear_activation_f32(&matrix, 2, 2, &[1.0_f32], ActivationFn::Relu, &mut output)
                .expect_err("vector length mismatch should fail");
        assert!(matches!(
            vector_err,
            LinearActivationError::InvalidVectorLength { .. }
        ));

        let mut short_output = [0.0_f32; 1];
        let output_err = linear_activation_f32(
            &matrix,
            2,
            2,
            &vector,
            ActivationFn::Relu,
            &mut short_output,
        )
        .expect_err("output length mismatch should fail");
        assert!(matches!(
            output_err,
            LinearActivationError::InvalidOutputLength { .. }
        ));
    }

    #[test]
    fn rms_norm_scales_by_root_mean_square() {
        let input = [1.0_f32, 2.0, 3.0];
        let weight = [1.0_f32, 1.0, 1.0];
        let mut output = [0.0_f32; 3];

        rms_norm_f32(&input, &weight, 0.0, &mut output).expect("rms norm should succeed");

        assert!((output[0] - 0.462_910_06).abs() < 1e-6);
        assert!((output[1] - 0.925_820_1).abs() < 1e-6);
        assert!((output[2] - 1.388_730_2).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_matches_pytorch_reference() {
        let input = [0.25_f32, -1.5, 3.0, 0.75, -0.5];
        let weight = [1.2_f32, 0.8, -0.6, 2.0, 0.5];
        let expected = [
            0.192_648_f32,
            -0.770_592_03,
            -1.155_888_1,
            0.963_24,
            -0.160_54,
        ];
        let mut output = [0.0_f32; 5];

        rms_norm_f32(&input, &weight, 1e-5, &mut output).expect("rms norm should succeed");

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn rms_norm_rejects_zero_dimension() {
        let err =
            rms_norm_f32(&[], &[], 1e-5, &mut []).expect_err("zero-length output should fail");
        assert_eq!(err, RmsNormError::ZeroDimension);
    }

    #[test]
    fn rms_norm_rejects_mismatched_lengths() {
        let mut output = [0.0_f32; 2];
        let input_err = rms_norm_f32(&[1.0_f32], &[1.0_f32, 1.0], 1e-5, &mut output)
            .expect_err("input length mismatch should fail");
        assert!(matches!(input_err, RmsNormError::InvalidInputLength { .. }));

        let weight_err = rms_norm_f32(&[1.0_f32, 2.0], &[1.0_f32], 1e-5, &mut output)
            .expect_err("weight length mismatch should fail");
        assert!(matches!(
            weight_err,
            RmsNormError::InvalidWeightLength { .. }
        ));
    }

    #[test]
    fn layer_norm_normalizes_and_applies_affine_transform() {
        let input = [1.0_f32, 2.0, 3.0];
        let weight = [1.0_f32, 1.0, 1.0];
        let bias = [0.0_f32, 0.0, 0.0];
        let mut output = [0.0_f32; 3];

        layer_norm_f32(&input, &weight, &bias, 0.0, &mut output)
            .expect("layer norm should succeed");

        assert!((output[0] + 1.224_744_8).abs() < 1e-6);
        assert!((output[1] - 0.0).abs() < 1e-6);
        assert!((output[2] - 1.224_744_8).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_matches_pytorch_reference() {
        let input = [0.25_f32, -1.5, 3.0, 0.75, -0.5];
        let weight = [1.2_f32, 0.8, -0.6, 2.0, 0.5];
        let bias = [0.1_f32, -0.2, 0.3, -0.4, 0.9];
        let expected = [
            -0.019_601_732_f32,
            -1.209_970_1,
            -0.736_548_3,
            0.065_117_806,
            0.600_995_66,
        ];
        let mut output = [0.0_f32; 5];

        layer_norm_f32(&input, &weight, &bias, 1e-5, &mut output)
            .expect("layer norm should succeed");

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn layer_norm_rejects_mismatched_lengths() {
        let mut output = [0.0_f32; 2];
        let input_err = layer_norm_f32(
            &[1.0_f32],
            &[1.0_f32, 1.0],
            &[0.0_f32, 0.0],
            1e-5,
            &mut output,
        )
        .expect_err("input length mismatch should fail");
        assert!(matches!(
            input_err,
            LayerNormError::InvalidInputLength { .. }
        ));

        let weight_err = layer_norm_f32(
            &[1.0_f32, 2.0],
            &[1.0_f32],
            &[0.0_f32, 0.0],
            1e-5,
            &mut output,
        )
        .expect_err("weight length mismatch should fail");
        assert!(matches!(
            weight_err,
            LayerNormError::InvalidWeightLength { .. }
        ));

        let bias_err = layer_norm_f32(
            &[1.0_f32, 2.0],
            &[1.0_f32, 1.0],
            &[0.0_f32],
            1e-5,
            &mut output,
        )
        .expect_err("bias length mismatch should fail");
        assert!(matches!(bias_err, LayerNormError::InvalidBiasLength { .. }));
    }

    #[test]
    fn softmax_outputs_probabilities_that_sum_to_one() {
        let input = [1.0_f32, 2.0, 3.0];
        let mut output = [0.0_f32; 3];

        softmax_f32(&input, &mut output).expect("softmax should succeed");

        assert!((output[0] - 0.090_030_57).abs() < 1e-7);
        assert!((output[1] - 0.244_728_48).abs() < 1e-7);
        assert!((output[2] - 0.665_240_94).abs() < 1e-7);
        assert!((output.iter().sum::<f32>() - 1.0).abs() < 1e-7);
    }

    #[test]
    fn softmax_matches_pytorch_reference() {
        let input = [1.25_f32, -0.5, 3.75, 0.0, -2.25];
        let expected = [
            0.073_137_f32,
            0.012_709_305,
            0.890_991_1,
            0.020_954_102,
            0.002_208_546_3,
        ];
        let mut output = [0.0_f32; 5];

        softmax_f32(&input, &mut output).expect("softmax should succeed");

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-7);
        }
    }

    #[test]
    fn softmax_is_numerically_stable_for_large_inputs() {
        let input = [10_000.0_f32, 9_999.0, 9_998.0];
        let mut output = [0.0_f32; 3];

        softmax_f32(&input, &mut output).expect("softmax should succeed");

        assert!(output.iter().all(|value| value.is_finite()));
        assert!((output.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(output[0] > output[1]);
        assert!(output[1] > output[2]);
    }

    #[test]
    fn softmax_rejects_mismatched_lengths() {
        let mut output = [0.0_f32; 2];
        let err =
            softmax_f32(&[1.0_f32], &mut output).expect_err("mismatched input length should fail");
        assert!(matches!(err, SoftmaxError::InvalidInputLength { .. }));
    }
}
