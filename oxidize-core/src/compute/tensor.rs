use crate::gguf::GgufQuantizationType;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const QK8_0: usize = 32;
const BLOCK_Q8_0_SIZE: usize = 2 + QK8_0;
const QK_K: usize = 256;
const BLOCK_Q4_K_SIZE: usize = 2 * std::mem::size_of::<u16>() + 12 + QK_K / 2;
const BLOCK_Q2_K_SIZE: usize = 2 * std::mem::size_of::<u16>() + QK_K / 16 + QK_K / 4;
const BLOCK_Q6_K_SIZE: usize = std::mem::size_of::<u16>() + QK_K / 16 + 3 * QK_K / 4;
const FLASH_ATTENTION_BLOCK_TOKENS: usize = 64;
const PARALLEL_GEMV_MIN_OPS: usize = 1 << 20;
const TRANSPOSED_GEMV_COL_CHUNK: usize = QK_K;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I8,
    I16,
    I32,
    I64,
}

impl DType {
    /// Return the size of a single element in bytes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemvError {
    InvalidMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidVectorLength {
        expected: usize,
        actual: usize,
    },
    InvalidOutputLength {
        expected: usize,
        actual: usize,
    },
    UnsupportedQuantizationType {
        quantization: GgufQuantizationType,
    },
    #[cfg(feature = "cuda")]
    Cuda(String),
    #[cfg(feature = "metal")]
    Metal(String),
    #[cfg(feature = "webgpu")]
    WebGpu(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmError {
    InvalidLeftMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidRightMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidOutputLength {
        expected: usize,
        actual: usize,
    },
    #[cfg(feature = "cuda")]
    Cuda(String),
    #[cfg(feature = "metal")]
    Metal(String),
    #[cfg(feature = "webgpu")]
    WebGpu(String),
    InvalidTensorParallelShardCount {
        shared_dim: usize,
        shard_count: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionError {
    ZeroHeadDim,
    InvalidQueryLength { expected: usize, actual: usize },
    InvalidKeyLength { expected: usize, actual: usize },
    InvalidValueLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
    InvalidKvHead { kv_head: usize, kv_heads: usize },
    InvalidHeadGrouping { num_heads: usize, kv_heads: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RopeError {
    InvalidInputLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
    OddHeadDim { head_dim: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwiGluError {
    InvalidGateLength { expected: usize, actual: usize },
    InvalidUpLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFn {
    Relu,
    Gelu,
    Silu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearActivationError {
    InvalidMatrixLength { expected: usize, actual: usize },
    InvalidVectorLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RmsNormError {
    ZeroDimension,
    InvalidInputLength { expected: usize, actual: usize },
    InvalidWeightLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerNormError {
    InvalidInputLength { expected: usize, actual: usize },
    InvalidWeightLength { expected: usize, actual: usize },
    InvalidBiasLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SoftmaxError {
    InvalidInputLength { expected: usize, actual: usize },
}

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

    #[cfg(feature = "cuda")]
    if crate::cuda::cuda_build_info().detected_at_build {
        return crate::cuda::gemv_f32_cuda(matrix, rows, cols, vector, output)
            .map_err(|err| GemvError::Cuda(format!("{err:?}")));
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
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
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
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

#[inline]
fn dot_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
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

    // Rows are processed in chunks to amortize rayon task dispatch overhead.
    const ROW_CHUNK: usize = 16;
    let process_row = |row_idx: usize, partial: &mut [f32]| {
        let qm_ptr = qm_ptr_addr as *const u8;
        let in_ptr = in_ptr_addr as *const f32;
        let mut scratch = [0.0_f32; QK_K];
        partial.fill(0.0);
        let row_base = unsafe { qm_ptr.add(row_idx * row_stride_bytes) };
        for block_idx in 0..blocks_per_row {
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
                let (s0, s1, s2, s3) =
                    unsafe { dot4_f32_avx2(scratch.as_ptr(), v0, v1, v2, v3, QK_K) };
                partial[t0] += s0;
                partial[t0 + 1] += s1;
                partial[t0 + 2] += s2;
                partial[t0 + 3] += s3;
            }
            for ti in 0..tail {
                let t = chunks * 4 + ti;
                let v_ptr = unsafe { in_ptr.add(t * cols + in_offset_floats) };
                partial[t] += unsafe { dot_f32_avx2(scratch.as_ptr(), v_ptr, QK_K) };
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
                        let (s0, s1, s2, s3) =
                            unsafe { dot4_f32_avx2(scratch.as_ptr(), v0, v1, v2, v3, QK_K) };
                        partial[t0] += s0;
                        partial[t0 + 1] += s1;
                        partial[t0 + 2] += s2;
                        partial[t0 + 3] += s3;
                    }
                    for ti in 0..tail {
                        let t = chunks * 4 + ti;
                        let v_ptr = unsafe { in_ptr.add(t * cols + in_offset_floats) };
                        partial[t] += unsafe { dot_f32_avx2(scratch.as_ptr(), v_ptr, QK_K) };
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

pub fn gemv_quantized_f32(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    #[cfg(feature = "cuda")]
    if crate::cuda::cuda_build_info().detected_at_build {
        return crate::cuda::gemv_quantized_cuda(
            quantization,
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
        )
        .map_err(|err| GemvError::Cuda(format!("{err:?}")));
    }

    match quantization {
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
        GgufQuantizationType::Q6_K => {
            gemv_q6_k_f32_fused(quantized_matrix, rows, cols, vector, output)
        }
        _ => Err(GemvError::UnsupportedQuantizationType { quantization }),
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
        let row_start = row_idx * row_bytes;
        let row = &weights[row_start..row_start + row_bytes];
        // Safety: q4_k_q8_k_avx2_available() was checked before dispatch.
        unsafe { q4_k_q8_k_row_dot_avx2(row, blocks_per_row, &q8k) }
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
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
fn quantize_vector_q8_k_into(vector: &[f32], n_blocks: usize, out: &mut [u8]) {
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
unsafe fn q4_k_q8_k_row_dot_avx2(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    let mask = _mm256_set1_epi8(0x0f);
    let ones = _mm256_set1_epi16(1);
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w_ptr = row.as_ptr().wrapping_add(block_idx * BLOCK_Q4_K_SIZE);
        let q8_ptr = q8k.as_ptr().wrapping_add(block_idx * BLOCK_Q8_K_BYTES);

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
            let p32_low = _mm256_madd_epi16(p16_low, ones);
            let p32_high = _mm256_madd_epi16(p16_high, ones);
            let s1_v = _mm256_set1_epi32(s1 as i32);
            let s2_v = _mm256_set1_epi32(s2 as i32);
            vec_pos = _mm256_add_epi32(vec_pos, _mm256_mullo_epi32(p32_low, s1_v));
            vec_pos = _mm256_add_epi32(vec_pos, _mm256_mullo_epi32(p32_high, s2_v));

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
    let ones = _mm256_set1_epi16(1);
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

        // Broadcast scales for vector mullo.
        let s_v = [
            _mm256_set1_epi32(g_scales[0]),
            _mm256_set1_epi32(g_scales[1]),
            _mm256_set1_epi32(g_scales[2]),
            _mm256_set1_epi32(g_scales[3]),
            _mm256_set1_epi32(g_scales[4]),
            _mm256_set1_epi32(g_scales[5]),
            _mm256_set1_epi32(g_scales[6]),
            _mm256_set1_epi32(g_scales[7]),
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
                let p32_low = _mm256_madd_epi16(p16_low, ones);
                let p32_high = _mm256_madd_epi16(p16_high, ones);
                let scaled_low = _mm256_mullo_epi32(p32_low, s_v[g1]);
                let scaled_high = _mm256_mullo_epi32(p32_high, s_v[g2]);
                vec_pos = _mm256_add_epi32(vec_pos, scaled_low);
                vec_pos = _mm256_add_epi32(vec_pos, scaled_high);
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
    let rows = output.len();
    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        matrix
            .par_chunks_exact(cols)
            .zip(output.par_iter_mut())
            .for_each(|(row_values, out)| {
                *out = row_values
                    .iter()
                    .zip(vector.iter())
                    .map(|(weight, value)| weight * value)
                    .sum();
            });
    } else {
        for (row_values, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
            *out = row_values
                .iter()
                .zip(vector.iter())
                .map(|(weight, value)| weight * value)
                .sum();
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
    const PREFETCH_DISTANCE: usize = 16;
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
            let right_col = &right_transposed[col * shared_dim..(col + 1) * shared_dim];
            let mut sum = 0.0_f32;
            for (k, left_value) in left_row.iter().enumerate() {
                if let (Some(next_left), Some(next_right)) = (
                    left_row.get(k + PREFETCH_DISTANCE),
                    right_col.get(k + PREFETCH_DISTANCE),
                ) {
                    std::hint::black_box(*next_left + *next_right);
                }
                sum += left_value * right_col[k];
            }
            *out_cell = sum;
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
                (actual - reference).abs() <= 1e-4 * (1.0 + reference.abs()),
                "batched q4_k gemm value {actual} diverged from repeated gemv {reference}"
            );
        }
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
            assert!((lhs - rhs).abs() < 1e-4);
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
            assert!((lhs - rhs).abs() < 1e-4);
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
