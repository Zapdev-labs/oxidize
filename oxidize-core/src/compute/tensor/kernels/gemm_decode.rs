use super::*;

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
pub(super) fn gemm_quantized_f32_inner(
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
        GgufQuantizationType::IQ4_XS => {
            Some((BLOCK_IQ4_XS_SIZE, iq4_xs_dot as fn(&[u8], &[f32]) -> f32))
        }
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
pub(super) unsafe fn decode_q4_k_group_avx2(
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
pub(super) fn decode_q4_k_block(block: &[u8], out: &mut [f32]) {
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
pub(super) fn decode_q8_0_block(block: &[u8], out: &mut [f32]) {
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
pub(super) unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
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

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub(super) unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..len {
        sum += unsafe { *a.add(i) * *b.add(i) };
    }
    sum
}

/// Four dot products that share the `a` load: `(a·b0, a·b1, a·b2, a·b3)`.
/// Halves the L1 traffic on `a` versus four separate `dot_f32_avx2` calls,
/// which is the dominant cost in batched quantized GEMM after the per-row
/// weight scan amortization.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
pub(super) unsafe fn dot4_f32_avx2(
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

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub(super) unsafe fn dot4_f32_avx2(
    a: *const f32,
    b0: *const f32,
    b1: *const f32,
    b2: *const f32,
    b3: *const f32,
    len: usize,
) -> (f32, f32, f32, f32) {
    (
        unsafe { dot_f32_avx2(a, b0, len) },
        unsafe { dot_f32_avx2(a, b1, len) },
        unsafe { dot_f32_avx2(a, b2, len) },
        unsafe { dot_f32_avx2(a, b3, len) },
    )
}

/// AVX-512 counterpart of [`dot4_f32_avx2`]: 16-wide FMA, four shared-`a`
/// accumulators. On Skylake-SP this doubles the dot lanes versus AVX2 while the
/// 32-zmm register file absorbs the four input streams without spilling.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
pub(super) unsafe fn dot4_f32_avx512(
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

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub(super) unsafe fn dot4_f32_avx512(
    a: *const f32,
    b0: *const f32,
    b1: *const f32,
    b2: *const f32,
    b3: *const f32,
    len: usize,
) -> (f32, f32, f32, f32) {
    unsafe { dot4_f32_avx2(a, b0, b1, b2, b3, len) }
}

/// AVX-512 counterpart of [`dot_f32_avx2`].
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
pub(super) unsafe fn dot_f32_avx512(a: *const f32, b: *const f32, len: usize) -> f32 {
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

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub(super) unsafe fn dot_f32_avx512(a: *const f32, b: *const f32, len: usize) -> f32 {
    unsafe { dot_f32_avx2(a, b, len) }
}

#[inline]
pub(super) fn dot_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            return unsafe { dot_f32_avx512(a.as_ptr(), b.as_ptr(), a.len()) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_avx2(a.as_ptr(), b.as_ptr(), a.len()) };
        }
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// Fast batched Q4_K GEMM path: for each output row, walk each 144-byte block
// once, decode it into a 256-element f32 scratch buffer, then for each batch
// token compute `scratch · input_chunk` with AVX2 FMA. Decode cost is paid
// once per row block; the dot product is paid per token but hits L1 since both
// operands are tiny.
oc_gemm_decode_dispatch!(
    gemm_q4_k_decode_once,
    gemm_q4_k_decode_once_avx2,
    BLOCK_Q4_K_SIZE,
    QK_K,
    |matrix: &[u8],
     inputs: &[f32],
     cols: usize,
     blocks_per_row: usize,
     batch: usize,
     row_idx: usize,
     partial: &mut [f32]| {
        let mut scratch = [0.0_f32; QK_K];
        let row_start = row_idx * blocks_per_row * BLOCK_Q4_K_SIZE;
        let row_blocks = &matrix[row_start..row_start + blocks_per_row * BLOCK_Q4_K_SIZE];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q4_K_SIZE).enumerate() {
            decode_q4_k_block(block, &mut scratch);
            let in_offset = block_idx * QK_K;
            for t in 0..batch {
                let v = &inputs[t * cols + in_offset..t * cols + in_offset + QK_K];
                partial[t] += dot_f32_fast(&scratch, v);
            }
        }
    }
);

/// AVX2/FMA hot-path body of [`gemm_q4_k_decode_once`]. Hoists the runtime
/// CPU-feature check out of the (rows × blocks × batch) inner loop and lets
/// the compiler inline `decode_q4_k_group_avx2` + the dot kernel directly.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn gemm_q4_k_decode_once_avx2(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
    blocks_per_row: usize,
) -> Result<(), GemvError> {
    let in_ptr_addr = inputs.as_ptr() as usize;
    let out_ptr_addr = outputs.as_mut_ptr() as usize;
    let row_stride_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;

    // The weight decode stays AVX2, but the per-token dot product over the
    // 256-element f32 scratch runs 16-wide on AVX-512 hardware (Skylake-SP).
    // Detected once here; the branch in the inner loop is perfectly predicted.
    let use_avx512 = {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    };

    // Rows are processed in chunks to amortize rayon task dispatch overhead.
    const ROW_CHUNK: usize = 16;
    let process_row = |row_idx: usize, partial: &mut [f32]| {
        let in_ptr = in_ptr_addr as *const f32;
        let mut scratch = [0.0_f32; QK_K];
        partial.fill(0.0);
        let row_start = row_idx * row_stride_bytes;
        let row_slice = &quantized_matrix[row_start..row_start + row_stride_bytes];
        for block_idx in 0..blocks_per_row {
            let block = &row_slice[block_idx * BLOCK_Q4_K_SIZE..][..BLOCK_Q4_K_SIZE];
            let d = f16_le_to_f32([block[0], block[1]]);
            let min = f16_le_to_f32([block[2], block[3]]);
            let scales = &block[4..16];
            let qs_ptr = block[16..].as_ptr();
            for g in 0..8 {
                let (sc, m) = get_scale_min_k4(g, scales);
                let dl = d * sc as f32;
                let ml = min * m as f32;
                let pair = g / 2;
                decode_q4_k_group_avx2(
                    qs_ptr.wrapping_add(pair * 32),
                    dl,
                    ml,
                    g & 1 != 0,
                    scratch.as_mut_ptr().add(g * 32),
                );
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

// Fast batched Q8_0 GEMM, same shape as [`gemm_q4_k_decode_once`].
oc_gemm_decode_dispatch!(
    gemm_q8_0_decode_once,
    BLOCK_Q8_0_SIZE,
    QK8_0,
    |matrix: &[u8],
     inputs: &[f32],
     cols: usize,
     blocks_per_row: usize,
     batch: usize,
     row_idx: usize,
     partial: &mut [f32]| {
        let mut scratch = [0.0_f32; QK8_0];
        let row_start = row_idx * blocks_per_row * BLOCK_Q8_0_SIZE;
        let row_blocks = &matrix[row_start..row_start + blocks_per_row * BLOCK_Q8_0_SIZE];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q8_0_SIZE).enumerate() {
            decode_q8_0_block(block, &mut scratch);
            let in_offset = block_idx * QK8_0;
            for t in 0..batch {
                let v = &inputs[t * cols + in_offset..t * cols + in_offset + QK8_0];
                partial[t] += dot_f32_fast(&scratch, v);
            }
        }
    }
);

/// Q6_K batched GEMM with decode-once optimization. Mirrors
/// `gemm_q4_k_decode_once` for 6-bit super-blocks: decode each 256-element
/// block to f32 once, then fan out a fast f32 dot against every batch token.
pub(super) fn gemm_q6_k_decode_once(
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
            let use_avx512 = {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl")
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            };
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
pub(super) fn decode_q6_k_block(block: &[u8], out: &mut [f32]) {
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

oc_gemm_decode_dispatch!(
    gemm_iq1_s_decode_once,
    BLOCK_IQ1_S_SIZE,
    QK_K,
    |matrix: &[u8],
     inputs: &[f32],
     cols: usize,
     blocks_per_row: usize,
     batch: usize,
     row_idx: usize,
     partial: &mut [f32]| {
        let row_start = row_idx * blocks_per_row * BLOCK_IQ1_S_SIZE;
        let row_blocks = &matrix[row_start..row_start + blocks_per_row * BLOCK_IQ1_S_SIZE];
        let mut scratch = [0.0_f32; QK_K];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_IQ1_S_SIZE).enumerate() {
            iq1s_dequantize_block(block, &mut scratch);
            let in_offset = block_idx * QK_K;
            for t in 0..batch {
                let v = &inputs[t * cols + in_offset..t * cols + in_offset + QK_K];
                partial[t] += crate::flash_attention::dot_product_f32(&scratch, v);
            }
        }
    }
);

oc_gemm_decode_dispatch!(
    gemm_iq1_m_decode_once,
    BLOCK_IQ1_M_SIZE,
    QK_K,
    |matrix: &[u8],
     inputs: &[f32],
     cols: usize,
     blocks_per_row: usize,
     batch: usize,
     row_idx: usize,
     partial: &mut [f32]| {
        let row_start = row_idx * blocks_per_row * BLOCK_IQ1_M_SIZE;
        let row_blocks = &matrix[row_start..row_start + blocks_per_row * BLOCK_IQ1_M_SIZE];
        let mut scratch = [0.0_f32; QK_K];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_IQ1_M_SIZE).enumerate() {
            iq1m_dequantize_block(block, &mut scratch);
            let in_offset = block_idx * QK_K;
            for t in 0..batch {
                let v = &inputs[t * cols + in_offset..t * cols + in_offset + QK_K];
                partial[t] += crate::flash_attention::dot_product_f32(&scratch, v);
            }
        }
    }
);

oc_gemm_decode_dispatch!(
    gemm_nvfp4_decode_once,
    BLOCK_NVFP4_SIZE,
    QK_NVFP4,
    |matrix: &[u8],
     inputs: &[f32],
     cols: usize,
     blocks_per_row: usize,
     batch: usize,
     row_idx: usize,
     partial: &mut [f32]| {
        let row_start = row_idx * blocks_per_row * BLOCK_NVFP4_SIZE;
        let row_blocks = &matrix[row_start..row_start + blocks_per_row * BLOCK_NVFP4_SIZE];
        let mut scratch = [0.0_f32; QK_NVFP4];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_NVFP4_SIZE).enumerate() {
            nvfp4_dequantize_block(block, &mut scratch);
            let in_offset = block_idx * QK_NVFP4;
            for t in 0..batch {
                let v = &inputs[t * cols + in_offset..t * cols + in_offset + QK_NVFP4];
                partial[t] += dot_f32_fast(&scratch, v);
            }
        }
    }
);

#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_k_quant_block(
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
