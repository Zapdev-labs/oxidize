use super::*;

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
pub(super) fn run_output_chunks(output: &mut [f32], chunk: usize, body: impl Fn(usize, &mut [f32]) + Sync) {
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
pub(super) mod gemv_profile {
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
        GgufQuantizationType::IQ4_XS if cols.is_multiple_of(QK_K) => {
            gemv_iq4_xs_f32(quantized_matrix, rows, cols, vector, output)
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
pub(super) fn q4_k_q8_k_avx2_available() -> bool {
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
pub(super) fn q4_k_q8_k_vnni_available() -> bool {
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
pub(super) enum GemvMode {
    Legacy,
    #[cfg(feature = "oxk")]
    Oxk,
    #[cfg(feature = "oxk")]
    Shadow,
}

#[cfg_attr(not(feature = "oxk"), allow(dead_code))]
pub(super) fn gemv_mode() -> GemvMode {
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
pub(super) fn shadow_q4k_range(
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
pub(super) unsafe fn q4_k_q8_k_row_dot(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
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
pub(super) fn gemv_q6_k_q8_k_fused(
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
pub(super) fn gemv_q4_k_q8_k_fused(
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
pub(super) fn gemm_q4_k_q8_k_fused(
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
pub(super) unsafe fn gemm_q4_k_q8_k_fused_avx2(
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
