use crate::gguf::GgufQuantizationType;
use crate::quantization::{
    BLOCK_IQ4_NL_SIZE, BLOCK_IQ4_XS_SIZE, BLOCK_NVFP4_SIZE, BLOCK_Q2_K_SIZE, BLOCK_Q4_K_SIZE,
    BLOCK_Q6_K_SIZE, BLOCK_Q8_0_SIZE, KVALUES_IQ4NL, QK_K, QK_NVFP4, QK_NVFP4_SUB, QK4_NL, QK8_0,
    dequantize_iq4_xs_scalar,
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

/// Stamp a quantized GEMV wrapper: shape validation (blocks per row from
/// `$qk` elements, `$block_size` bytes per block), then row-parallel or
/// serial dispatch above/below `PARALLEL_GEMV_MIN_OPS`. `$compute_row`
/// receives `(row_bytes, blocks_per_row)` and returns the row's dot
/// product; it must close over the input vector.
///
/// The generated function is the exact shape every hand-written wrapper
/// in q_kernels.rs / gemv.rs used before this macro existed.
#[allow(unused_macros)]
macro_rules! oc_gemv_dispatch {
    // With prologue statements (cfg-detected flags etc.): $([prologue])?
    ($name:ident, $block_size:expr, $qk:expr, $rows:ident, $cols:ident,
     $matrix:ident, $vector:ident, $output:ident,
     [$($prologue:tt)*]
     $compute_row:expr) => {
        oc_gemv_dispatch! { @impl $name, $block_size, $qk, $rows, $cols,
            $matrix, $vector, $output, [$($prologue)*], $compute_row }
    };
    ($name:ident, $block_size:expr, $qk:expr, $rows:ident, $cols:ident,
     $matrix:ident, $vector:ident, $output:ident,
     $compute_row:expr) => {
        oc_gemv_dispatch! { @impl $name, $block_size, $qk, $rows, $cols,
            $matrix, $vector, $output, [], $compute_row }
    };
    (@impl $name:ident, $block_size:expr, $qk:expr, $rows:ident, $cols:ident,
     $matrix:ident, $vector:ident, $output:ident,
     [$($prologue:tt)*], $compute_row:expr) => {
        pub(super) fn $name(
            $matrix: &[u8],
            $rows: usize,
            $cols: usize,
            $vector: &[f32],
            $output: &mut [f32],
        ) -> Result<(), GemvError> {
            let blocks_per_row = $cols / $qk;
            let expected_matrix_len = $rows * blocks_per_row * $block_size;
            if $matrix.len() != expected_matrix_len {
                return Err(GemvError::InvalidMatrixLength {
                    expected: expected_matrix_len,
                    actual: $matrix.len(),
                });
            }
            if $vector.len() != $cols {
                return Err(GemvError::InvalidVectorLength {
                    expected: $cols,
                    actual: $vector.len(),
                });
            }
            if $output.len() != $rows {
                return Err(GemvError::InvalidOutputLength {
                    expected: $rows,
                    actual: $output.len(),
                });
            }
            $($prologue)*
            let row_bytes = blocks_per_row * $block_size;
            let compute_row = $compute_row;
            if $rows.saturating_mul($cols) >= PARALLEL_GEMV_MIN_OPS {
                $output
                    .par_iter_mut()
                    .with_min_len(32)
                    .enumerate()
                    .for_each(|(row_idx, out)| *out = compute_row(row_bytes, row_idx, blocks_per_row));
            } else {
                for (row_idx, out) in $output.iter_mut().enumerate() {
                    *out = compute_row(row_bytes, row_idx, blocks_per_row);
                }
            }
            Ok(())
        }
    };
}
pub(super) use oc_gemv_dispatch;

/// Stamp a batched decode-once GEMM wrapper (gemm_*_decode_once): shape
/// validation, AVX2 fast-path dispatch to `$avx2`, then the generic
/// row-major + transpose epilogue. `$compute_row` receives
/// `(row_idx, partial)` where partial is the batch-long output slice.
#[allow(unused_macros)]
macro_rules! oc_gemm_decode_dispatch {
    // With a hoisted AVX2 fast-path body.
    ($name:ident, $avx2:ident, $block_size:expr, $qk:expr, $compute_row:expr) => {
        pub(super) fn $name(
            quantized_matrix: &[u8],
            rows: usize,
            cols: usize,
            inputs: &[f32],
            outputs: &mut [f32],
            batch: usize,
        ) -> Result<(), GemvError> {
            let blocks_per_row = cols / $qk;
            let expected_matrix_len = rows
                .saturating_mul(blocks_per_row)
                .saturating_mul($block_size);
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
                        $avx2(
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
            let compute_row = $compute_row;
            let total_ops = rows.saturating_mul(cols).saturating_mul(batch);
            if total_ops >= PARALLEL_GEMV_MIN_OPS {
                row_major
                    .par_chunks_mut(batch)
                    .enumerate()
                    .for_each(|(row_idx, slice)| {
                        compute_row(
                            quantized_matrix,
                            inputs,
                            cols,
                            blocks_per_row,
                            batch,
                            row_idx,
                            slice,
                        )
                    });
            } else {
                for (row_idx, slice) in row_major.chunks_mut(batch).enumerate() {
                    compute_row(
                        quantized_matrix,
                        inputs,
                        cols,
                        blocks_per_row,
                        batch,
                        row_idx,
                        slice,
                    );
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
    };
    // Scalar/inline-AVX2 closure only: no hoisted fast-path function.
    ($name:ident, $block_size:expr, $qk:expr, $compute_row:expr) => {
        pub(super) fn $name(
            quantized_matrix: &[u8],
            rows: usize,
            cols: usize,
            inputs: &[f32],
            outputs: &mut [f32],
            batch: usize,
        ) -> Result<(), GemvError> {
            let blocks_per_row = cols / $qk;
            let expected_matrix_len = rows
                .saturating_mul(blocks_per_row)
                .saturating_mul($block_size);
            if quantized_matrix.len() != expected_matrix_len {
                return Err(GemvError::InvalidMatrixLength {
                    expected: expected_matrix_len,
                    actual: quantized_matrix.len(),
                });
            }
            let mut row_major = vec![0.0_f32; rows.saturating_mul(batch)];
            let compute_row = $compute_row;
            let total_ops = rows.saturating_mul(cols).saturating_mul(batch);
            if total_ops >= PARALLEL_GEMV_MIN_OPS {
                row_major
                    .par_chunks_mut(batch)
                    .enumerate()
                    .for_each(|(row_idx, slice)| {
                        compute_row(
                            quantized_matrix,
                            inputs,
                            cols,
                            blocks_per_row,
                            batch,
                            row_idx,
                            slice,
                        )
                    });
            } else {
                for (row_idx, slice) in row_major.chunks_mut(batch).enumerate() {
                    compute_row(
                        quantized_matrix,
                        inputs,
                        cols,
                        blocks_per_row,
                        batch,
                        row_idx,
                        slice,
                    );
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
    };
}
pub(super) use oc_gemm_decode_dispatch;

/// Rows per spin-pool dispatch chunk. Small chunks cost nothing under static
/// partitioning (no claim contention) and cut straggler imbalance on
/// mid-sized regions; 8 still holds two 4-row kernel quads.
const GEMV_CHUNK_ROWS: usize = 32;

const TRANSPOSED_GEMV_COL_CHUNK: usize = QK_K;

/// Per-block Q8_K layout (matches llama.cpp's `block_q8_K`):
///   bytes 0..4   : f32 d (1/iscale)
///   bytes 4..260 : 256 int8 quants
///   bytes 260..292 : 16 int16 bsums (sum of int8 quants in groups of 16)
const BLOCK_Q8_K_BYTES: usize = 4 + 256 + 32;
const BLOCK_IQ1_S_SIZE: usize = 2 + 32 + 16; // ggml_half d + qs[32] + qh[16]
const IQ1S_DELTA: f32 = 0.125;
const BLOCK_IQ1_M_SIZE: usize = 32 + 16 + 8; // qs[32] + qh[16] + scales[8]

mod activation;
mod core_types;
mod gemm;
mod gemm_decode;
mod gemv;
mod gpu;
mod q_kernels;
mod transposed;
mod types;

pub use activation::*;
pub use core_types::*;
pub use gemm::*;
pub use gemm_decode::*;
pub use gemv::*;
pub use gpu::*;
pub use q_kernels::*;
pub use transposed::*;
pub use types::*;

#[cfg(test)]
mod tests;
