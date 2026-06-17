//! CPU tensor kernels, dtypes, and GEMV/GEMM entrypoints.
//!
//! Split incrementally from the former monolithic `tensor.rs`. `unsafe` in [`kernels`] is
//! limited to SIMD intrinsics and raw pointer math with documented `SAFETY` preconditions.

mod errors;
mod kernels;

pub use errors::*;
pub use kernels::*;
