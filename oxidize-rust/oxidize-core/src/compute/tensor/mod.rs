//! CPU tensor kernels, dtypes, and GEMV/GEMM entrypoints.
//!
//! Split incrementally from the former monolithic `tensor.rs`. `unsafe` in [`kernels`] is
//! limited to SIMD intrinsics, GPU dispatch, and parallel slice construction.

mod errors;
mod kernels;
mod types;

pub use errors::*;
pub use kernels::*;
pub use types::*;
