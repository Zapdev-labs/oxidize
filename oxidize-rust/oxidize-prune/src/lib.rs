//! `oxidize-prune` — copy a GGUF, optionally pruning weights by
//! Wanda, magnitude, or tensor-name filtering.
//!
//! See `AGENTS.md` (in the same directory) for the public API, the
//! L2-norms cache format, and reference papers. The CLI binary
//! `oxidize-prune` consumes this library; downstream crates
//! (`oxidize-convert`) can also call it directly.

#![allow(clippy::needless_range_loop, clippy::ptr_arg)]

pub mod filter;
pub mod gguf_copy;
pub mod mask;
pub mod wanda;
pub mod writer;
