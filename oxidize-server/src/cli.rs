//! Command-line surface: `Args`, `Backend`, `BatchMode`.

use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    Cpu,
    Metal,
    /// macOS only
    Mlx,
    Cuda,
    Vulkan,
}

impl Backend {
    pub fn to_core_backend(self) -> oxidize_core::backend::Backend {
        match self {
            Backend::Cpu => oxidize_core::backend::Backend::Cpu,
            Backend::Metal => oxidize_core::backend::Backend::Metal,
            Backend::Mlx => oxidize_core::backend::Backend::Mlx,
            Backend::Cuda => oxidize_core::backend::Backend::Cuda,
            Backend::Vulkan => oxidize_core::backend::Backend::Vulkan,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BatchMode {
    Sequential,
    Paged,
}

impl BatchMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            BatchMode::Sequential => "sequential",
            BatchMode::Paged => "paged",
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "oxidize-server")]
pub struct Args {
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    pub host: IpAddr,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    pub backend: Backend,
    #[arg(long, value_enum, default_value_t = BatchMode::Sequential)]
    pub batch_mode: BatchMode,
    #[arg(long, default_value = "oxidize-default")]
    pub model_id: String,
    #[arg(long, default_value_t = 512)]
    pub max_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,
    #[arg(long)]
    pub top_p: Option<f32>,
    #[arg(long)]
    pub top_k: Option<usize>,
    #[arg(long)]
    pub ctx_size: Option<usize>,
    #[arg(long, default_value_t = 512)]
    pub prefill_batch_size: usize,
    #[arg(long, default_value_t = false)]
    pub cpu_optimized: bool,
    #[arg(long, default_value_t = false)]
    pub ram_offload: bool,
    #[arg(long, default_value_t = false)]
    pub mmap_prefetch: bool,
    #[arg(long, default_value_t = false)]
    pub mmap_hugepages: bool,
    #[arg(long, default_value_t = false)]
    pub layer_wise: bool,
    #[arg(long, default_value_t = 1)]
    pub layer_cache: usize,
    /// Use TurboQuant block-quantized KV cache (only affects --kv-cache-dtype q4/q8).
    #[arg(long, default_value_t = false)]
    pub turboquant_kv: bool,
    /// Enable mesh cluster mode: this node becomes the master that routes
    /// OpenAI-compatible requests to worker shards over the mesh data plane.
    #[arg(long, default_value_t = false)]
    pub mesh: bool,
    /// Port for the mesh libp2p listener (0 = ephemeral).
    #[arg(long, default_value_t = 0)]
    pub mesh_port: u16,
    /// External GGUF file that contains the tokenizer metadata.
    /// Useful for draft models (e.g. DFlash) that do not embed a tokenizer.
    #[arg(long)]
    pub tokenizer_model: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_use_expected_defaults() {
        let args = Args::parse_from(["oxidize-server"]);
        assert_eq!(args.host, IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(args.port, 8080);
    }

    #[test]
    fn args_accept_explicit_values() {
        let args = Args::parse_from(["oxidize-server", "--host", "0.0.0.0", "--port", "3000"]);
        assert_eq!(args.host, IpAddr::V4(Ipv4Addr::UNSPECIFIED));
        assert_eq!(args.port, 3000);
    }
}
