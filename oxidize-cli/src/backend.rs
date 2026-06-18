use clap::ValueEnum;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    Cpu,
    Metal,
    /// macOS only
    Mlx,
    Cuda,
    /// AMD ROCm / HIP
    Rocm,
    Vulkan,
    /// Intel Arc GPUs via Vulkan compute
    IntelArc,
}

impl Backend {
    pub fn to_core_backend(self) -> oxidize_core::backend::Backend {
        match self {
            Backend::Cpu => oxidize_core::backend::Backend::Cpu,
            Backend::Metal => oxidize_core::backend::Backend::Metal,
            Backend::Mlx => oxidize_core::backend::Backend::Mlx,
            Backend::Cuda => oxidize_core::backend::Backend::Cuda,
            Backend::Rocm => oxidize_core::backend::Backend::Rocm,
            Backend::Vulkan => oxidize_core::backend::Backend::Vulkan,
            Backend::IntelArc => oxidize_core::backend::Backend::IntelArc,
        }
    }

    #[allow(dead_code)]
    pub fn as_arg(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Metal => "metal",
            Backend::Mlx => "mlx",
            Backend::Cuda => "cuda",
            Backend::Rocm => "rocm",
            Backend::Vulkan => "vulkan",
            Backend::IntelArc => "intel-arc",
        }
    }
}
