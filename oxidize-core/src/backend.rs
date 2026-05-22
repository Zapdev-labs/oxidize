//! Backend selection and platform-aware fallback logic.

/// Supported compute backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
    Cuda,
    Mlx,
}

impl std::str::FromStr for Backend {
    type Err = ();

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "cpu" => Ok(Backend::Cpu),
            "metal" => Ok(Backend::Metal),
            "cuda" => Ok(Backend::Cuda),
            "mlx" => Ok(Backend::Mlx),
            _ => Err(()),
        }
    }
}

impl Backend {
    /// Return the canonical name of this backend.
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
            Backend::Mlx => "mlx",
        }
    }

    /// Determine the effective backend for the current platform.
    ///
    /// On non-macOS platforms, `Mlx` is downgraded to `Cpu` and a warning
    /// message is returned.
    pub fn effective(self) -> (Self, Option<&'static str>) {
        match self {
            Backend::Mlx if !cfg!(target_os = "macos") => (
                Backend::Cpu,
                Some("MLX backend requested but unavailable on Linux; falling back to CPU"),
            ),
            other => (other, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn backend_parses_all_variants() {
        assert_eq!(Backend::from_str("cpu"), Ok(Backend::Cpu));
        assert_eq!(Backend::from_str("metal"), Ok(Backend::Metal));
        assert_eq!(Backend::from_str("cuda"), Ok(Backend::Cuda));
        assert_eq!(Backend::from_str("mlx"), Ok(Backend::Mlx));
        assert_eq!(Backend::from_str("unknown"), Err(()));
    }

    #[test]
    fn backend_roundtrips_through_str() {
        for backend in [Backend::Cpu, Backend::Metal, Backend::Cuda, Backend::Mlx] {
            assert_eq!(Backend::from_str(backend.as_str()), Ok(backend));
        }
    }

    #[test]
    fn mlx_fallback_on_linux() {
        if !cfg!(target_os = "macos") {
            let (effective, warning) = Backend::Mlx.effective();
            assert_eq!(effective, Backend::Cpu);
            assert!(
                warning.is_some(),
                "expected a warning when requesting MLX on non-macOS"
            );
            assert_eq!(
                warning.unwrap(),
                "MLX backend requested but unavailable on Linux; falling back to CPU"
            );
        }
    }

    #[test]
    fn cpu_always_effective() {
        let (effective, warning) = Backend::Cpu.effective();
        assert_eq!(effective, Backend::Cpu);
        assert!(warning.is_none());
    }

    #[test]
    fn metal_always_effective() {
        let (effective, warning) = Backend::Metal.effective();
        assert_eq!(effective, Backend::Metal);
        assert!(warning.is_none());
    }

    #[test]
    fn cuda_always_effective() {
        let (effective, warning) = Backend::Cuda.effective();
        assert_eq!(effective, Backend::Cuda);
        assert!(warning.is_none());
    }
}
