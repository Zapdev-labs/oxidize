//! Apple MLX compute backend (macOS only).

/// Build-time detection info for the MLX backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxBuildInfo {
    pub detected_at_build: bool,
}

/// Returns whether the MLX backend was detected at build time.
pub fn mlx_build_info() -> MlxBuildInfo {
    MlxBuildInfo {
        detected_at_build: cfg!(target_os = "macos"),
    }
}

/// Error type for MLX kernel operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlxKernelError {
    InvalidMatrixLength { expected: usize, actual: usize },
    InvalidVectorLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_build_info_reports_macos_detection() {
        assert_eq!(mlx_build_info().detected_at_build, cfg!(target_os = "macos"));
    }
}
