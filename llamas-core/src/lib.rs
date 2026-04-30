use serde::{Deserialize, Serialize};

pub mod gguf;
pub mod model_loader;
pub mod quantization;
pub mod simd;
pub mod tensor;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceHealth {
    pub status: &'static str,
}

pub fn workspace_health() -> WorkspaceHealth {
    WorkspaceHealth { status: "ready" }
}

pub fn benchmark_input() -> WorkspaceHealth {
    workspace_health()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn workspace_health_is_ready() {
        assert_eq!(workspace_health().status, "ready");
    }

    #[test]
    fn benchmark_input_is_ready() {
        assert_eq!(benchmark_input().status, "ready");
    }

    #[test]
    fn workspace_has_arm64_and_wasm32_targets_configured() {
        let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(".cargo")
            .join("config.toml");
        let config =
            std::fs::read_to_string(config_path).expect("workspace .cargo/config.toml exists");

        assert!(config.contains("[target.aarch64-unknown-linux-gnu]"));
        assert!(config.contains("[target.wasm32-unknown-unknown]"));
    }

    #[test]
    fn workspace_release_profile_enables_lto_and_abort_panic() {
        let workspace_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(workspace_cargo_toml).expect("workspace Cargo.toml exists");

        assert!(cargo_toml.contains("[profile.release]"));
        assert!(cargo_toml.contains("lto = true"));
        assert!(cargo_toml.contains("panic = \"abort\""));
    }
}
