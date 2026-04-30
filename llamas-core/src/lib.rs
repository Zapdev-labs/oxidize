use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceHealth {
    pub status: &'static str,
}

pub fn workspace_health() -> WorkspaceHealth {
    WorkspaceHealth { status: "ready" }
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
    fn workspace_has_arm64_and_wasm32_targets_configured() {
        let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(".cargo")
            .join("config.toml");
        let config = std::fs::read_to_string(config_path).expect("workspace .cargo/config.toml exists");

        assert!(config.contains("[target.aarch64-unknown-linux-gnu]"));
        assert!(config.contains("[target.wasm32-unknown-unknown]"));
    }
}
