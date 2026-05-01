//! Core APIs for `llamas-cpp`.
//!
//! This crate exposes model/runtime primitives and a small public health surface
//! used by CLI, server, and WASM integrations.
//!
//! # API quick check
//!
//! ```
//! use llamas_core::{benchmark_input, workspace_health};
//!
//! assert_eq!(workspace_health().status, "ready");
//! assert_eq!(benchmark_input().status, "ready");
//! ```
//!
//! Build local API docs with:
//!
//! ```text
//! cargo doc -p llamas-core --no-deps
//! ```
//!
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

pub mod benchmark_suite;
pub mod cuda;
pub mod generation;
pub mod gguf;
pub mod inference;
pub mod kv_cache;
pub mod llama;
pub mod lora;
pub mod metal;
pub mod model;
pub mod model_loader;
pub mod offload;
pub mod quantization;
pub mod sampling;
pub mod simd;
pub mod tensor;
pub mod tokenizer;
pub mod web_worker;
pub mod webgpu;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceHealth {
    /// Human-readable workspace readiness status.
    pub status: &'static str,
}

/// Returns the current workspace readiness signal.
///
/// # Examples
///
/// ```
/// use llamas_core::workspace_health;
///
/// assert_eq!(workspace_health().status, "ready");
/// ```
pub fn workspace_health() -> WorkspaceHealth {
    WorkspaceHealth { status: "ready" }
}

/// Returns health input used by benchmark harnesses.
///
/// # Examples
///
/// ```
/// use llamas_core::benchmark_input;
///
/// assert_eq!(benchmark_input().status, "ready");
/// ```
pub fn benchmark_input() -> WorkspaceHealth {
    workspace_health()
}

#[cfg_attr(all(target_arch = "wasm32", feature = "wasm"), wasm_bindgen)]
/// Returns the workspace status string for WASM consumers.
pub fn wasm_workspace_status() -> String {
    workspace_health().status.to_string()
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
        let workspace_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(workspace_cargo_toml).expect("workspace Cargo.toml exists");

        assert!(cargo_toml.contains("[profile.release]"));
        assert!(cargo_toml.contains("lto = true"));
        assert!(cargo_toml.contains("panic = \"abort\""));
    }

    #[test]
    fn llamas_core_declares_optional_cuda_pipeline() {
        let crate_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(crate_cargo_toml).expect("llamas-core Cargo.toml exists");

        assert!(cargo_toml.contains("build = \"build.rs\""));
        assert!(cargo_toml.contains("cuda = [\"dep:cublas-sys\", \"dep:cust\"]"));
        assert!(cargo_toml.contains("cublas-sys = { version = \"0.1\", optional = true }"));
        assert!(cargo_toml.contains("cust = { version = \"0.3\", optional = true }"));
    }

    #[test]
    fn cuda_build_script_tracks_expected_environment_inputs() {
        let build_script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("build.rs");
        let build_script = std::fs::read_to_string(build_script_path).expect("build.rs exists");

        assert!(build_script.contains("cargo:rerun-if-env-changed=CUDA_HOME"));
        assert!(build_script.contains("cargo:rerun-if-env-changed=CUDA_PATH"));
        assert!(build_script.contains("cargo:rustc-check-cfg=cfg(cuda_available)"));
    }

    #[test]
    fn llamas_core_declares_metal_feature_and_macos_build_dependency() {
        let crate_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(crate_cargo_toml).expect("llamas-core Cargo.toml exists");

        assert!(cargo_toml.contains("metal = []"));
        assert!(cargo_toml.contains("[target.'cfg(target_os = \"macos\")'.build-dependencies]"));
        assert!(cargo_toml.contains("metal = \"0.31\""));
    }

    #[test]
    fn metal_build_script_sets_expected_cfg_and_detection_path() {
        let build_script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("build.rs");
        let build_script = std::fs::read_to_string(build_script_path).expect("build.rs exists");

        assert!(build_script.contains("cargo:rustc-check-cfg=cfg(metal_available)"));
        assert!(build_script.contains("if detect_metal_available()"));
        assert!(build_script.contains("metal::Device::system_default().is_some()"));
    }

    #[test]
    fn llamas_core_declares_webgpu_feature_and_dependency() {
        let crate_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(crate_cargo_toml).expect("llamas-core Cargo.toml exists");

        assert!(cargo_toml.contains("webgpu = [\"dep:wgpu\"]"));
        assert!(cargo_toml.contains("wgpu = { version = \"25\", optional = true }"));
    }

    #[test]
    fn webgpu_build_script_sets_expected_cfg_and_feature_detection() {
        let build_script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("build.rs");
        let build_script = std::fs::read_to_string(build_script_path).expect("build.rs exists");

        assert!(build_script.contains("cargo:rustc-check-cfg=cfg(webgpu_available)"));
        assert!(build_script.contains("if detect_webgpu_available()"));
        assert!(build_script.contains("env::var_os(\"CARGO_FEATURE_WEBGPU\").is_some()"));
    }

    #[test]
    fn wasm_workspace_status_matches_workspace_health() {
        assert_eq!(wasm_workspace_status(), workspace_health().status);
    }

    #[test]
    fn llamas_core_declares_wasm_bindgen_build_surface() {
        let crate_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(crate_cargo_toml).expect("llamas-core Cargo.toml exists");

        assert!(cargo_toml.contains("crate-type = [\"cdylib\", \"rlib\"]"));
        assert!(cargo_toml.contains("wasm = [\"dep:wasm-bindgen\"]"));
        assert!(cargo_toml.contains("wasm-bindgen = { version = \"0.2\", optional = true }"));
    }

    #[test]
    fn makefile_exposes_wasm_bindgen_build_target() {
        let makefile = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("Makefile");
        let makefile = std::fs::read_to_string(makefile).expect("workspace Makefile exists");

        assert!(makefile.contains(".PHONY: help fmt lint audit test build wasm check ci"));
        assert!(makefile.contains(
            "cargo build -p llamas-core --target wasm32-unknown-unknown --release --features wasm"
        ));
        assert!(
            makefile.contains("command -v wasm-bindgen >/dev/null || cargo install --locked wasm-bindgen-cli --version 0.2.120")
        );
        assert!(makefile.contains("wasm-bindgen --target web --out-dir dist/wasm"));
    }

    #[test]
    fn web_demo_assets_exist_and_include_wasm_integration() {
        let demo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("web-demo");
        let html =
            std::fs::read_to_string(demo_root.join("index.html")).expect("demo index.html exists");
        let js = std::fs::read_to_string(demo_root.join("app.js")).expect("demo app.js exists");
        let css =
            std::fs::read_to_string(demo_root.join("styles.css")).expect("demo styles.css exists");

        assert!(html.contains("<script type=\"module\" src=\"./app.js\"></script>"));
        assert!(js.contains("from \"../../dist/wasm/llamas_core.js\""));
        assert!(js.contains("wasm_workspace_status"));
        assert!(js.contains("wasm_collect_worker_stream"));
        assert!(css.contains(".app"));
    }

    #[test]
    fn llamas_core_docsrs_metadata_is_declared_for_rustdoc_builds() {
        let crate_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(crate_cargo_toml).expect("llamas-core Cargo.toml exists");

        assert!(cargo_toml.contains("[package.metadata.docs.rs]"));
        assert!(cargo_toml.contains("all-features = true"));
        assert!(cargo_toml.contains("rustdoc-args = [\"--cfg\", \"docsrs\"]"));
    }
}
