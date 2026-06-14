//! Core APIs for `oxidize`.
//!
//! This crate exposes model/runtime primitives and a small public health surface
//! used by CLI, server, and WASM integrations.
//!
//! # API quick check
//!
//! ```
//! use oxidize_core::{benchmark_input, workspace_health};
//!
//! assert_eq!(workspace_health().status, "ready");
//! assert_eq!(benchmark_input().status, "ready");
//! ```
//!
//! Build local API docs with:
//!
//! ```text
//! cargo doc -p oxidize-core --no-deps
//! ```
//!
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

pub use futures_core::Stream;

#[path = "backend.rs"]
pub mod backend;
pub use backend::ComputeBackend;
#[path = "model/advanced_features.rs"]
pub mod advanced_features;
#[path = "util/benchmark_suite.rs"]
pub mod benchmark_suite;
#[path = "format/conversion.rs"]
pub mod conversion;
#[path = "compute/cpu_kernels.rs"]
pub mod cpu_kernels;
#[path = "validation/cross_validation.rs"]
pub mod cross_validation;
#[path = "backends/cuda.rs"]
pub mod cuda;
#[path = "model/dflash.rs"]
pub mod dflash;
#[path = "compute/flash_attention.rs"]
pub mod flash_attention;
#[path = "model/generation.rs"]
pub mod generation;
#[path = "format/gguf.rs"]
pub mod gguf;
#[path = "cluster/gpu_cluster.rs"]
pub mod gpu_cluster;
#[path = "model/inference.rs"]
pub mod inference;
#[path = "compute/kv_cache.rs"]
pub mod kv_cache;
#[path = "model/layer_wise.rs"]
pub mod layer_wise;
#[path = "model/llama.rs"]
pub mod llama;
#[path = "model/lora.rs"]
pub mod lora;
#[path = "mesh/mod.rs"]
pub mod mesh;
#[path = "backends/metal.rs"]
pub mod metal;
#[cfg(target_os = "macos")]
#[path = "backends/mlx.rs"]
pub mod mlx;
#[path = "model/mlx_inference.rs"]
pub mod mlx_inference;
#[path = "model/model.rs"]
pub mod model;
#[path = "model/loader.rs"]
pub mod model_loader;
#[path = "compute/numa.rs"]
pub mod numa;
#[path = "model/offload.rs"]
pub mod offload;
#[path = "paged_attention/mod.rs"]
pub mod paged_attention;
#[path = "model/prefix_cache.rs"]
pub mod prefix_cache;
#[path = "compute/quantization.rs"]
pub mod quantization;
#[path = "format/safetensors.rs"]
pub mod safetensors;
#[path = "format/safetensors_to_gguf.rs"]
pub mod safetensors_to_gguf;
#[path = "model/sampling.rs"]
pub mod sampling;
#[path = "compute/simd.rs"]
pub mod simd;
#[path = "model/speculative.rs"]
pub mod speculative;
#[path = "compute/spinpool.rs"]
pub mod spinpool;
#[path = "backends/strix.rs"]
pub mod strix;
#[path = "compute/tensor.rs"]
pub mod tensor;
#[path = "format/tokenizer.rs"]
pub mod tokenizer;
#[path = "compute/turboquant.rs"]
pub mod turboquant;
#[path = "video/mod.rs"]
pub mod video;
#[path = "model/video.rs"]
pub mod video_model;
#[path = "vision/mod.rs"]
pub mod vision;
#[cfg(feature = "vulkan")]
#[path = "backends/vulkan.rs"]
pub mod vulkan;
#[cfg(not(feature = "vulkan"))]
#[path = "backends/vulkan_stub.rs"]
pub mod vulkan;
#[path = "util/web_worker.rs"]
pub mod web_worker;
#[path = "backends/webgpu.rs"]
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
/// use oxidize_core::workspace_health;
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
/// use oxidize_core::benchmark_input;
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
    fn oxidize_core_declares_optional_cuda_pipeline() {
        let crate_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(crate_cargo_toml).expect("oxidize-core Cargo.toml exists");

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
    fn oxidize_core_declares_metal_feature_and_macos_build_dependency() {
        let crate_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(crate_cargo_toml).expect("oxidize-core Cargo.toml exists");

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
    fn oxidize_core_declares_vulkan_feature_and_dependencies() {
        let crate_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(crate_cargo_toml).expect("oxidize-core Cargo.toml exists");

        assert!(
            cargo_toml.contains("vulkan = [\"dep:ash\", \"dep:gpu-allocator\", \"dep:shaderc\"]")
        );
        assert!(cargo_toml.contains("ash = { version = \"0.38\", optional = true }"));
        assert!(cargo_toml.contains("gpu-allocator = { version = \"0.27\", optional = true }"));
        assert!(cargo_toml.contains("shaderc = { version = \"0.8.3\", optional = true }"));
    }

    #[test]
    fn vulkan_build_script_sets_expected_cfg_and_loader_detection() {
        let build_script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("build.rs");
        let build_script = std::fs::read_to_string(build_script_path).expect("build.rs exists");

        assert!(build_script.contains("cargo:rustc-check-cfg=cfg(vulkan_available)"));
        assert!(build_script.contains("if detect_vulkan_available()"));
        assert!(build_script.contains("env::var_os(\"CARGO_FEATURE_VULKAN\").is_none()"));
        assert!(build_script.contains("libvulkan.so.1"));
    }

    #[test]
    fn oxidize_core_declares_webgpu_feature_and_dependency() {
        let crate_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(crate_cargo_toml).expect("oxidize-core Cargo.toml exists");

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
    fn oxidize_core_declares_wasm_bindgen_build_surface() {
        let crate_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(crate_cargo_toml).expect("oxidize-core Cargo.toml exists");

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
            "cargo build -p oxidize-core --target wasm32-unknown-unknown --release --features wasm"
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
        assert!(js.contains("from \"../../dist/wasm/oxidize_core.js\""));
        assert!(js.contains("wasm_workspace_status"));
        assert!(js.contains("wasm_collect_worker_stream"));
        assert!(css.contains(".app"));
    }

    #[test]
    fn oxidize_core_docsrs_metadata_is_declared_for_rustdoc_builds() {
        let crate_cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cargo_toml =
            std::fs::read_to_string(crate_cargo_toml).expect("oxidize-core Cargo.toml exists");

        assert!(cargo_toml.contains("[package.metadata.docs.rs]"));
        assert!(cargo_toml.contains("all-features = true"));
        assert!(cargo_toml.contains("rustdoc-args = [\"--cfg\", \"docsrs\"]"));
    }
}
