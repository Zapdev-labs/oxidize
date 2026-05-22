use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(cuda_available)");
    println!("cargo:rustc-check-cfg=cfg(metal_available)");
    println!("cargo:rustc-check-cfg=cfg(webgpu_available)");
    println!("cargo:rustc-check-cfg=cfg(vulkan_available)");
    println!("cargo:rustc-check-cfg=cfg(mlx_available)");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if let Some(cuda_root) = detect_cuda_root() {
        println!("cargo:rustc-cfg=cuda_available");
        println!("cargo:rustc-env=OXIDIZE_CUDA_PATH={}", cuda_root.display());

        let lib64 = cuda_root.join("lib64");
        if lib64.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
            println!("cargo:rustc-link-lib=dylib=cudart");
        }
    }

    if detect_metal_available() {
        println!("cargo:rustc-cfg=metal_available");
    }

    if detect_webgpu_available() {
        println!("cargo:rustc-cfg=webgpu_available");
    }

    if detect_vulkan_available() {
        println!("cargo:rustc-cfg=vulkan_available");
    }

    if detect_mlx_available() {
        println!("cargo:rustc-cfg=mlx_available");
    }
}

fn detect_cuda_root() -> Option<PathBuf> {
    for key in ["CUDA_HOME", "CUDA_PATH"] {
        match env::var_os(key).map(PathBuf::from) {
            Some(path) if path.is_dir() => return Some(path),
            _ => {}
        }
    }

    let default = Path::new("/usr/local/cuda");
    if default.is_dir() {
        Some(default.to_path_buf())
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
fn detect_metal_available() -> bool {
    metal::Device::system_default().is_some()
}

#[cfg(not(target_os = "macos"))]
fn detect_metal_available() -> bool {
    false
}

fn detect_webgpu_available() -> bool {
    env::var_os("CARGO_FEATURE_WEBGPU").is_some()
}

fn detect_vulkan_available() -> bool {
    env::var_os("CARGO_FEATURE_VULKAN").is_some()
}

fn detect_mlx_available() -> bool {
    detect_metal_available()
}
