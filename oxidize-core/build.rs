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
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    if let Some(cuda_root) = detect_cuda_root() {
        println!("cargo:rustc-cfg=cuda_available");
        println!("cargo:rustc-env=OXIDIZE_CUDA_PATH={}", cuda_root.display());

        let lib64 = cuda_root.join("lib64");
        if lib64.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
            println!("cargo:rustc-link-lib=dylib=cudart");
        }

        // Compile CUDA kernels to PTX at build time so the Rust backend loads
        // fresh, forward-compatible PTX instead of a stale checked-in file.
        let nvcc = cuda_root.join("bin").join("nvcc");
        if nvcc.is_file() {
            let out_dir = env::var_os("OUT_DIR").map(PathBuf::from).unwrap_or_default();
            let ptx_path = out_dir.join("gemv_f32.ptx");
            let cu_path = Path::new("kernels/gemv_f32.cu");
            println!("cargo:rerun-if-changed={}", cu_path.display());
            let status = std::process::Command::new(&nvcc)
                .args(["-ptx", "-O3", "--use_fast_math", "-arch=sm_52"])
                .arg(cu_path)
                .arg("-o")
                .arg(&ptx_path)
                .status();
            match status {
                Ok(s) if s.success() => {
                    println!("cargo:rustc-env=OXIDIZE_CUDA_PTX={}", ptx_path.display());
                }
                Ok(s) => {
                    eprintln!("warning: nvcc PTX compilation failed with status {}", s);
                }
                Err(e) => {
                    eprintln!("warning: failed to run nvcc: {}", e);
                }
            }
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
    // The vulkan feature must be enabled for us to even check
    if env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return false;
    }

    // Check for VULKAN_SDK environment variable
    if env::var_os("VULKAN_SDK").is_some() {
        return true;
    }

    // Check for Vulkan loader on the system
    #[cfg(target_os = "linux")]
    {
        for path in [
            "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
            "/usr/lib64/libvulkan.so.1",
            "/usr/lib/libvulkan.so.1",
            "/lib/x86_64-linux-gnu/libvulkan.so.1",
            "/lib64/libvulkan.so.1",
        ] {
            if Path::new(path).exists() {
                return true;
            }
        }
        // Also check via pkg-config or ldconfig fallback
        if env::var_os("LD_LIBRARY_PATH").is_some() {
            // If LD_LIBRARY_PATH is set, user may have a custom Vulkan loader;
            // be optimistic when the feature is enabled.
            return true;
        }
    }

    #[cfg(target_os = "windows")]
    {
        for path in [
            "C:\\Windows\\System32\\vulkan-1.dll",
            "C:\\Windows\\SysWOW64\\vulkan-1.dll",
        ] {
            if Path::new(path).exists() {
                return true;
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        for path in [
            "/usr/local/lib/libvulkan.dylib",
            "/opt/homebrew/lib/libvulkan.dylib",
            "/usr/lib/libvulkan.dylib",
        ] {
            if Path::new(path).exists() {
                return true;
            }
        }
        // Check for MoltenVK
        if Path::new("/usr/local/lib/libMoltenVK.dylib").exists()
            || Path::new("/opt/homebrew/lib/libMoltenVK.dylib").exists()
        {
            return true;
        }
    }

    false
}

fn detect_mlx_available() -> bool {
    detect_metal_available()
}
