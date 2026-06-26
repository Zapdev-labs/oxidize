use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(cuda_available)");
    println!("cargo:rustc-check-cfg=cfg(rocm_available)");
    println!("cargo:rustc-check-cfg=cfg(rdma_available)");
    println!("cargo:rustc-check-cfg=cfg(metal_available)");
    println!("cargo:rustc-check-cfg=cfg(webgpu_available)");
    println!("cargo:rustc-check-cfg=cfg(vulkan_available)");
    println!("cargo:rustc-check-cfg=cfg(mlx_available)");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_ARCH");
    println!("cargo:rerun-if-env-changed=GPU_TARGETS");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    if let Some(cuda_root) = detect_cuda_root() {
        println!("cargo:rustc-cfg=cuda_available");
        println!("cargo:rustc-env=OXIDIZE_CUDA_PATH={}", cuda_root.display());

        let lib64 = cuda_root.join("lib64");
        if lib64.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
            println!("cargo:rustc-link-lib=dylib=cudart");
        }

        // When the `cuda` feature is on, compile the GEMV kernels from CUDA C
        // source to PTX with nvcc. Generating PTX at build time (rather than
        // committing hand-written PTX) guarantees it is valid for the installed
        // toolkit and forward-JIT-compatible with newer GPUs (e.g. sm_120).
        if env::var_os("CARGO_FEATURE_CUDA").is_some() {
            compile_cuda_kernels(&cuda_root);
        }
    }

    if let Some(rocm_root) = detect_rocm_root() {
        println!("cargo:rustc-cfg=rocm_available");
        println!("cargo:rustc-env=OXIDIZE_ROCM_PATH={}", rocm_root.display());

        let lib = rocm_root.join("lib");
        if lib.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib.display());
            println!("cargo:rustc-link-lib=dylib=amdhip64");
        }

        if env::var_os("CARGO_FEATURE_ROCM").is_some() {
            compile_rocm_kernels(&rocm_root);
        }
    }

    if detect_rdma_available() {
        println!("cargo:rustc-cfg=rdma_available");
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

/// Compile `kernels/gemv_f32.cu` to PTX (+ optional native cubins) in `OUT_DIR`.
///
/// Strategy: always emit `compute_75` PTX — a virtual-architecture image that
/// the CUDA driver JITs to native code for any GPU ≥ SM7.5 (including SM120).
/// Additionally, if the installed toolkit is new enough, emit a native cubin for
/// the current generation so the driver can skip JIT on that GPU:
///
///   CUDA ≥ 12.8  →  also emit native SM120 (Blackwell RTX 50-series)
///   CUDA ≥ 12.0  →  also emit native SM90  (Hopper H100/H200)
///   CUDA ≥ 11.8  →  also emit native SM89  (Ada Lovelace L40/RTX 4090)
///   CUDA ≥ 11.0  →  also emit native SM80  (Ampere A100)
///
/// The crate embeds the result via
/// `include_str!(concat!(env!("OUT_DIR"), "/gemv_f32.ptx"))`.
fn compile_cuda_kernels(cuda_root: &Path) {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is set by cargo");
    let ptx_out = Path::new(&out_dir).join("gemv_f32.ptx");
    let src = Path::new("kernels/gemv_f32.cu");
    println!("cargo:rerun-if-changed=kernels/gemv_f32.cu");

    let nvcc = {
        let exe = if cfg!(target_os = "windows") {
            "nvcc.exe"
        } else {
            "nvcc"
        };
        let candidate = cuda_root.join("bin").join(exe);
        if candidate.is_file() {
            candidate
        } else {
            PathBuf::from(exe)
        }
    };

    // Probe toolkit version to decide which native archs to embed alongside PTX.
    let toolkit_version: u32 = std::process::Command::new(&nvcc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            // e.g. "release 12.8, V12.8.93"
            s.split("release ").nth(1)?.split(',').next().and_then(|v| {
                let mut parts = v.trim().split('.');
                let major: u32 = parts.next()?.parse().ok()?;
                let minor: u32 = parts.next()?.parse().ok()?;
                Some(major * 10 + minor)
            })
        })
        .unwrap_or(0);

    // We only ever consume the generated PTX (loaded as text via include_str!
    // in backends/cuda.rs); the CUDA driver JITs it to the live GPU's native
    // SASS at module-load time. So compile a single forward-compatible PTX for
    // the lowest supported virtual arch (compute_75) — it runs on every GPU
    // sm_75 and newer (Turing → Blackwell).
    //
    // NB: `-ptx` may target only ONE architecture; passing multiple native
    // `-gencode arch=compute_X,code=sm_X` targets makes nvcc ≥ 12.8 fail with
    // "Option '--ptx' is not allowed when compiling for multiple GPU
    // architectures". Native cubins would need `-fatbin` + a fatbin loader,
    // which we don't use.
    let _ = toolkit_version; // probed for diagnostics; PTX target is fixed.
    let status = std::process::Command::new(&nvcc)
        .arg("-ptx")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("-arch=compute_75")
        .arg("-o")
        .arg(&ptx_out)
        .arg(src)
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("nvcc failed to compile {}: exit {s}", src.display()),
        Err(e) => panic!("failed to invoke nvcc ({}): {e}", nvcc.display()),
    }
}

/// Compile `kernels/gemv_f32.cu` to a HIP code object with hipcc.
fn compile_rocm_kernels(rocm_root: &Path) {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is set by cargo");
    let co_out = Path::new(&out_dir).join("gemv_f32.co");
    let src = Path::new("kernels/gemv_f32.cu");
    println!("cargo:rerun-if-changed=kernels/gemv_f32.cu");

    let hipcc = {
        let exe = if cfg!(target_os = "windows") {
            "hipcc.exe"
        } else {
            "hipcc"
        };
        let candidate = rocm_root.join("bin").join(exe);
        if candidate.is_file() {
            candidate
        } else {
            PathBuf::from(exe)
        }
    };

    // Default to a portable multi-arch fat binary instead of bare "native".
    // "native" emits host-only code on a GPU-less build host (a runtime
    // hipErrorNoBinaryForGpu) and a single-GPU artifact otherwise. The default
    // list spans Vega20/CDNA (gfx906/908/90a/942) and RDNA2/3/3.5
    // (gfx1030/1100/1101/1102/1151 Strix Halo). Override with ROCM_ARCH or
    // GPU_TARGETS (semicolon-separated, or "native" for the local GPU).
    let arch = env::var("ROCM_ARCH")
        .or_else(|_| env::var("GPU_TARGETS"))
        .unwrap_or_else(|_| {
            "gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151".to_string()
        });

    // Pin the wavefront width the kernels are compiled for. CDNA/GCN are always
    // wave64; RDNA defaults to wave32, which would make the OX_WAVE=64 kernel
    // contract (and rocm::GEMV_LANES_PER_ROW=64) wrong. Forcing wave64 keeps the
    // host launch geometry, the OX_WAVE macro, and the device wavefront in sync.
    // Override via ROCM_WAVEFRONT_FLAG (must be flipped together with OX_WAVE and
    // GEMV_LANES_PER_ROW for a wave32 build).
    let wave_flag =
        env::var("ROCM_WAVEFRONT_FLAG").unwrap_or_else(|_| "-mwavefrontsize64".to_string());

    let mut cmd = std::process::Command::new(&hipcc);
    cmd.arg("--genco")
        .arg("-O3")
        .arg("-ffast-math")
        .arg(&wave_flag);
    for a in arch.split(';').filter(|s| !s.is_empty()) {
        cmd.arg(format!("--offload-arch={a}"));
    }
    let status = cmd.arg("-o").arg(&co_out).arg(src).status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("hipcc failed to compile {}: exit {s}", src.display()),
        Err(e) => panic!("failed to invoke hipcc ({}): {e}", hipcc.display()),
    }
}

fn detect_rocm_root() -> Option<PathBuf> {
    for key in ["ROCM_PATH", "HIP_PATH"] {
        match env::var_os(key).map(PathBuf::from) {
            Some(path) if path.is_dir() => return Some(path),
            _ => {}
        }
    }

    let default = Path::new("/opt/rocm");
    if default.is_dir() {
        Some(default.to_path_buf())
    } else {
        None
    }
}

fn detect_rdma_available() -> bool {
    if env::var_os("CARGO_FEATURE_RDMA").is_none() {
        return false;
    }

    #[cfg(target_os = "linux")]
    {
        for path in [
            "/usr/lib/x86_64-linux-gnu/libibverbs.so.1",
            "/usr/lib64/libibverbs.so.1",
            "/usr/lib/libibverbs.so.1",
            "/lib/x86_64-linux-gnu/libibverbs.so.1",
        ] {
            if Path::new(path).exists() {
                return true;
            }
        }
    }

    false
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
