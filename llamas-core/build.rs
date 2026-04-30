use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(cuda_available)");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if let Some(cuda_root) = detect_cuda_root() {
        println!("cargo:rustc-cfg=cuda_available");
        println!("cargo:rustc-env=LLAMAS_CUDA_PATH={}", cuda_root.display());

        let lib64 = cuda_root.join("lib64");
        if lib64.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
            println!("cargo:rustc-link-lib=dylib=cudart");
        }
    }
}

fn detect_cuda_root() -> Option<PathBuf> {
    for key in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(path) = env::var_os(key).map(PathBuf::from)
            && path.is_dir()
        {
            return Some(path);
        }
    }

    let default = Path::new("/usr/local/cuda");
    if default.is_dir() {
        Some(default.to_path_buf())
    } else {
        None
    }
}
