# oxidize-cpp / modal / image.py
#
# Modal image definition shared by benchmark.py.
#
# Builds BOTH engines from the repository that contains this file:
#   1. The C++ engine (oxidize-cpp) via CMake with -DOXIDIZE_CUDA=ON
#        -> /opt/oxidize-cpp/build/oxidize-cpp
#   2. The Rust 'oxidize' CLI via `cargo build --release -p oxidize-cli`
#        -> /opt/oxidize/target/release/oxidize
#
# Base: nvidia/cuda:12.4.1-devel-ubuntu22.04 (has nvcc + cuBLAS headers for the
# CUDA backend; sm_80 = A100, sm_90 = H100 are baked into CMakeLists.txt).
#
# The repository root is uploaded into the image at build time so the build is
# fully self-contained (no network checkout). REPO_ROOT is the workspace that
# holds both `oxidize-cpp/` and `oxidize-cli/`.

import os
import pathlib

import modal

# This file lives at <repo>/oxidize-cpp/modal/image.py.
# parents[0]=modal, [1]=oxidize-cpp, [2]=<repo root with oxidize-cli + oxidize-cpp>.
_THIS = pathlib.Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]

# Where the engines live, and where artifacts land, inside the image.
IMAGE_REPO = "/opt/oxidize-src"
CPP_DIR = f"{IMAGE_REPO}/oxidize-cpp"
CPP_BUILD = f"{CPP_DIR}/build"
CPP_BIN = f"{CPP_BUILD}/oxidize-cpp"
RUST_BIN = f"{IMAGE_REPO}/target/release/oxidize"

# Mount point for the user-supplied GGUF models volume.
MODELS_VOLUME_NAME = os.environ.get("OXIDIZE_MODELS_VOLUME", "oxidize-models")
MODELS_MOUNT = "/models"

models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)


def _build_commands() -> list[str]:
    return [
        # ---- C++ engine: CMake + CUDA backend (REQUIRED) ------------------
        # nvcc compiles here at image-build time (CPU) so CUDA compile errors
        # surface without ever touching a paid GPU.
        f"cmake -S {CPP_DIR} -B {CPP_BUILD} "
        f"-DCMAKE_BUILD_TYPE=Release -DOXIDIZE_CUDA=ON "
        f"-DCMAKE_CUDA_ARCHITECTURES='80;90'",
        f"cmake --build {CPP_BUILD} --target oxidize-cpp -j",
        f"test -x {CPP_BIN}",  # hard gate: the C++ CUDA engine MUST build
        # ---- Rust 'oxidize' CLI (BEST-EFFORT baseline) --------------------
        # Non-fatal: the Rust workspace is large and the GPU baseline is
        # optional. A failure here must not block the C++ image.
        f"bash -lc 'source $HOME/.cargo/env && cd {IMAGE_REPO} && "
        f"cargo build --release -p oxidize-cli "
        f"|| echo OXIDIZE_RUST_BUILD_FAILED'",
        f"bash -lc 'test -x {RUST_BIN} && echo rust-ok || "
        f"echo \"rust baseline unavailable (non-fatal)\"'",
    ]


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "g++",
        "cmake",
        "ninja-build",
        "git",
        "curl",
        "ca-certificates",
        "pkg-config",
        "libssl-dev",
        "libomp-dev",
    )
    .pip_install("huggingface_hub")
    # Rust toolchain (stable; edition-2024 capable). Pin via RUST_TOOLCHAIN if needed.
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --default-toolchain stable --profile minimal",
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "CUDACXX": "/usr/local/cuda/bin/nvcc",
            "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
        }
    )
    # Upload the whole workspace so both engines build from identical source.
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=IMAGE_REPO,
        ignore=[
            "**/target/**",
            "**/build/**",
            "**/.git/**",
            "**/models/**",
            "**/*.gguf",
            "**/node_modules/**",
            # Heavy non-Rust trees not needed to build either engine.
            "oxidize-golang/**",
            "oxidize-python/**",
            "dist/**",
            "evidence/**",
            "results/**",
            "docs/**",
            ".firecrawl/**",
            "**/__pycache__/**",
        ],
        copy=True,  # needed so subsequent run_commands can compile the source
    )
    .run_commands(*_build_commands())
)
