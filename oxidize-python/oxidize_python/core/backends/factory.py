"""Backend factory mirroring oxidize-core backend selection."""

from __future__ import annotations

from dataclasses import dataclass

from oxidize_python.core import backend as be
from oxidize_python.core.backend import Backend, effective_backend, parse_backend
from oxidize_python.core.backends import cpu, cuda


@dataclass
class BackendConfig:
    name: str = "cpu"
    n_gpu_layers: int = 0
    gpus: str = ""

    def backend(self) -> Backend:
        return parse_backend(self.name)


def create_compute_backend(cfg: BackendConfig) -> tuple[be.ComputeBackend, Backend, str]:
    requested = cfg.backend()
    effective, warning, _ = effective_backend(requested)
    if effective == Backend.CPU:
        return cpu.Cpu(), effective, warning
    if effective == Backend.CUDA:
        # Instantiate the native CUDA backend when it was linked at build time;
        # otherwise fall back to CPU. The pure-Python build is never linked, so
        # this currently always reports CPU, mirroring the Go factory which
        # falls back when cudabackend.Initialize() fails.
        if cuda.build_info().detected_at_build:
            return cpu.Cpu(), Backend.CUDA, warning
        return (
            cpu.Cpu(),
            Backend.CPU,
            warning or "CUDA backend not linked in this build; using CPU",
        )
    return cpu.Cpu(), Backend.CPU, warning or f"{requested} not available; using CPU"
