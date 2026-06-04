"""Backend factory mirroring oxidize-core backend selection."""

from __future__ import annotations

from dataclasses import dataclass

from oxidize_python.core import backend as be
from oxidize_python.core.backend import Backend, effective_backend, parse_backend
from oxidize_python.core.backends import cpu


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
    return cpu.Cpu(), Backend.CPU, warning or f"{requested} not available; using CPU"
