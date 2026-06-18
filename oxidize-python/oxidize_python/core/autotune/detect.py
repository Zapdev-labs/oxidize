"""Hardware detection for autotune (mirrors oxidize-golang/core/autotune/detect.go)."""

from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from oxidize_python.gpucluster import GpuFamily, DetectedGpu, detect_gpus
from oxidize_python.core.simd.simd import Backend, preferred


class OsKind(Enum):
    LINUX = auto()
    MACOS = auto()
    WINDOWS = auto()
    OTHER = auto()


class CpuVendor(Enum):
    UNKNOWN = auto()
    INTEL = auto()
    AMD = auto()
    ARM = auto()


@dataclass
class HardwareInventory:
    os: OsKind
    cpu_vendor: CpuVendor
    simd: Backend
    physical_cores: int
    logical_cores: int
    numa_nodes: int
    min_node_ram_bytes: int
    total_ram_bytes: int
    has_gpu: bool
    gpu_family: Optional[GpuFamily]
    gpu_vram_bytes: int
    has_metal: bool
    has_cuda: bool
    has_rocm: bool
    has_rdma: bool
    is_wsl: bool
    container_mem_limit: Optional[int]
    hugepages_2mib_avail: bool

    def summary(self) -> str:
        gpu = "gpu=none"
        if self.has_gpu:
            fam = self.gpu_family.name.lower() if self.gpu_family else "unknown"
            gpu = f"gpu={fam} vram={self.gpu_vram_bytes // (1024 * 1024)} MiB"
        return (
            f"os={self.os.name} cpu={self.cpu_vendor.name} simd={self.simd.name} "
            f"cores={self.physical_cores} ({self.logical_cores}t) numa={self.numa_nodes} "
            f"ram={self.total_ram_bytes // (1 << 30)} GiB {gpu} "
            f"metal={self.has_metal} cuda={self.has_cuda} wsl={self.is_wsl}"
        )


def detect() -> HardwareInventory:
    os_kind = _detect_os()
    physical = os.cpu_count() or 1
    logical = physical
    min_node = 4 << 30
    total = _detect_total_ram_bytes() or min_node

    gpus = detect_gpus()
    has_gpu = len(gpus) > 0
    vram = sum(int(g.memory_total_mib) * 1024 * 1024 for g in gpus)
    fam: Optional[GpuFamily] = None
    for g in gpus:
        if g.family is not None and fam is None:
            fam = g.family

    return HardwareInventory(
        os=os_kind,
        cpu_vendor=_detect_cpu_vendor(),
        simd=preferred(),
        physical_cores=physical,
        logical_cores=logical,
        numa_nodes=_detect_numa_nodes(),
        min_node_ram_bytes=min_node,
        total_ram_bytes=total,
        has_gpu=has_gpu,
        gpu_family=fam,
        gpu_vram_bytes=vram,
        has_metal=platform.system() == "Darwin",
        has_cuda=has_gpu,
        has_rocm=False,
        has_rdma=False,
        is_wsl=_detect_wsl(),
        container_mem_limit=_detect_cgroup_mem_limit(),
        hugepages_2mib_avail=_detect_hugepages_2mib(),
    )


def is_skylake_sp() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        data = open("/proc/cpuinfo", encoding="utf-8").read().lower()
    except OSError:
        return False
    return "skylake" in data and "xeon" in data


def _detect_os() -> OsKind:
    system = platform.system()
    if system == "Linux":
        return OsKind.LINUX
    if system == "Darwin":
        return OsKind.MACOS
    if system == "Windows":
        return OsKind.WINDOWS
    return OsKind.OTHER


def _detect_total_ram_bytes() -> int:
    if platform.system() != "Linux":
        return 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except OSError:
        return 0
    return 0


def _detect_cpu_vendor() -> CpuVendor:
    machine = platform.machine().lower()
    if machine.startswith("arm") or machine.startswith("aarch"):
        return CpuVendor.ARM
    if platform.system() != "Linux":
        return CpuVendor.UNKNOWN
    try:
        data = open("/proc/cpuinfo", encoding="utf-8").read().lower()
    except OSError:
        return CpuVendor.UNKNOWN
    if "authenticamd" in data:
        return CpuVendor.AMD
    if "genuineintel" in data:
        return CpuVendor.INTEL
    return CpuVendor.UNKNOWN


def _detect_numa_nodes() -> int:
    if platform.system() != "Linux":
        return 1
    try:
        nodes = [n for n in os.listdir("/sys/devices/system/node") if n.startswith("node")]
        return max(len(nodes), 1)
    except OSError:
        return 1


def _detect_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    for path in ("/proc/sys/kernel/osrelease", "/proc/version"):
        try:
            data = open(path, encoding="utf-8").read().lower()
        except OSError:
            continue
        if "microsoft" in data or "wsl" in data:
            return True
    return False


def _detect_cgroup_mem_limit() -> Optional[int]:
    if platform.system() != "Linux":
        return None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path, encoding="utf-8").read().strip()
        except OSError:
            continue
        if raw in ("", "max"):
            continue
        try:
            n = int(raw)
        except ValueError:
            continue
        if 0 < n < (1 << 60):
            return n
    return None


def _detect_hugepages_2mib() -> bool:
    path = "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"
    try:
        n = int(open(path, encoding="utf-8").read().strip())
        return n > 0
    except (OSError, ValueError):
        return False
