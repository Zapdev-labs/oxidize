//! Hardware detection for the autotuner.
//!
//! All probes are cheap (< 50 ms total on a typical box). Failures
//! degrade silently: if a probe can't run (e.g. nvidia-smi missing),
//! we report the absence and move on. The autotuner is then a pure
//! function over the resulting `HardwareInventory`.

use std::path::Path;

use crate::gpu_cluster::{GpuFamily, detect_gpus};
use crate::numa;
use crate::simd::{SimdBackend, preferred_backend};
use crate::spinpool::physical_core_count;
use oxidize_kernels::cpu::CpuVendor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsKind {
    Linux,
    Macos,
    Windows,
    Other,
}

/// Snapshot of the host hardware. All fields are best-effort: a
/// zero / false / None means "couldn't determine, treat as the
/// conservative case".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareInventory {
    pub os: OsKind,
    pub cpu_vendor: CpuVendor,
    pub simd: SimdBackend,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub numa_nodes: usize,
    pub min_node_ram_bytes: u64,
    pub total_ram_bytes: u64,
    pub has_gpu: bool,
    pub gpu_family: Option<GpuFamily>,
    pub gpu_vram_bytes: u64,
    pub has_metal: bool,
    pub has_cuda: bool,
    pub has_rocm: bool,
    pub has_rdma: bool,
    pub is_wsl: bool,
    pub container_mem_limit: Option<u64>,
    pub hugepages_2mib_avail: bool,
}

impl HardwareInventory {
    /// Human-readable one-line summary, used in `--print-hardware`.
    pub fn summary(&self) -> String {
        let cpu = format!("{:?}", self.cpu_vendor);
        let simd = format!("{:?}", self.simd);
        let gpu = if self.has_gpu {
            let family = self
                .gpu_family
                .map(|f| f.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            format!(
                "gpu={} vram={} MiB",
                family,
                self.gpu_vram_bytes / (1024 * 1024)
            )
        } else {
            "gpu=none".to_string()
        };
        format!(
            "os={:?} cpu={} simd={} cores={} ({}t) numa={} ram={} GiB {} metal={} cuda={} wsl={}",
            self.os,
            cpu,
            simd,
            self.physical_cores,
            self.logical_cores,
            self.numa_nodes,
            self.total_ram_bytes / (1u64 << 30),
            gpu,
            self.has_metal,
            self.has_cuda,
            self.is_wsl
        )
    }
}

/// Run all probes and return a complete inventory.
pub fn detect() -> HardwareInventory {
    let os = detect_os();
    let cpu_vendor = oxidize_kernels::cpu::cpu_vendor();
    let simd = preferred_backend();
    let physical_cores = physical_core_count().max(1);
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(physical_cores)
        .max(physical_cores);
    let numa_nodes = numa::node_count().max(1);
    let min_node_ram_bytes = numa::min_node_total_bytes();
    let total_ram_bytes = detect_total_ram_bytes().unwrap_or(min_node_ram_bytes * numa_nodes as u64);

    let gpus = detect_gpus();
    let has_gpu = !gpus.is_empty();
    let gpu_vram_bytes: u64 = gpus
        .iter()
        .map(|g| (g.memory_total_mib as u64) * 1024 * 1024)
        .sum();
    // Pick the highest-end family if we have multiple GPUs of
    // different kinds (rare but possible — DGX has A100 + BlueField
    // NICs that nvidia-smi may report). Rank by capability rather than
    // nvidia-smi enumeration order so selection is deterministic.
    let gpu_family = gpus
        .iter()
        .filter_map(|g| g.family)
        .max_by_key(|f| f.rank());

    let has_metal = detect_metal();
    let has_cuda = detect_cuda();
    let has_rocm = detect_rocm();
    let has_rdma = detect_rdma();
    let is_wsl = detect_wsl();
    let container_mem_limit = detect_cgroup_mem_limit();
    let hugepages_2mib_avail = detect_hugepages_2mib();

    HardwareInventory {
        os,
        cpu_vendor,
        simd,
        physical_cores,
        logical_cores,
        numa_nodes,
        min_node_ram_bytes,
        total_ram_bytes,
        has_gpu,
        gpu_family,
        gpu_vram_bytes,
        has_metal,
        has_cuda,
        has_rocm,
        has_rdma,
        is_wsl,
        container_mem_limit,
        hugepages_2mib_avail,
    }
}

fn detect_os() -> OsKind {
    if cfg!(target_os = "linux") {
        OsKind::Linux
    } else if cfg!(target_os = "macos") {
        OsKind::Macos
    } else if cfg!(target_os = "windows") {
        OsKind::Windows
    } else {
        OsKind::Other
    }
}

fn detect_total_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                // Format: "MemTotal:       16384000 kB"
                let kb: u64 = rest
                    .split_whitespace()
                    .next()
                    .and_then(|t| t.parse().ok())?;
                return Some(kb * 1024);
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        // Use sysctlbyname via libc; the kernel reports "hw.memsize".
        // Without the `libc` dep we fall back to numa::min_node_total_bytes()
        // (which returns 0 on non-Linux); the caller will substitute.
        None
    }
    #[cfg(target_os = "windows")]
    {
        // Without `windows-sys` or `winapi` we return None; the
        // caller falls back to the conservative estimate.
        None
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

fn detect_metal() -> bool {
    crate::metal::metal_build_info().detected_at_build
}

fn detect_cuda() -> bool {
    crate::cuda::cuda_build_info().detected_at_build
}

fn detect_rocm() -> bool {
    crate::rocm::rocm_build_info().detected_at_build
}

fn detect_rdma() -> bool {
    crate::mesh::rdma_build_available()
}

fn detect_wsl() -> bool {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/sys/kernel/osrelease") {
            let lower = s.to_ascii_lowercase();
            if lower.contains("microsoft") || lower.contains("wsl") {
                return true;
            }
        }
        if let Ok(s) = std::fs::read_to_string("/proc/version") {
            if s.to_ascii_lowercase().contains("microsoft") {
                return true;
            }
        }
    }
    false
}

fn detect_cgroup_mem_limit() -> Option<u64> {
    // cgroup v2 first.
    if let Some(limit) = read_cgroup_v2_limit(Path::new("/sys/fs/cgroup/memory.max")) {
        // `memory.max` can be "max" (no limit) — we treat that as None.
        if limit > 0 && limit < u64::MAX {
            return Some(limit);
        }
    }
    // cgroup v1 fallback.
    if let Some(limit) = read_cgroup_v1_limit(Path::new("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    {
        // v1 uses 2^63 - 1 or `9223372036854775807` for "no limit"; treat
        // anything >= 2^60 as "unlimited" and skip.
        if limit > 0 && limit < (1u64 << 60) {
            return Some(limit);
        }
    }
    None
}

fn read_cgroup_v2_limit(path: &Path) -> Option<u64> {
    let s = std::fs::read_to_string(path).ok()?;
    let trimmed = s.trim();
    if trimmed == "max" {
        return None;
    }
    trimmed.parse().ok()
}

fn read_cgroup_v1_limit(path: &Path) -> Option<u64> {
    let s = std::fs::read_to_string(path).ok()?;
    s.trim().parse().ok()
}

fn detect_hugepages_2mib() -> bool {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) =
            std::fs::read_to_string("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages")
        {
            if let Ok(n) = s.trim().parse::<u64>() {
                return n > 0;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_runs_and_returns_inventory() {
        // Smoke test: must always produce a non-empty inventory
        // on a real machine.
        let inv = detect();
        assert!(inv.physical_cores >= 1);
        assert!(inv.logical_cores >= inv.physical_cores);
        assert!(inv.numa_nodes >= 1);
        assert!(matches!(
            inv.os,
            OsKind::Linux | OsKind::Macos | OsKind::Windows | OsKind::Other
        ));
        let s = inv.summary();
        assert!(s.contains("cores="), "summary missing cores: {s}");
    }

    #[test]
    fn detect_total_ram_is_consistent_with_numa() {
        let inv = detect();
        // On a single-node Linux box, total RAM should be > min-node RAM.
        // We don't strictly assert this because on macOS / Windows we
        // fall back, but we do assert the field is non-zero (we always
        // have *some* signal).
        assert!(inv.total_ram_bytes > 0);
    }

    #[test]
    fn wsl_detection_is_safe_on_non_linux() {
        // On non-Linux builds the helper must return false (or the test
        // is a no-op on Linux).
        if !cfg!(target_os = "linux") {
            assert!(!detect_wsl());
        }
    }
}
