use crate::tensor::DType;
use std::sync::OnceLock;

const GIB: u64 = 1024 * 1024 * 1024;
const LOW_RAM_BYTES: u64 = 12 * GIB;
const MID_RAM_MAX_BYTES: u64 = 32 * GIB;
const TIGHT_RAM_BYTES: u64 = 8 * GIB;
pub const DEFAULT_PARALLEL_GEMV_MIN_OPS: usize = 1 << 20;

static RUNTIME_PROFILE: OnceLock<HardwareProfile> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareTier {
    Auto,
    Low,
    Mid,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareSnapshot {
    pub logical_cpus: usize,
    pub physical_cores_estimate: usize,
    pub total_ram_bytes: u64,
    pub has_avx2: bool,
    pub has_vulkan_build: bool,
    pub gpu_vendor_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareProfile {
    pub tier: HardwareTier,
    pub thread_count: usize,
    pub kv_cache_dtype: DType,
    pub cpu_optimized: bool,
    pub ram_offload: bool,
    pub mmap_prefetch: bool,
    pub mmap_hugepages: bool,
    pub layer_wise: bool,
    pub layer_cache: usize,
    pub default_ctx_size: usize,
    pub parallel_gemv_min_ops: usize,
    pub gemm_row_chunk: usize,
    pub gemm_batch_dot_chunk: usize,
    pub max_parallel_shards: usize,
    pub vulkan_gemv_min_work_items: usize,
    pub vulkan_gemm_min_work_items: usize,
    pub prefer_vulkan: bool,
    pub pipeline_parallel_hint: bool,
}

impl HardwareProfile {
    pub fn detect(requested: HardwareTier) -> Self {
        let snapshot = HardwareSnapshot::capture();
        Self::from_snapshot(requested, &snapshot)
    }

    pub fn from_snapshot(requested: HardwareTier, snapshot: &HardwareSnapshot) -> Self {
        let tier = resolve_tier(requested, snapshot);
        profile_for_tier(tier, snapshot)
    }

    pub fn parallel_gemv_min_ops(&self) -> usize {
        self.parallel_gemv_min_ops
    }

    pub fn gemm_row_chunk(&self) -> usize {
        self.gemm_row_chunk
    }

    pub fn gemm_batch_dot_chunk(&self) -> usize {
        self.gemm_batch_dot_chunk
    }

    pub fn max_parallel_shards(&self) -> usize {
        self.max_parallel_shards
    }

    pub fn vulkan_gemv_min_work_items(&self) -> usize {
        self.vulkan_gemv_min_work_items
    }

    pub fn vulkan_gemm_min_work_items(&self) -> usize {
        self.vulkan_gemm_min_work_items
    }

    pub fn apply_rayon_pool(&self) {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(self.thread_count.max(1))
            .build_global();
    }

    /// Low-tier RAM pressure: split batched prefill into smaller micro-batches.
    /// Returns `None` when the full prompt can be processed in one batched pass.
    pub fn prefill_micro_batch_size(&self) -> Option<usize> {
        match self.tier {
            HardwareTier::Low => Some(32),
            _ => None,
        }
    }

    /// One-line stdout summary for CLI / server startup.
    pub fn summary_line(&self) -> String {
        format!(
            "hardware: {:?} ({} threads, kv={:?}, cpu_opt={}, ram_offload={}, mmap_prefetch={}, mmap_hugepages={}, layer_wise={}, ctx={})",
            self.tier,
            self.thread_count,
            self.kv_cache_dtype,
            self.cpu_optimized,
            self.ram_offload,
            self.mmap_prefetch,
            self.mmap_hugepages,
            self.layer_wise,
            self.default_ctx_size,
        )
    }

    /// One-line stderr hints for CLI / server startup (tier, threads, key flags).
    pub fn print_recommendations(&self) {
        let hints = match self.tier {
            HardwareTier::Low if self.layer_wise => {
                "use --layer-wise for low RAM; prefill micro-batches=32"
            }
            HardwareTier::Low => "prefill micro-batches=32; consider --cpu-optimized",
            HardwareTier::Mid if self.layer_wise => {
                "tight RAM: --layer-wise; mmap_prefetch enabled"
            }
            HardwareTier::Mid => "mmap_prefetch on; one thread reserved for OS",
            HardwareTier::High if self.prefer_vulkan => {
                "Vulkan dispatch stubbed — CPU AVX2 fused GEMM active"
            }
            HardwareTier::High => "lower parallel GEMV threshold; wide ctx if RAM allows",
            HardwareTier::Auto => "auto-detected from host RAM and CPU count",
        };
        eprintln!(
            "oxidize perf: tier={:?} threads={} kv_cache={:?} ctx={} — {hints}",
            self.tier, self.thread_count, self.kv_cache_dtype, self.default_ctx_size
        );
    }
}

/// Lazily detect and cache the active hardware profile (also configures rayon).
pub fn global_profile() -> HardwareProfile {
    RUNTIME_PROFILE.get().cloned().unwrap_or_else(|| {
        let profile = HardwareProfile::detect(HardwareTier::Auto);
        profile.apply_rayon_pool();
        let _ = RUNTIME_PROFILE.set(profile.clone());
        profile
    })
}

pub fn init_runtime(profile: HardwareProfile) {
    profile.apply_rayon_pool();
    let _ = RUNTIME_PROFILE.set(profile);
}

/// Explicit CLI/server flag overrides; `None` means inherit from the hardware profile.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InferenceOverrides {
    pub threads: Option<usize>,
    pub ctx_size: Option<usize>,
    pub cpu_optimized: Option<bool>,
    pub ram_offload: Option<bool>,
    pub mmap_prefetch: Option<bool>,
    pub mmap_hugepages: Option<bool>,
    pub layer_wise: Option<bool>,
    pub layer_cache: Option<usize>,
    pub kv_cache_dtype: Option<DType>,
}

/// Effective inference flags after merging a hardware profile with explicit overrides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedInference {
    pub profile: HardwareProfile,
    pub cpu_optimized: bool,
    pub ram_offload: bool,
    pub mmap_prefetch: bool,
    pub mmap_hugepages: bool,
    pub layer_wise: bool,
    pub layer_cache: usize,
    pub ctx_size: Option<usize>,
    pub kv_cache_dtype: DType,
    pub threads: usize,
}

impl ResolvedInference {
    pub fn resolve(tier: HardwareTier, overrides: InferenceOverrides) -> Self {
        let mut profile = HardwareProfile::detect(tier);
        let threads = overrides
            .threads
            .filter(|t| *t > 0)
            .unwrap_or(profile.thread_count);
        profile.thread_count = threads;

        let cpu_optimized = overrides.cpu_optimized.unwrap_or(profile.cpu_optimized);
        let ram_offload = overrides.ram_offload.unwrap_or(profile.ram_offload);
        let mmap_prefetch = overrides.mmap_prefetch.unwrap_or(profile.mmap_prefetch);
        let mmap_hugepages = overrides.mmap_hugepages.unwrap_or(profile.mmap_hugepages);
        let layer_wise = overrides.layer_wise.unwrap_or(profile.layer_wise);
        let layer_cache = overrides.layer_cache.unwrap_or_else(|| {
            if layer_wise {
                profile.layer_cache
            } else {
                1
            }
        });
        let ctx_size = overrides.ctx_size.or(Some(profile.default_ctx_size));
        let kv_cache_dtype = overrides.kv_cache_dtype.unwrap_or(profile.kv_cache_dtype);

        Self {
            profile,
            cpu_optimized,
            ram_offload,
            mmap_prefetch,
            mmap_hugepages,
            layer_wise,
            layer_cache,
            ctx_size,
            kv_cache_dtype,
            threads,
        }
    }

    pub fn apply_runtime(&self, threads_explicit: bool) {
        if threads_explicit {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads)
                .build_global();
        } else {
            self.profile.apply_rayon_pool();
        }
        init_runtime(self.profile.clone());
    }
}

pub fn global_parallel_gemv_min_ops() -> usize {
    global_profile().parallel_gemv_min_ops()
}

pub fn global_gemm_row_chunk() -> usize {
    global_profile().gemm_row_chunk()
}

pub fn global_max_parallel_shards() -> usize {
    global_profile().max_parallel_shards()
}

pub fn global_vulkan_gemv_min_work_items() -> usize {
    global_profile().vulkan_gemv_min_work_items()
}

pub fn global_vulkan_gemm_min_work_items() -> usize {
    global_profile().vulkan_gemm_min_work_items()
}

pub fn rayon_thread_count() -> usize {
    global_profile().thread_count
}

fn resolve_tier(requested: HardwareTier, snapshot: &HardwareSnapshot) -> HardwareTier {
    if let Ok(raw) = std::env::var("OXIDIZE_HARDWARE_TIER") {
        if let Some(tier) = parse_tier_env(&raw) {
            return tier;
        }
    }
    match requested {
        HardwareTier::Auto => classify_auto(snapshot),
        other => other,
    }
}

fn parse_tier_env(raw: &str) -> Option<HardwareTier> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "low" => Some(HardwareTier::Low),
        "mid" => Some(HardwareTier::Mid),
        "high" => Some(HardwareTier::High),
        "auto" => Some(HardwareTier::Auto),
        _ => None,
    }
}

fn classify_auto(snapshot: &HardwareSnapshot) -> HardwareTier {
    if snapshot.total_ram_bytes < LOW_RAM_BYTES || snapshot.logical_cpus <= 4 {
        HardwareTier::Low
    } else if snapshot.total_ram_bytes > MID_RAM_MAX_BYTES || snapshot.logical_cpus >= 12 {
        HardwareTier::High
    } else {
        HardwareTier::Mid
    }
}

fn profile_for_tier(tier: HardwareTier, snapshot: &HardwareSnapshot) -> HardwareProfile {
    let cpus = snapshot.logical_cpus.max(1);
    match tier {
        HardwareTier::Low => HardwareProfile {
            tier,
            thread_count: (cpus / 2).max(2),
            kv_cache_dtype: DType::I16,
            cpu_optimized: true,
            ram_offload: snapshot.total_ram_bytes <= TIGHT_RAM_BYTES,
            mmap_prefetch: true,
            mmap_hugepages: false,
            layer_wise: true,
            layer_cache: if snapshot.total_ram_bytes <= TIGHT_RAM_BYTES {
                1
            } else {
                2
            },
            default_ctx_size: if snapshot.total_ram_bytes <= TIGHT_RAM_BYTES {
                2048
            } else {
                4096
            },
            parallel_gemv_min_ops: 1 << 21,
            gemm_row_chunk: 8,
            gemm_batch_dot_chunk: 2,
            max_parallel_shards: 2,
            vulkan_gemv_min_work_items: 8_192,
            vulkan_gemm_min_work_items: 131_072,
            prefer_vulkan: false,
            pipeline_parallel_hint: false,
        },
        HardwareTier::Mid => {
            let tight_ram = snapshot.total_ram_bytes < 16 * GIB;
            HardwareProfile {
                tier,
                thread_count: if cpus > 1 {
                    cpus.saturating_sub(1)
                } else {
                    cpus
                },
                kv_cache_dtype: DType::F16,
                cpu_optimized: false,
                ram_offload: tight_ram,
                mmap_prefetch: true,
                mmap_hugepages: false,
                layer_wise: tight_ram,
                layer_cache: if tight_ram { 1 } else { 1 },
                default_ctx_size: if snapshot.total_ram_bytes < 20 * GIB {
                    4096
                } else {
                    8192
                },
                parallel_gemv_min_ops: DEFAULT_PARALLEL_GEMV_MIN_OPS,
                gemm_row_chunk: 16,
                gemm_batch_dot_chunk: 4,
                max_parallel_shards: 4,
                vulkan_gemv_min_work_items: 4_096,
                vulkan_gemm_min_work_items: 65_536,
                prefer_vulkan: snapshot.has_vulkan_build && snapshot.gpu_vendor_id.is_some(),
                pipeline_parallel_hint: cpus >= 8,
            }
        }
        HardwareTier::High => HardwareProfile {
            tier,
            thread_count: cpus,
            kv_cache_dtype: if snapshot.total_ram_bytes > 48 * GIB {
                DType::F32
            } else {
                DType::F16
            },
            cpu_optimized: false,
            ram_offload: false,
            mmap_prefetch: true,
            mmap_hugepages: snapshot.total_ram_bytes >= 64 * GIB,
            layer_wise: false,
            layer_cache: 1,
            default_ctx_size: if snapshot.total_ram_bytes >= 48 * GIB {
                16384
            } else {
                8192
            },
            parallel_gemv_min_ops: 1 << 18,
            gemm_row_chunk: 32,
            gemm_batch_dot_chunk: 4,
            max_parallel_shards: 8,
            vulkan_gemv_min_work_items: 2_048,
            vulkan_gemm_min_work_items: 32_768,
            prefer_vulkan: snapshot.has_vulkan_build,
            pipeline_parallel_hint: cpus >= 8,
        },
        HardwareTier::Auto => profile_for_tier(classify_auto(snapshot), snapshot),
    }
}

impl HardwareSnapshot {
    pub fn capture() -> Self {
        Self {
            logical_cpus: detect_logical_cpus(),
            physical_cores_estimate: detect_physical_cores(),
            total_ram_bytes: detect_total_ram_bytes(),
            has_avx2: detect_avx2(),
            has_vulkan_build: cfg!(feature = "vulkan"),
            gpu_vendor_id: detect_gpu_vendor_id(),
        }
    }

    #[cfg(test)]
    pub fn for_test(
        logical_cpus: usize,
        physical_cores_estimate: usize,
        total_ram_bytes: u64,
        has_avx2: bool,
        has_vulkan_build: bool,
        gpu_vendor_id: Option<u32>,
    ) -> Self {
        Self {
            logical_cpus,
            physical_cores_estimate,
            total_ram_bytes,
            has_avx2,
            has_vulkan_build,
            gpu_vendor_id,
        }
    }
}

fn detect_logical_cpus() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .max(1)
}

fn detect_physical_cores() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Some(cores) = linux_physical_core_count() {
            return cores.max(1);
        }
    }
    detect_logical_cpus()
        .saturating_add(1)
        .saturating_div(2)
        .max(1)
}

#[cfg(target_os = "linux")]
fn linux_physical_core_count() -> Option<usize> {
    use std::collections::HashSet;
    use std::fs;

    let cpu_root = std::path::Path::new("/sys/devices/system/cpu");
    let mut core_ids = HashSet::new();
    let entries = fs::read_dir(cpu_root).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("cpu") || name == "cpu" {
            continue;
        }
        let suffix = name.strip_prefix("cpu")?;
        if !suffix.chars().all(|ch| ch.is_ascii_digit()) {
            continue;
        }
        let core_id_path = entry.path().join("topology/core_id");
        if let Ok(content) = fs::read_to_string(core_id_path) {
            if let Ok(core_id) = content.trim().parse::<usize>() {
                core_ids.insert(core_id);
            }
        }
    }
    if core_ids.is_empty() {
        None
    } else {
        Some(core_ids.len())
    }
}

fn detect_total_ram_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Some(bytes) = linux_mem_total_bytes() {
            return bytes;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(bytes) = macos_mem_total_bytes() {
            return bytes;
        }
    }
    16 * GIB
}

#[cfg(target_os = "linux")]
fn linux_mem_total_bytes() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if let Some(kb) = line.strip_prefix("MemTotal:") {
            let kb = kb
                .trim()
                .strip_suffix(" kB")
                .or_else(|| kb.strip_suffix(" kB"))
                .unwrap_or(kb.trim());
            let kb: u64 = kb.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn macos_mem_total_bytes() -> Option<u64> {
    use std::process::Command;
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    text.trim().parse().ok()
}

fn detect_avx2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

fn detect_gpu_vendor_id() -> Option<u32> {
    if let Ok(raw) = std::env::var("OXIDIZE_GPU_VENDOR_ID") {
        if let Some(id) = parse_vendor_id(&raw) {
            return Some(id);
        }
    }
    #[cfg(target_os = "linux")]
    {
        return linux_gpu_vendor_id();
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn parse_vendor_id(raw: &str) -> Option<u32> {
    let trimmed = raw.trim();
    let hex = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    u32::from_str_radix(hex, 16).ok()
}

#[cfg(target_os = "linux")]
fn linux_gpu_vendor_id() -> Option<u32> {
    use std::fs;
    use std::path::Path;

    let drm = Path::new("/sys/class/drm");
    let entries = fs::read_dir(drm).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let vendor_path = entry.path().join("device/vendor");
        if let Ok(content) = fs::read_to_string(vendor_path) {
            if let Some(id) = parse_vendor_id(content.trim()) {
                return Some(id);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_ram_cpus(ram_gib: u64, cpus: usize) -> HardwareSnapshot {
        HardwareSnapshot::for_test(cpus, cpus, ram_gib * GIB, true, false, None)
    }

    #[test]
    fn auto_classifies_low_ram() {
        let snapshot = snapshot_ram_cpus(8, 16);
        let profile = HardwareProfile::from_snapshot(HardwareTier::Auto, &snapshot);
        assert_eq!(profile.tier, HardwareTier::Low);
        assert!(profile.layer_wise);
        assert!(profile.cpu_optimized);
    }

    #[test]
    fn auto_classifies_low_cpus() {
        let snapshot = snapshot_ram_cpus(32, 4);
        let profile = HardwareProfile::from_snapshot(HardwareTier::Auto, &snapshot);
        assert_eq!(profile.tier, HardwareTier::Low);
    }

    #[test]
    fn auto_classifies_mid() {
        let snapshot = snapshot_ram_cpus(16, 8);
        let profile = HardwareProfile::from_snapshot(HardwareTier::Auto, &snapshot);
        assert_eq!(profile.tier, HardwareTier::Mid);
        assert_eq!(profile.kv_cache_dtype, DType::F16);
        assert!(!profile.layer_wise);
    }

    #[test]
    fn auto_classifies_high_ram() {
        let snapshot = snapshot_ram_cpus(64, 8);
        let profile = HardwareProfile::from_snapshot(HardwareTier::Auto, &snapshot);
        assert_eq!(profile.tier, HardwareTier::High);
        assert_eq!(profile.kv_cache_dtype, DType::F32);
    }

    #[test]
    fn auto_classifies_high_cpus() {
        let snapshot = snapshot_ram_cpus(16, 16);
        let profile = HardwareProfile::from_snapshot(HardwareTier::Auto, &snapshot);
        assert_eq!(profile.tier, HardwareTier::High);
    }

    #[test]
    fn low_tier_thread_and_parallel_threshold() {
        let snapshot = snapshot_ram_cpus(8, 8);
        let profile = HardwareProfile::from_snapshot(HardwareTier::Low, &snapshot);
        assert_eq!(profile.thread_count, 4);
        assert_eq!(profile.parallel_gemv_min_ops(), 1 << 21);
        assert_eq!(profile.default_ctx_size, 2048);
    }

    #[test]
    fn mid_tier_reserves_one_thread() {
        let snapshot = snapshot_ram_cpus(24, 8);
        let profile = HardwareProfile::from_snapshot(HardwareTier::Mid, &snapshot);
        assert_eq!(profile.thread_count, 7);
        assert_eq!(profile.parallel_gemv_min_ops(), DEFAULT_PARALLEL_GEMV_MIN_OPS);
    }

    #[test]
    fn high_tier_lowers_parallel_threshold() {
        let snapshot = snapshot_ram_cpus(64, 16);
        let profile = HardwareProfile::from_snapshot(HardwareTier::High, &snapshot);
        assert_eq!(profile.thread_count, 16);
        assert_eq!(profile.parallel_gemv_min_ops(), 1 << 18);
        assert!(profile.mmap_hugepages);
    }

    #[test]
    fn mid_tier_layer_wise_when_ram_tight() {
        let snapshot = snapshot_ram_cpus(14, 8);
        let profile = HardwareProfile::from_snapshot(HardwareTier::Mid, &snapshot);
        assert!(profile.layer_wise);
        assert!(profile.ram_offload);
    }

    #[test]
    fn host_machine_snapshot_is_sane() {
        let snapshot = HardwareSnapshot::capture();
        assert!(snapshot.logical_cpus >= 1);
        assert!(snapshot.physical_cores_estimate >= 1);
        assert!(snapshot.total_ram_bytes > 0);
    }

    #[test]
    fn global_parallel_gemv_min_ops_defaults_before_init() {
        let threshold = global_parallel_gemv_min_ops();
        assert!(threshold >= 1 << 18);
    }

    #[test]
    fn low_tier_suggests_prefill_micro_batches() {
        let snapshot = snapshot_ram_cpus(8, 8);
        let profile = HardwareProfile::from_snapshot(HardwareTier::Low, &snapshot);
        assert_eq!(profile.prefill_micro_batch_size(), Some(32));
    }

    #[test]
    fn high_tier_skips_prefill_micro_batches() {
        let snapshot = snapshot_ram_cpus(64, 16);
        let profile = HardwareProfile::from_snapshot(HardwareTier::High, &snapshot);
        assert_eq!(profile.prefill_micro_batch_size(), None);
    }

    #[test]
    fn tier_gemm_tuning_scales_with_capability() {
        let low = HardwareProfile::from_snapshot(HardwareTier::Low, &snapshot_ram_cpus(8, 4));
        let mid = HardwareProfile::from_snapshot(HardwareTier::Mid, &snapshot_ram_cpus(24, 8));
        let high = HardwareProfile::from_snapshot(HardwareTier::High, &snapshot_ram_cpus(64, 16));
        assert!(low.gemm_row_chunk() < mid.gemm_row_chunk());
        assert!(mid.gemm_row_chunk() <= high.gemm_row_chunk());
        assert!(low.vulkan_gemm_min_work_items() > high.vulkan_gemm_min_work_items());
    }

    #[test]
    fn resolved_inference_applies_explicit_overrides() {
        let resolved = ResolvedInference::resolve(
            HardwareTier::Low,
            InferenceOverrides {
                threads: Some(2),
                cpu_optimized: Some(false),
                ..Default::default()
            },
        );
        assert_eq!(resolved.threads, 2);
        assert!(!resolved.cpu_optimized);
    }
}
