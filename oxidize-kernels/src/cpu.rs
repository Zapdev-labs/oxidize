//! CPU vendor / ISA detection and per-vendor kernel tuning.
//!
//! Q4_K decode GEMV is DRAM-bandwidth bound, so the per-vendor levers are in
//! the memory pipeline, not the ALU sequence: software-prefetch distance,
//! cache hint, and whether to use the wider AVX-512 instructions on parts
//! where they help more than they hurt.

use std::sync::OnceLock;

use crate::BLOCK_Q4_K_SIZE;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpuVendor {
    Intel,
    Amd,
    Other,
}

/// Snapshot of the CPU we are running on.
#[derive(Clone, Copy, Debug)]
pub struct CpuInfo {
    pub vendor: CpuVendor,
    pub family: u32,
    pub model: u32,
    pub stepping: u32,
    pub has_avx2: bool,
    pub has_fma: bool,
    pub has_avx512f: bool,
    pub has_avx512bw: bool,
    pub has_avx512vnni: bool,
    pub has_avxvnni: bool,
    /// Kernel-selected default: use AVX-512F/BW path when available.  The
    /// default is conservative (false on Skylake-SP because AVX-512 tends to
    /// down-clock, true on newer Intel cores where it is a clear win).  Users
    /// can override with `OXIDIZE_OXK_AVX512=1|0`.
    pub use_avx512: bool,
}

/// Memory-pipeline tuning consumed by the SIMD kernels.
#[derive(Clone, Copy, Debug)]
pub struct OxkTune {
    /// Prefetch distance in bytes ahead of the current weight block pointer
    /// (multiple of `BLOCK_Q4_K_SIZE`; 0 disables software prefetch).
    pub pf_bytes: usize,
    /// Prefetch with `_MM_HINT_NTA` instead of `_MM_HINT_T0`.
    pub pf_nta: bool,
}

pub fn cpu_vendor() -> CpuVendor {
    cpuinfo().vendor
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpuid_leaf(leaf: u32) -> (u32, u32, u32, u32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::__cpuid;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::__cpuid;
    let r = __cpuid(leaf);
    (r.eax, r.ebx, r.ecx, r.edx)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpuid_leaf_sub(leaf: u32, sub: u32) -> (u32, u32, u32, u32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::__cpuid_count;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::__cpuid_count;
    let r = __cpuid_count(leaf, sub);
    (r.eax, r.ebx, r.ecx, r.edx)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_cpuinfo() -> CpuInfo {
    let (_, ebx0, ecx0, edx0) = cpuid_leaf(0);
    let mut v = [0_u8; 12];
    v[0..4].copy_from_slice(&ebx0.to_le_bytes());
    v[4..8].copy_from_slice(&edx0.to_le_bytes());
    v[8..12].copy_from_slice(&ecx0.to_le_bytes());
    let vendor = match &v {
        b"GenuineIntel" => CpuVendor::Intel,
        b"AuthenticAMD" => CpuVendor::Amd,
        _ => CpuVendor::Other,
    };

    let (eax1, _, _, _) = cpuid_leaf(1);
    let base_family = (eax1 >> 8) & 0xf;
    let base_model = (eax1 >> 4) & 0xf;
    let family = if base_family == 0xf {
        base_family + ((eax1 >> 20) & 0xff)
    } else {
        base_family
    };
    let model = if base_family == 0x6 || base_family == 0xf {
        (base_model & 0xf) | ((eax1 >> 12) & 0xf0)
    } else {
        base_model
    };
    let stepping = eax1 & 0xf;

    let (_, ebx7, ecx7, _) = cpuid_leaf_sub(7, 0);
    let has_avx2 = std::arch::is_x86_feature_detected!("avx2");
    let has_fma = std::arch::is_x86_feature_detected!("fma");
    let has_avx512f = (ebx7 >> 16) & 1 != 0;
    let has_avx512bw = (ebx7 >> 30) & 1 != 0;
    let has_avx512vnni = (ecx7 >> 11) & 1 != 0;
    // VEX-encoded AVX-VNNI (Alder Lake+, Zen 4+) is reported in leaf 7
    // subleaf 1, EAX bit 4 — NOT leaf 7 subleaf 0 EDX bit 4 (which is
    // FSRM/other).
    let (eax7_1, _, _, _) = cpuid_leaf_sub(7, 1);
    let has_avxvnni = (eax7_1 >> 4) & 1 != 0;

    // Default AVX-512 enablement: only when it has VNNI (where the ISA is a
    // clear win) or on parts where the wider register alone has proven useful.
    // Skylake-SP / Xeon Silver keeps AVX2 default unless the user opts in,
    // because AVX-512 without VNNI often loses to AVX2 under sustained decode
    // due to frequency drop.
    let mut use_avx512 = match (vendor, family, model) {
        (CpuVendor::Intel, 6, m) if matches!(m, 106 | 108 | 126 | 143 | 207) && has_avx512vnni => {
            true
        }
        (CpuVendor::Intel, 6, m) if matches!(m, 85 | 86) && has_avx512f && has_avx512bw => {
            // Skylake-SP / Skylake-X: keep AVX2 default, but allow override.
            false
        }
        _ => false,
    };
    if let Ok(v) = std::env::var("OXIDIZE_OXK_AVX512") {
        use_avx512 = v == "1" || v.eq_ignore_ascii_case("true");
    }

    CpuInfo {
        vendor,
        family,
        model,
        stepping,
        has_avx2,
        has_fma,
        has_avx512f,
        has_avx512bw,
        has_avx512vnni,
        has_avxvnni,
        use_avx512,
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_cpuinfo() -> CpuInfo {
    CpuInfo {
        vendor: CpuVendor::Other,
        family: 0,
        model: 0,
        stepping: 0,
        has_avx2: false,
        has_fma: false,
        has_avx512f: false,
        has_avx512bw: false,
        has_avx512vnni: false,
        has_avxvnni: false,
        use_avx512: false,
    }
}

pub fn cpuinfo() -> &'static CpuInfo {
    static INFO: OnceLock<CpuInfo> = OnceLock::new();
    INFO.get_or_init(detect_cpuinfo)
}

/// Tuning profile for this process, resolved once from CPU vendor + env.
pub fn tune() -> OxkTune {
    static TUNE: OnceLock<OxkTune> = OnceLock::new();
    *TUNE.get_or_init(|| {
        let info = cpuinfo();
        let default_blocks = match info.vendor {
            // Measured on 2x Xeon Silver 4110 (Skylake-SP, DDR4-2133) with the
            // contended persistent-worker bench (302 MB fixture, 32T,
            // interleaved pf in {0..8} x {t0,nta}): pf=1/t0 ~72-74 GB/s = the
            // platform pure-read ceiling; pf=2 ~70, pf=4 ~63.5, pf=0 ~62.7,
            // and NTA consistently regressed (~57). One block ahead is enough
            // for the L2 streamer to take over; longer leads evict useful
            // lines under 32-thread contention.
            CpuVendor::Intel => 1_usize,
            // Zen's hardware prefetcher is strong; a small software nudge is
            // enough and bigger distances can collide.
            CpuVendor::Amd => 2_usize,
            CpuVendor::Other => 2_usize,
        };
        let blocks = std::env::var("OXIDIZE_OXK_PF")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_blocks);
        let pf_nta = match std::env::var("OXIDIZE_OXK_PF_HINT").as_deref() {
            Ok("nta") => true,
            Ok("t0") | Err(_) => false,
            Ok(other) => {
                eprintln!("OXIDIZE_OXK_PF_HINT={other} unknown (use t0|nta); using t0");
                false
            }
        };
        OxkTune {
            pf_bytes: blocks * BLOCK_Q4_K_SIZE,
            pf_nta,
        }
    })
}

/// One-line human-readable summary of detected CPU + chosen tuning, for
/// benches and `OXIDIZE_GEMV` debug logging.
pub fn oxk_cpu_summary() -> String {
    let info = cpuinfo();
    let vendor = match info.vendor {
        CpuVendor::Intel => "intel",
        CpuVendor::Amd => "amd",
        CpuVendor::Other => "other",
    };
    let t = tune();
    format!(
        "vendor={vendor} fam={} model={} step={} avx2={} fma={} avx512f={} avx512bw={} avx512vnni={} avxvnni={} use_avx512={} pf_blocks={} pf_hint={}",
        info.family,
        info.model,
        info.stepping,
        info.has_avx2,
        info.has_fma,
        info.has_avx512f,
        info.has_avx512bw,
        info.has_avx512vnni,
        info.has_avxvnni,
        info.use_avx512,
        t.pf_bytes / BLOCK_Q4_K_SIZE,
        if t.pf_nta { "nta" } else { "t0" },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tune_is_block_aligned_and_stable() {
        let t = tune();
        assert_eq!(t.pf_bytes % BLOCK_Q4_K_SIZE, 0);
        let t2 = tune();
        assert_eq!(t.pf_bytes, t2.pf_bytes);
        assert_eq!(t.pf_nta, t2.pf_nta);
    }

    #[test]
    fn summary_mentions_vendor() {
        let s = oxk_cpu_summary();
        assert!(s.contains("vendor="), "{s}");
    }

    #[test]
    fn cpuinfo_is_stable() {
        let a = cpuinfo();
        let b = cpuinfo();
        assert_eq!(a.family, b.family);
        assert_eq!(a.model, b.model);
    }
}
