//! CPU vendor / ISA detection and per-vendor kernel tuning.
//!
//! Q4_K decode GEMV is DRAM-bandwidth bound, so the per-vendor levers are in
//! the memory pipeline, not the ALU sequence: software-prefetch distance and
//! cache hint. Intel Skylake-SP (Xeon Silver) and AMD Zen have different L2
//! prefetchers and L3 fill policies, so each vendor gets its own default,
//! selected once per process. Both are overridable for tuning on new parts:
//!
//! * `OXIDIZE_OXK_PF`      — prefetch distance in Q4_K blocks (0 disables).
//! * `OXIDIZE_OXK_PF_HINT` — `t0` (default) or `nta` (non-temporal; keeps
//!   streamed weights from evicting KV cache / activations out of L3).

use std::sync::OnceLock;

use crate::BLOCK_Q4_K_SIZE;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpuVendor {
    Intel,
    Amd,
    Other,
}

/// Memory-pipeline tuning consumed by the AVX2 kernels.
#[derive(Clone, Copy, Debug)]
pub struct OxkTune {
    /// Prefetch distance in bytes ahead of the current weight block pointer
    /// (multiple of `BLOCK_Q4_K_SIZE`; 0 disables software prefetch).
    pub pf_bytes: usize,
    /// Prefetch with `_MM_HINT_NTA` instead of `_MM_HINT_T0`.
    pub pf_nta: bool,
}

pub fn cpu_vendor() -> CpuVendor {
    static VENDOR: OnceLock<CpuVendor> = OnceLock::new();
    *VENDOR.get_or_init(detect_vendor)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_vendor() -> CpuVendor {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::__cpuid;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::__cpuid;
    // cpuid leaf 0 is valid on every x86 CPU that can run this code.
    let r = __cpuid(0);
    let mut v = [0_u8; 12];
    v[0..4].copy_from_slice(&r.ebx.to_le_bytes());
    v[4..8].copy_from_slice(&r.edx.to_le_bytes());
    v[8..12].copy_from_slice(&r.ecx.to_le_bytes());
    match &v {
        b"GenuineIntel" => CpuVendor::Intel,
        b"AuthenticAMD" => CpuVendor::Amd,
        _ => CpuVendor::Other,
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_vendor() -> CpuVendor {
    CpuVendor::Other
}

/// Tuning profile for this process, resolved once from CPU vendor + env.
pub fn tune() -> OxkTune {
    static TUNE: OnceLock<OxkTune> = OnceLock::new();
    *TUNE.get_or_init(|| {
        // 4 blocks (576 B ≈ 9 cache lines) is the Skylake-SP (Xeon Silver)
        // tuning from the OXK plan. A contended 8-thread sweep on Zen 3+
        // (Ryzen 6850H, pf ∈ {0,2,4,8} × {t0,nta}) showed every config within
        // noise — Zen's hardware prefetcher already covers this pattern, and
        // pf=8 was mildly worse — so AMD shares the Intel default rather than
        // diverging on an unmeasurable difference. Re-tune per part with the
        // env overrides + `oxk_q4k_bench` (OXK_BENCH_THREADS=physical cores).
        let default_blocks = match cpu_vendor() {
            CpuVendor::Intel | CpuVendor::Amd | CpuVendor::Other => 4,
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
    let vendor = match cpu_vendor() {
        CpuVendor::Intel => "intel",
        CpuVendor::Amd => "amd",
        CpuVendor::Other => "other",
    };
    let t = tune();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let isa = format!(
        "avx2={} fma={} avxvnni={} avx512vnni={}",
        std::arch::is_x86_feature_detected!("avx2"),
        std::arch::is_x86_feature_detected!("fma"),
        std::arch::is_x86_feature_detected!("avxvnni"),
        std::arch::is_x86_feature_detected!("avx512vnni"),
    );
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let isa = "non-x86".to_string();
    format!(
        "vendor={vendor} {isa} pf_blocks={} pf_hint={}",
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
        // OnceLock: second call returns the identical profile.
        let t2 = tune();
        assert_eq!(t.pf_bytes, t2.pf_bytes);
        assert_eq!(t.pf_nta, t2.pf_nta);
    }

    #[test]
    fn summary_mentions_vendor() {
        let s = oxk_cpu_summary();
        assert!(s.contains("vendor="), "{s}");
    }
}
