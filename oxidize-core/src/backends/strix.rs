#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrixMode {
    Cpu,
    Vulkan,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrixProfile {
    pub mode: StrixMode,
    pub lazy_loading: bool,
    pub rdna35_tuning: bool,
}

impl Default for StrixProfile {
    fn default() -> Self {
        Self {
            mode: detect_strix_mode(),
            lazy_loading: true,
            rdna35_tuning: true,
        }
    }
}

pub fn detect_strix_mode() -> StrixMode {
    // gfx1151 (Strix Halo, RDNA3.5) is the prime ROCm-iGPU target. Prefer the
    // HIP GEMV path when the rocm feature was compiled with device code present
    // at build time — the kernels are now wavefront-correct (OX_WAVE). Fall back
    // to Vulkan, then CPU.
    if cfg!(feature = "rocm") && crate::rocm::rocm_build_info().detected_at_build {
        StrixMode::Hybrid
    } else if cfg!(feature = "vulkan") && crate::vulkan::vulkan_build_info().detected_at_build {
        StrixMode::Vulkan
    } else {
        StrixMode::Cpu
    }
}

/// Wavefront width the HIP GEMV kernels are compiled for on RDNA3.5. Mirrors
/// `OX_WAVE` in `kernels/gemv_f32.cu` and `rocm::GEMV_LANES_PER_ROW`. Defaults
/// to 64 (wave64), which `build.rs` pins via `-mwavefrontsize64` for both CDNA
/// and RDNA. Only changes if the kernels are rebuilt `-mwavefrontsize32`.
pub const fn rdna35_wavefront_width() -> u32 {
    64
}

pub fn should_lazy_load_layer(layer_index: usize, resident_layers: usize) -> bool {
    layer_index >= resident_layers
}

pub fn rdna35_workgroup_size(hidden_size: usize) -> u32 {
    if hidden_size >= 4096 {
        256
    } else if hidden_size >= 2048 {
        128
    } else {
        64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strix_profile_enables_lazy_loading_and_tuning() {
        let profile = StrixProfile::default();
        assert!(profile.lazy_loading);
        assert!(profile.rdna35_tuning);
        assert_eq!(rdna35_workgroup_size(4096), 256);
        assert!(should_lazy_load_layer(12, 8));
    }

    #[test]
    fn rdna35_workgroups_are_wavefront_multiples() {
        // Workgroup sizes advertised for RDNA3.5 must each be a whole number of
        // wavefronts. (The HIP GEMV path itself launches a fixed 256-thread
        // block = 4 wave64 wavefronts; this invariant guards any other consumer
        // of rdna35_workgroup_size against a partial-wavefront block.)
        let wave = rdna35_wavefront_width();
        assert_eq!(wave, 64);
        for hidden in [1024, 2048, 4096, 8192] {
            assert_eq!(rdna35_workgroup_size(hidden) % wave, 0);
        }
    }
}
