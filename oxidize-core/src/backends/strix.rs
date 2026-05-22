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
    if cfg!(feature = "vulkan") && crate::vulkan::vulkan_build_info().detected_at_build {
        StrixMode::Vulkan
    } else {
        StrixMode::Cpu
    }
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
}
