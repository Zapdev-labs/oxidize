#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512f,
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    Neon,
}

impl SimdBackend {
    pub fn lane_width_f32(self) -> usize {
        match self {
            Self::Scalar => 1,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => 4,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx => 8,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => 8,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx512f => 16,
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            Self::Neon => 4,
        }
    }
}

pub fn available_backends() -> Vec<SimdBackend> {
    let mut backends = vec![SimdBackend::Scalar];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cfg!(target_feature = "sse2") {
            backends.push(SimdBackend::Sse2);
        }
        if cfg!(target_feature = "avx") {
            backends.push(SimdBackend::Avx);
        }
        if cfg!(target_feature = "avx2") {
            backends.push(SimdBackend::Avx2);
        }
        if cfg!(target_feature = "avx512f") {
            backends.push(SimdBackend::Avx512f);
        }
    }

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        if cfg!(target_feature = "neon") {
            backends.push(SimdBackend::Neon);
        }
    }

    backends
}

pub fn preferred_backend() -> SimdBackend {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cfg!(target_feature = "avx512f") {
            return SimdBackend::Avx512f;
        }
        if cfg!(target_feature = "avx2") {
            return SimdBackend::Avx2;
        }
        if cfg!(target_feature = "avx") {
            return SimdBackend::Avx;
        }
        if cfg!(target_feature = "sse2") {
            return SimdBackend::Sse2;
        }
    }

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        if cfg!(target_feature = "neon") {
            return SimdBackend::Neon;
        }
    }

    SimdBackend::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_backends_always_include_scalar() {
        assert!(available_backends().contains(&SimdBackend::Scalar));
    }

    #[test]
    fn preferred_backend_is_available() {
        let available = available_backends();
        assert!(available.contains(&preferred_backend()));
    }

    #[test]
    fn lane_widths_are_non_zero() {
        for backend in available_backends() {
            assert!(backend.lane_width_f32() > 0);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_backend_order_matches_capability_priority() {
        let preferred = preferred_backend();
        let expected = if cfg!(target_feature = "avx512f") {
            SimdBackend::Avx512f
        } else if cfg!(target_feature = "avx2") {
            SimdBackend::Avx2
        } else if cfg!(target_feature = "avx") {
            SimdBackend::Avx
        } else if cfg!(target_feature = "sse2") {
            SimdBackend::Sse2
        } else {
            SimdBackend::Scalar
        };
        assert_eq!(preferred, expected);
    }

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    #[test]
    fn arm_prefers_neon_when_enabled() {
        let expected = if cfg!(target_feature = "neon") {
            SimdBackend::Neon
        } else {
            SimdBackend::Scalar
        };
        assert_eq!(preferred_backend(), expected);
    }
}
