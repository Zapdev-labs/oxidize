use std::collections::BTreeMap;

const PAGE_BYTES: usize = 4096;
pub const GEMV_KERNEL_NAME: &str = "gemv_f32_kernel";
pub const GEMV_Q8_0_KERNEL_NAME: &str = "gemv_q8_0_f32_kernel";
const GEMV_F32_MSL: &str = include_str!("../kernels/gemv_f32.metal");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalBuildInfo {
    pub detected_at_build: bool,
}

pub fn metal_build_info() -> MetalBuildInfo {
    MetalBuildInfo {
        detected_at_build: cfg!(metal_available),
    }
}

pub fn gemv_msl_source() -> &'static str {
    GEMV_F32_MSL
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnifiedMemoryError {
    OutOfMemory {
        requested: usize,
        available: usize,
    },
    SizeMismatch {
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnifiedMemoryStats {
    pub budget_bytes: usize,
    pub resident_bytes: usize,
    pub active_bytes: usize,
    pub cached_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct UnifiedBuffer {
    len: usize,
    capacity: usize,
    bytes: Vec<u8>,
}

impl UnifiedBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn copy_from_host(&mut self, host: &[u8]) -> Result<(), UnifiedMemoryError> {
        if host.len() != self.len {
            return Err(UnifiedMemoryError::SizeMismatch {
                expected: self.len,
                actual: host.len(),
            });
        }
        self.bytes[..self.len].copy_from_slice(host);
        Ok(())
    }

    pub fn copy_to_host(&self, host: &mut [u8]) -> Result<(), UnifiedMemoryError> {
        if host.len() != self.len {
            return Err(UnifiedMemoryError::SizeMismatch {
                expected: self.len,
                actual: host.len(),
            });
        }
        host.copy_from_slice(&self.bytes[..self.len]);
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct UnifiedBufferManager {
    budget_bytes: usize,
    resident_bytes: usize,
    active_bytes: usize,
    cache: BTreeMap<usize, Vec<Vec<u8>>>,
}

impl UnifiedBufferManager {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            ..Self::default()
        }
    }

    pub fn allocate(&mut self, len: usize) -> Result<UnifiedBuffer, UnifiedMemoryError> {
        let capacity = page_align(len);
        if let Some(cached) = self.cache.get_mut(&capacity).and_then(Vec::pop) {
            self.active_bytes = self.active_bytes.saturating_add(capacity);
            return Ok(UnifiedBuffer {
                len,
                capacity,
                bytes: cached,
            });
        }

        let available = self.budget_bytes.saturating_sub(self.resident_bytes);
        if capacity > available {
            return Err(UnifiedMemoryError::OutOfMemory {
                requested: capacity,
                available,
            });
        }

        self.resident_bytes = self.resident_bytes.saturating_add(capacity);
        self.active_bytes = self.active_bytes.saturating_add(capacity);
        Ok(UnifiedBuffer {
            len,
            capacity,
            bytes: vec![0_u8; capacity],
        })
    }

    pub fn release(&mut self, buffer: UnifiedBuffer) {
        self.active_bytes = self.active_bytes.saturating_sub(buffer.capacity);
        self.cache
            .entry(buffer.capacity)
            .or_default()
            .push(buffer.bytes);
    }

    pub fn trim_cache(&mut self) {
        let cached = self.cached_bytes();
        self.resident_bytes = self.resident_bytes.saturating_sub(cached);
        self.cache.clear();
    }

    pub fn stats(&self) -> UnifiedMemoryStats {
        UnifiedMemoryStats {
            budget_bytes: self.budget_bytes,
            resident_bytes: self.resident_bytes,
            active_bytes: self.active_bytes,
            cached_bytes: self.cached_bytes(),
        }
    }

    fn cached_bytes(&self) -> usize {
        self.cache
            .iter()
            .map(|(capacity, cached)| capacity.saturating_mul(cached.len()))
            .sum()
    }
}

fn page_align(len: usize) -> usize {
    if len == 0 {
        0
    } else {
        len.div_ceil(PAGE_BYTES) * PAGE_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_build_info_reports_cfg_detection() {
        assert_eq!(metal_build_info().detected_at_build, cfg!(metal_available));
    }

    #[test]
    fn gemv_msl_contains_expected_kernel_entries() {
        let source = gemv_msl_source();
        assert!(source.contains("kernel void gemv_f32_kernel"));
        assert!(source.contains("kernel void gemv_q8_0_f32_kernel"));
        assert_eq!(GEMV_KERNEL_NAME, "gemv_f32_kernel");
        assert_eq!(GEMV_Q8_0_KERNEL_NAME, "gemv_q8_0_f32_kernel");
    }

    #[test]
    fn allocates_buffer_with_page_aligned_capacity() {
        let mut manager = UnifiedBufferManager::new(PAGE_BYTES * 4);
        let buffer = manager.allocate(5000).expect("allocation should succeed");

        assert_eq!(buffer.len(), 5000);
        let stats = manager.stats();
        assert_eq!(stats.active_bytes, PAGE_BYTES * 2);
        assert_eq!(stats.resident_bytes, PAGE_BYTES * 2);
        assert_eq!(stats.cached_bytes, 0);
    }

    #[test]
    fn reuses_cached_buffer_without_growing_resident_memory() {
        let mut manager = UnifiedBufferManager::new(PAGE_BYTES * 4);
        let buffer = manager.allocate(128).expect("allocation should succeed");
        manager.release(buffer);
        let before = manager.stats();

        let _reused = manager.allocate(64).expect("reuse should succeed");
        let after = manager.stats();
        assert_eq!(before.resident_bytes, after.resident_bytes);
        assert_eq!(after.cached_bytes, 0);
        assert_eq!(after.active_bytes, PAGE_BYTES);
    }

    #[test]
    fn rejects_allocation_when_budget_is_exceeded() {
        let mut manager = UnifiedBufferManager::new(PAGE_BYTES);
        let _first = manager
            .allocate(PAGE_BYTES)
            .expect("first allocation should consume budget");

        let err = manager
            .allocate(1)
            .expect_err("second allocation should exceed budget");
        assert_eq!(
            err,
            UnifiedMemoryError::OutOfMemory {
                requested: PAGE_BYTES,
                available: 0
            }
        );
    }

    #[test]
    fn supports_host_transfers_for_unified_buffer() {
        let mut manager = UnifiedBufferManager::new(PAGE_BYTES);
        let mut buffer = manager.allocate(4).expect("allocation should succeed");
        buffer
            .copy_from_host(&[1_u8, 2, 3, 4])
            .expect("write should succeed");

        let mut host = [0_u8; 4];
        buffer.copy_to_host(&mut host).expect("read should succeed");
        assert_eq!(host, [1, 2, 3, 4]);
    }

    #[test]
    fn trim_cache_releases_resident_budget() {
        let mut manager = UnifiedBufferManager::new(PAGE_BYTES * 2);
        let buffer = manager.allocate(10).expect("allocation should succeed");
        manager.release(buffer);
        assert_eq!(manager.stats().resident_bytes, PAGE_BYTES);

        manager.trim_cache();
        let stats = manager.stats();
        assert_eq!(stats.cached_bytes, 0);
        assert_eq!(stats.resident_bytes, 0);
    }
}
