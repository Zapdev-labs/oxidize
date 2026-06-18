//! Safe wrappers for memory mapping and byte-level reads.
//!
//! Platform-specific `unsafe` is centralized here so call sites stay in safe Rust.

use std::fs::File;
use std::io;
use memmap2::Mmap;

/// Memory-map a file for read-only access.
pub fn map_readonly(file: &File) -> io::Result<Mmap> {
    // SAFETY: Mapping is read-only; callers keep the file unmodified while mapped
    // and own the `Mmap` for its lifetime.
    unsafe { Mmap::map(file) }
}

#[inline]
pub fn read_le_i16(bytes: &[u8], byte_offset: usize) -> i16 {
    i16::from_le_bytes([bytes[byte_offset], bytes[byte_offset + 1]])
}

/// Q8_K blockscale sum entry (little-endian i16 pairs in the bsums tail).
#[inline]
pub fn read_q8_k_bsum(bsums: &[u8], index: usize) -> i16 {
    read_le_i16(bsums, index * 2)
}

/// Volatile byte read for page prefaulting / NUMA warm-up.
#[inline]
pub fn read_volatile_byte(bytes: &[u8], offset: usize) -> u8 {
    // SAFETY: `offset` is always page-aligned and in-bounds at call sites.
    unsafe { std::ptr::read_volatile(bytes.as_ptr().add(offset)) }
}
