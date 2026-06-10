//! NUMA weight replication for dual-socket decode.
//!
//! On this class of machine ~half of all weight reads hit the remote socket
//! (the page cache spreads the mmap across nodes), paying ~1.5x latency plus
//! Skylake's directory-write tax on every remote line. With weights
//! replicated into node-bound buffers per socket, every spin-pool worker
//! reads only node-local memory.
//!
//! Two granularities, both registered for [`local_slice`] translation:
//! - [`replicate`]: the whole mapping (one region). Right when the model fits
//!   in every node's memory (e.g. a 35 GB GGUF on 92 GB nodes).
//! - [`replicate_ranges`]: selected byte ranges only (coalesced into regions).
//!   Used for MoE models too large to copy per node, where the dense
//!   (non-expert) tensors are a few GB but carry ~half the per-token reads.
//!
//! Enabled with `OXIDIZE_NUMA_REPLICATE` at model load; silently skipped on
//! single-node systems, allocation failure, or non-Linux targets.

#[cfg(target_os = "linux")]
mod imp {
    use std::sync::OnceLock;

    struct Region {
        src_start: usize,
        len: usize,
        /// Node-bound replica base per node id.
        bases: Vec<usize>,
    }

    /// Sorted by `src_start`; set once at model load.
    static REGIONS: OnceLock<Vec<Region>> = OnceLock::new();

    fn num_nodes() -> usize {
        std::fs::read_to_string("/sys/devices/system/node/online")
            .ok()
            .and_then(|s| {
                let s = s.trim();
                s.rsplit('-').next().and_then(|n| n.parse::<usize>().ok())
            })
            .map(|max| max + 1)
            .unwrap_or(1)
    }

    /// Smallest `MemTotal` across online nodes, in bytes (0 if unreadable).
    pub fn min_node_total_bytes() -> u64 {
        let nodes = num_nodes();
        let mut min = u64::MAX;
        for node in 0..nodes {
            let path = format!("/sys/devices/system/node/node{node}/meminfo");
            let Ok(s) = std::fs::read_to_string(&path) else {
                return 0;
            };
            let Some(kb) = s
                .lines()
                .find(|l| l.contains("MemTotal:"))
                .and_then(|l| l.split_whitespace().rev().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
            else {
                return 0;
            };
            min = min.min(kb * 1024);
        }
        if min == u64::MAX { 0 } else { min }
    }

    fn alloc_on_node(len: usize, node: usize) -> Option<*mut u8> {
        unsafe {
            let p = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            if p == libc::MAP_FAILED {
                return None;
            }
            // 2MB THP for the replicas: 4KB anon pages cost ~4.5M TLB entries
            // for a 17GB model, while the page-cache mapping they replace gets
            // large folios. Sequential fault-in below populates huge pages.
            libc::madvise(p, len, libc::MADV_HUGEPAGE);
            let mask: u64 = 1 << node;
            // MPOL_BIND = 2: fault pages only on `node`.
            let r = libc::syscall(
                libc::SYS_mbind,
                p as usize,
                len,
                2usize,
                &mask as *const u64 as usize,
                64usize,
                0u32,
            );
            if r != 0 {
                libc::munmap(p, len);
                return None;
            }
            Some(p as *mut u8)
        }
    }

    fn copy_parallel(src: *const u8, dst: *mut u8, len: usize) {
        use rayon::prelude::*;
        let chunk = 64 << 20;
        let src_base = src as usize;
        let dst_base = dst as usize;
        // Pages fault on the bound node regardless of the writing CPU
        // (MPOL_BIND), so plain rayon chunks are fine.
        (0..len.div_ceil(chunk)).into_par_iter().for_each(|ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(len);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (src_base as *const u8).add(start),
                    (dst_base as *mut u8).add(start),
                    end - start,
                );
            }
        });
    }

    /// Coalesce sorted `(offset, len)` ranges, merging ranges separated by at
    /// most `gap` bytes (small inter-tensor gaps are cheaper to copy than to
    /// track as separate regions).
    fn coalesce(mut ranges: Vec<(usize, usize)>, gap: usize) -> Vec<(usize, usize)> {
        ranges.retain(|&(_, l)| l > 0);
        ranges.sort_unstable();
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
        for (start, len) in ranges {
            if let Some(last) = out.last_mut() {
                let last_end = last.0 + last.1;
                if start <= last_end.saturating_add(gap) {
                    last.1 = last.1.max(start + len - last.0);
                    continue;
                }
            }
            out.push((start, len));
        }
        out
    }

    /// Replicate the given byte ranges of `src` into node-bound buffers per
    /// NUMA node and register them for [`local_slice`] translation. Ranges are
    /// coalesced (2 MB merge gap). Call once at model load; returns the number
    /// of bytes replicated per node (0 = unavailable / already registered).
    pub fn replicate_ranges(src: &[u8], ranges: &[(usize, usize)]) -> usize {
        let nodes = num_nodes();
        if nodes < 2 || src.is_empty() || ranges.is_empty() || REGIONS.get().is_some() {
            return 0;
        }
        let src_base = src.as_ptr() as usize;
        let merged: Vec<(usize, usize)> = coalesce(ranges.to_vec(), 2 << 20)
            .into_iter()
            .filter(|&(start, len)| start + len <= src.len())
            .collect();
        if merged.is_empty() {
            return 0;
        }

        let mut regions: Vec<Region> = Vec::with_capacity(merged.len());
        let mut total = 0_usize;
        for &(start, len) in &merged {
            let mut bases = Vec::with_capacity(nodes);
            for node in 0..nodes {
                let Some(dst) = alloc_on_node(len, node) else {
                    // Roll back everything: replication is all-or-nothing so
                    // translation never mixes replicated and shared reads
                    // mid-model on failure.
                    for &b in &bases {
                        unsafe { libc::munmap(b as *mut libc::c_void, len) };
                    }
                    for region in &regions {
                        for &b in &region.bases {
                            unsafe { libc::munmap(b as *mut libc::c_void, region.len) };
                        }
                    }
                    return 0;
                };
                copy_parallel((src_base + start) as *const u8, dst, len);
                bases.push(dst as usize);
            }
            total += len;
            regions.push(Region {
                src_start: src_base + start,
                len,
                bases,
            });
        }
        // `merged` is sorted, so `regions` is sorted by src_start.
        if REGIONS.set(regions).is_ok() { total } else { 0 }
    }

    /// Replicate all of `src` (single region). See [`replicate_ranges`].
    pub fn replicate(src: &[u8]) -> bool {
        replicate_ranges(src, &[(0, src.len())]) > 0
    }

    thread_local! {
        /// Cached NUMA node of this thread. Spin-pool workers are pinned, so
        /// one lookup is exact; an unpinned submitter that migrates merely
        /// reads the other node's replica (slower, never incorrect).
        static MY_NODE: u8 = {
            let mut cpu = 0u32;
            let mut node = 0u32;
            unsafe {
                libc::syscall(
                    libc::SYS_getcpu,
                    &mut cpu as *mut u32,
                    &mut node as *mut u32,
                    0usize,
                );
            }
            node as u8
        };
    }

    /// Translate a weight slice into the calling thread's node-local replica.
    /// Slices outside every registered region (or before replication) pass
    /// through unchanged.
    #[inline]
    pub fn local_slice(s: &[u8]) -> &[u8] {
        let Some(regions) = REGIONS.get() else {
            return s;
        };
        let p = s.as_ptr() as usize;
        // Last region with src_start <= p (regions are sorted, disjoint).
        let idx = regions.partition_point(|r| r.src_start <= p);
        let Some(region) = idx.checked_sub(1).and_then(|i| regions.get(i)) else {
            return s;
        };
        if p + s.len() > region.src_start + region.len {
            return s;
        }
        let node = MY_NODE.with(|n| *n) as usize;
        let Some(&base) = region.bases.get(node) else {
            return s;
        };
        // Safety: the replica buffer mirrors the source region byte-for-byte,
        // is never written after replication, and lives for the process
        // lifetime (registered in a static).
        unsafe {
            std::slice::from_raw_parts((base + (p - region.src_start)) as *const u8, s.len())
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    pub fn replicate(_src: &[u8]) -> bool {
        false
    }

    pub fn replicate_ranges(_src: &[u8], _ranges: &[(usize, usize)]) -> usize {
        0
    }

    pub fn min_node_total_bytes() -> u64 {
        0
    }

    #[inline]
    pub fn local_slice(s: &[u8]) -> &[u8] {
        s
    }
}

pub use imp::{local_slice, min_node_total_bytes, replicate, replicate_ranges};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_slice_passes_through_unregistered_memory() {
        let data = vec![3u8; 4096];
        let out = local_slice(&data);
        assert_eq!(out.as_ptr(), data.as_ptr());
        assert_eq!(out, &data[..]);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn replicated_ranges_translate_and_match() {
        // 8MB synthetic "model" with two replicated ranges and a hole.
        // Replication succeeds only on multi-node hosts — on single-node CI
        // this exercises the pass-through path.
        let src: Vec<u8> = (0..8 << 20).map(|i| (i * 31 + 7) as u8).collect();
        let ranges = [(0_usize, 1 << 20), (6 << 20, 1 << 20)];
        let replicated = replicate_ranges(&src, &ranges) > 0;

        let inside = &src[100_000..600_000];
        let local = local_slice(inside);
        assert_eq!(local, inside);
        if replicated {
            assert_ne!(local.as_ptr(), inside.as_ptr(), "should hit a replica");
        }

        // The hole (between the ranges) must always pass through.
        let hole = &src[3 << 20..4 << 20];
        let hole_local = local_slice(hole);
        assert_eq!(hole_local.as_ptr(), hole.as_ptr());

        let second = &src[(6 << 20) + 4096..(6 << 20) + 8192];
        let second_local = local_slice(second);
        assert_eq!(second_local, second);
        if replicated {
            assert_ne!(second_local.as_ptr(), second.as_ptr());
        }
    }
}
