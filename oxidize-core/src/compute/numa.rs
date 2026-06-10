//! NUMA weight replication for dual-socket decode.
//!
//! On this class of machine ~half of all weight reads hit the remote socket
//! (the page cache spreads the mmap across nodes), paying ~1.5x latency plus
//! Skylake's directory-write tax on every remote line. With the model
//! replicated into one node-bound buffer per socket, every spin-pool worker
//! reads only node-local memory.
//!
//! Enabled with `OXIDIZE_NUMA_REPLICATE=1` at model load; silently skipped on
//! single-node systems, allocation failure, or non-Linux targets. Costs one
//! extra copy of the weights per NUMA node.

#[cfg(target_os = "linux")]
mod imp {
    use std::sync::OnceLock;

    struct Region {
        src_start: usize,
        len: usize,
        /// Node-bound replica base per node id.
        bases: Vec<usize>,
    }

    static REGION: OnceLock<Region> = OnceLock::new();

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

    /// Replicate `src` into one node-bound buffer per NUMA node and register
    /// the region for [`local_slice`] translation. Call once at model load.
    pub fn replicate(src: &[u8]) -> bool {
        let nodes = num_nodes();
        if nodes < 2 || src.is_empty() || REGION.get().is_some() {
            return false;
        }
        let len = src.len();
        let mut bases = Vec::with_capacity(nodes);
        for node in 0..nodes {
            let Some(dst) = alloc_on_node(len, node) else {
                // Roll back: leak nothing useful, unmap what we made.
                for &b in &bases {
                    unsafe { libc::munmap(b as *mut libc::c_void, len) };
                }
                return false;
            };
            // Parallel copy: pages fault on the bound node regardless of the
            // writing CPU (MPOL_BIND), so plain rayon chunks are fine.
            {
                use rayon::prelude::*;
                let chunk = 64 << 20;
                let src_base = src.as_ptr() as usize;
                let dst_base = dst as usize;
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
            bases.push(dst as usize);
        }
        REGION
            .set(Region {
                src_start: src.as_ptr() as usize,
                len,
                bases,
            })
            .is_ok()
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
    /// Slices outside the registered region (or before replication) pass
    /// through unchanged.
    #[inline]
    pub fn local_slice(s: &[u8]) -> &[u8] {
        let Some(region) = REGION.get() else {
            return s;
        };
        let p = s.as_ptr() as usize;
        if p < region.src_start || p + s.len() > region.src_start + region.len {
            return s;
        }
        let node = MY_NODE.with(|n| *n) as usize;
        let Some(&base) = region.bases.get(node) else {
            return s;
        };
        // Safety: the replica buffer mirrors the source region byte-for-byte,
        // is never written after `replicate`, and lives for the process
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

    #[inline]
    pub fn local_slice(s: &[u8]) -> &[u8] {
        s
    }
}

pub use imp::{local_slice, replicate};

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
    fn replicated_region_translates_and_matches() {
        // 8MB synthetic "model"; replication succeeds only on multi-node
        // hosts — on single-node CI this exercises the pass-through path.
        let src: Vec<u8> = (0..8 << 20).map(|i| (i * 31 + 7) as u8).collect();
        let replicated = replicate(&src);
        let slice = &src[1_000_000..1_500_000];
        let local = local_slice(slice);
        assert_eq!(local, slice);
        if replicated {
            assert_ne!(local.as_ptr(), slice.as_ptr(), "should hit a replica");
        }
    }
}
