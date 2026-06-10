//! Persistent spin-pool for latency-critical GEMV chunk dispatch.
//!
//! Token decode issues hundreds of small parallel regions per token; rayon's
//! sleep/wake worker handoff costs tens of microseconds per region, which
//! dominates wall time once the kernels themselves are fast. This pool keeps
//! workers resident: a region is published by bumping a serial counter,
//! workers claim chunk indices via CAS on a (serial, next) word, and the
//! submitter participates too. Workers spin briefly between regions (covering
//! the intra-token gaps) and park on a condvar when idle longer, so an idle
//! server does not burn CPU.
//!
//! Experimental: enable with `OXIDIZE_SPINPOOL=1` (default: rayon).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

struct Shared {
    /// High 32 bits: region serial. Low 32 bits: next chunk index. Workers
    /// CAS-increment the low half only while the high half matches the region
    /// they validated, so a stale worker can never claim a chunk of a region
    /// whose closure it has not loaded.
    ticket: AtomicU64,
    /// Erased fat pointer to the submitter's `&(dyn Fn(usize) + Sync)`; valid
    /// from the matching `ticket` publish until the submitter observes all
    /// chunks done (it blocks in `run`, keeping the borrow alive).
    task_data: AtomicU64,
    task_vtable: AtomicU64,
    n_chunks: AtomicUsize,
    done: AtomicUsize,
    busy: AtomicBool,
    shutdown: AtomicBool,
    idle_lock: Mutex<()>,
    idle_cv: Condvar,
}

pub struct SpinPool {
    shared: &'static Shared,
}

/// Iterations of `spin_loop` before a worker parks on the condvar. This
/// covers the per-layer glue between GEMVs during decode; truly idle workers
/// park and cost nothing.
const SPIN_BUDGET: u32 = 60_000;

#[inline]
fn serial_of(ticket: u64) -> u64 {
    ticket >> 32
}

impl SpinPool {
    fn new(workers: usize) -> Self {
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            ticket: AtomicU64::new(0),
            task_data: AtomicU64::new(0),
            task_vtable: AtomicU64::new(0),
            n_chunks: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            busy: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            idle_lock: Mutex::new(()),
            idle_cv: Condvar::new(),
        }));
        for worker_idx in 0..workers {
            std::thread::Builder::new()
                .name(format!("oxidize-spin-{worker_idx}"))
                .spawn(move || worker_loop(shared, worker_idx))
                .expect("spawn spin worker");
        }
        Self { shared }
    }

    /// Run `f(chunk_idx)` for every chunk in `0..n_chunks` across the pool.
    /// Blocks until all chunks complete. `f` must tolerate concurrent calls
    /// with distinct indices.
    ///
    /// Only one region may be live at a time; concurrent submissions from a
    /// second thread run inline (decode is single-stream, so this is cold).
    pub fn run(&self, n_chunks: usize, f: &(dyn Fn(usize) + Sync)) {
        if n_chunks == 0 {
            return;
        }
        let s = self.shared;
        if n_chunks == 1
            || s
                .busy
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
        {
            for i in 0..n_chunks {
                f(i);
            }
            return;
        }

        // Publish payload, then the new ticket (release): workers load the
        // payload only after acquiring a ticket with the new serial.
        let fat: [u64; 2] = unsafe { std::mem::transmute(f) };
        s.task_data.store(fat[0], Ordering::Relaxed);
        s.task_vtable.store(fat[1], Ordering::Relaxed);
        s.n_chunks.store(n_chunks, Ordering::Relaxed);
        s.done.store(0, Ordering::Relaxed);
        let serial = serial_of(s.ticket.load(Ordering::Relaxed)) + 1;
        s.ticket.store(serial << 32, Ordering::Release);
        s.idle_cv.notify_all();

        // Submitter claims chunks alongside the workers.
        take_chunks(s, serial, n_chunks, f);

        // Wait for stragglers still running their last chunk.
        while s.done.load(Ordering::Acquire) < n_chunks {
            std::hint::spin_loop();
        }
        s.busy.store(false, Ordering::Release);
    }
}

/// CAS-claim chunk indices for `serial` until the region is exhausted.
#[inline]
fn take_chunks(s: &Shared, serial: u64, n_chunks: usize, f: &dyn Fn(usize)) {
    let mut cur = s.ticket.load(Ordering::Relaxed);
    loop {
        if serial_of(cur) != serial {
            return; // region superseded
        }
        let idx = (cur & 0xffff_ffff) as usize;
        if idx >= n_chunks {
            return; // region exhausted
        }
        match s.ticket.compare_exchange_weak(
            cur,
            cur + 1,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                f(idx);
                s.done.fetch_add(1, Ordering::Release);
                cur = s.ticket.load(Ordering::Relaxed);
            }
            Err(actual) => cur = actual,
        }
    }
}

fn worker_loop(s: &'static Shared, worker_idx: usize) {
    // Pin like the rayon workers (identity map). The spin workers are never
    // active at the same time as a rayon GEMV region, so sharing cores is
    // fine; OXIDIZE_NO_PIN=1 disables.
    #[cfg(target_os = "linux")]
    unsafe {
        let ncpu = libc::sysconf(libc::_SC_NPROCESSORS_ONLN);
        if ncpu > 0 && std::env::var_os("OXIDIZE_NO_PIN").is_none() {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET((worker_idx + 1) % ncpu as usize, &mut set);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = worker_idx;

    let mut last_serial: u64 = 0;
    loop {
        if s.shutdown.load(Ordering::Relaxed) {
            return;
        }
        // Wait for a region with a serial we have not processed.
        let mut spins = 0_u32;
        let serial = loop {
            let t = s.ticket.load(Ordering::Acquire);
            let serial = serial_of(t);
            if serial != last_serial && (t & 0xffff_ffff) < s.n_chunks.load(Ordering::Relaxed) as u64
            {
                break serial;
            }
            if s.shutdown.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            if spins < SPIN_BUDGET {
                std::hint::spin_loop();
            } else {
                let guard = s.idle_lock.lock().unwrap();
                let _guard = s
                    .idle_cv
                    .wait_timeout(guard, std::time::Duration::from_millis(50))
                    .unwrap();
                spins = 0;
            }
        };
        last_serial = serial;
        // Payload was stored before the ticket release for this serial.
        let fat = [
            s.task_data.load(Ordering::Relaxed),
            s.task_vtable.load(Ordering::Relaxed),
        ];
        let f: &(dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(fat) };
        let n = s.n_chunks.load(Ordering::Relaxed);
        take_chunks(s, serial, n, f);
    }
}

static POOL: OnceLock<Option<SpinPool>> = OnceLock::new();

fn pool() -> Option<&'static SpinPool> {
    POOL.get_or_init(|| {
        // Experimental: opt-in only. The shared-ticket design measured SLOWER
        // under 16-thread contention (CAS line ping-pong across sockets) and
        // needs static chunk partitioning before it can be the default.
        if !std::env::var("OXIDIZE_SPINPOOL").is_ok_and(|v| v == "1") {
            return None;
        }
        let workers = rayon::current_num_threads().saturating_sub(1);
        if workers == 0 {
            return None;
        }
        Some(SpinPool::new(workers))
    })
    .as_ref()
}

/// Dispatch `f(chunk_idx)` over `0..n_chunks`: spin pool when enabled,
/// otherwise rayon. Entry point for the decode GEMV hot loops.
pub fn run_chunks(n_chunks: usize, f: impl Fn(usize) + Sync + Send) {
    match pool() {
        Some(p) => p.run(n_chunks, &f),
        None => {
            use rayon::prelude::*;
            (0..n_chunks).into_par_iter().for_each(f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn run_chunks_executes_every_chunk_exactly_once() {
        let counts: Vec<AtomicU32> = (0..1000).map(|_| AtomicU32::new(0)).collect();
        for round in 0..50 {
            run_chunks(counts.len(), |i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            });
            for (i, c) in counts.iter().enumerate() {
                assert_eq!(c.load(Ordering::Relaxed), round + 1, "chunk {i} round {round}");
            }
        }
    }

    #[test]
    fn run_chunks_handles_tiny_and_empty_regions() {
        run_chunks(0, |_| panic!("no chunks"));
        let hit = AtomicU32::new(0);
        run_chunks(1, |i| {
            assert_eq!(i, 0);
            hit.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hit.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn run_chunks_back_to_back_regions_with_different_sizes() {
        for n in [2usize, 7, 31, 128, 1] {
            let counts: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
            run_chunks(n, |i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        }
    }
}
