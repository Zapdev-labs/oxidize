//! Persistent spin-pool for latency-critical GEMV chunk dispatch.
//!
//! Token decode issues hundreds of small parallel regions per token; rayon's
//! sleep/wake worker handoff costs tens of microseconds per region, which
//! dominates wall time once the kernels themselves are fast. This pool keeps
//! workers resident and uses STATIC block partitioning: participant `p` of
//! `P` owns the contiguous chunk range `[p*n/P, (p+1)*n/P)`, so there is no
//! shared claim counter to contend on (a shared-CAS ticket measurably
//! collapsed under cross-socket contention) and each worker streams
//! sequential weight rows. Chunks are uniform, so blocks balance within one
//! chunk of ideal.
//!
//! Region lifecycle: the submitter stores the closure fat pointer + chunk
//! count, bumps `serial` (release), and processes its own share. Each worker
//! acks completion by writing the serial into its own cache-line-padded slot;
//! the submitter waits for every ack before returning, which both keeps the
//! closure borrow alive for stragglers and prevents the next region's payload
//! from overwriting one still being read.
//!
//! Workers spin briefly between regions (covering per-layer glue during
//! decode) and park on a condvar when idle, so an idle server costs nothing.
//!
//! Disable with `OXIDIZE_SPINPOOL=0` (falls back to rayon).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

#[repr(align(64))]
struct AckSlot {
    done_serial: AtomicU64,
}

struct Shared {
    /// Region serial; bumped (release) after the payload below is stored.
    serial: AtomicU64,
    /// Erased fat pointer to the submitter's `&(dyn Fn(usize) + Sync)`.
    /// Valid from the serial bump until every worker acks that serial.
    task_data: AtomicU64,
    task_vtable: AtomicU64,
    n_chunks: AtomicUsize,
    /// One ack slot per worker, cache-line padded: written only by its owner.
    acks: Box<[AckSlot]>,
    busy: AtomicBool,
    shutdown: AtomicBool,
    idle_lock: Mutex<()>,
    idle_cv: Condvar,
}

pub struct SpinPool {
    shared: &'static Shared,
    /// Workers + the submitting thread.
    participants: usize,
}

/// `spin_loop` iterations before a worker parks. On Skylake a pause is
/// ~100+ cycles, so this covers multi-millisecond gaps — far more than the
/// per-layer glue between decode GEMVs; truly idle workers park.
const SPIN_BUDGET: u32 = 60_000;

impl SpinPool {
    fn new(workers: usize) -> Self {
        let acks: Box<[AckSlot]> = (0..workers)
            .map(|_| AckSlot {
                done_serial: AtomicU64::new(0),
            })
            .collect();
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            serial: AtomicU64::new(0),
            task_data: AtomicU64::new(0),
            task_vtable: AtomicU64::new(0),
            n_chunks: AtomicUsize::new(0),
            acks,
            busy: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            idle_lock: Mutex::new(()),
            idle_cv: Condvar::new(),
        }));
        for worker_idx in 0..workers {
            std::thread::Builder::new()
                .name(format!("oxidize-spin-{worker_idx}"))
                .spawn(move || worker_loop(shared, worker_idx, workers + 1))
                .expect("spawn spin worker");
        }
        Self {
            shared,
            participants: workers + 1,
        }
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

        // Publish payload, then the new serial (release): workers read the
        // payload only after observing the bumped serial.
        let fat: [u64; 2] = unsafe { std::mem::transmute(f) };
        s.task_data.store(fat[0], Ordering::Relaxed);
        s.task_vtable.store(fat[1], Ordering::Relaxed);
        s.n_chunks.store(n_chunks, Ordering::Relaxed);
        let serial = s.serial.load(Ordering::Relaxed) + 1;
        s.serial.store(serial, Ordering::Release);
        // Pair with the worker's serial re-check under the lock so a worker
        // deciding to park right now cannot miss the wakeup.
        drop(s.idle_lock.lock().unwrap());
        s.idle_cv.notify_all();

        // Submitter is participant 0. Participants own CONTIGUOUS chunk
        // ranges so each worker streams sequential weight rows (strided
        // ownership defeats the hardware prefetcher).
        let participants = self.participants;
        for i in 0..n_chunks / participants {
            f(i);
        }
        // Tail chunks (n % P) belong to the last participants by the block
        // formula; participant 0's range is exactly [0, n/P).

        // Wait until every worker acks this serial; the payload and `f`'s
        // borrow must outlive any straggler still reading them.
        for slot in s.acks.iter() {
            while slot.done_serial.load(Ordering::Acquire) != serial {
                std::hint::spin_loop();
            }
        }
        s.busy.store(false, Ordering::Release);
    }
}

fn worker_loop(s: &'static Shared, worker_idx: usize, participants: usize) {
    // Pin like the rayon workers (identity map, submitter-adjacent CPUs).
    // The spin workers are never active at the same time as a rayon GEMV
    // region, so sharing cores is fine; OXIDIZE_NO_PIN=1 disables.
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

    let my_participant = worker_idx + 1;
    let mut last_serial: u64 = 0;
    loop {
        if s.shutdown.load(Ordering::Relaxed) {
            return;
        }
        // Wait for a region we have not processed.
        let mut spins = 0_u32;
        let serial = loop {
            let serial = s.serial.load(Ordering::Acquire);
            if serial != last_serial {
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
                // Re-check under the lock: the submitter bumps the serial
                // before taking this lock to notify, so we cannot sleep
                // through a publish.
                if s.serial.load(Ordering::Acquire) == last_serial {
                    let _guard = s
                        .idle_cv
                        .wait_timeout(guard, std::time::Duration::from_millis(50))
                        .unwrap();
                }
                spins = 0;
            }
        };
        last_serial = serial;
        // Payload was stored before the serial release for this region, and
        // stays valid until we ack below.
        let fat = [
            s.task_data.load(Ordering::Relaxed),
            s.task_vtable.load(Ordering::Relaxed),
        ];
        let f: &(dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(fat) };
        let n = s.n_chunks.load(Ordering::Relaxed);
        let start = (my_participant * n) / participants;
        let end = ((my_participant + 1) * n) / participants;
        for i in start..end {
            f(i);
        }
        s.acks[worker_idx]
            .done_serial
            .store(serial, Ordering::Release);
    }
}

static POOL: OnceLock<Option<SpinPool>> = OnceLock::new();

fn pool() -> Option<&'static SpinPool> {
    POOL.get_or_init(|| {
        if std::env::var("OXIDIZE_SPINPOOL").is_ok_and(|v| v == "0") {
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
        for round in 0..200 {
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
    fn run_chunks_back_to_back_regions_with_varied_sizes() {
        for n in [2usize, 7, 31, 128, 1, 333, 4096] {
            let counts: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
            run_chunks(n, |i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    fn run_chunks_results_match_serial_reference() {
        // GEMV-shaped check: each chunk writes a deterministic function of its
        // index into a disjoint slice; compare against serial execution.
        let n_chunks = 517usize;
        let chunk = 32usize;
        let mut parallel = vec![0.0f32; n_chunks * chunk];
        let base = parallel.as_mut_ptr() as usize;
        run_chunks(n_chunks, |ci| {
            let out = unsafe {
                std::slice::from_raw_parts_mut((base as *mut f32).add(ci * chunk), chunk)
            };
            for (j, v) in out.iter_mut().enumerate() {
                *v = (ci * 31 + j * 7) as f32 * 0.5;
            }
        });
        for ci in 0..n_chunks {
            for j in 0..chunk {
                assert_eq!(parallel[ci * chunk + j], (ci * 31 + j * 7) as f32 * 0.5);
            }
        }
    }
}
