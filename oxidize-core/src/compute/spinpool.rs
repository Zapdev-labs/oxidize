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
//! Enabled by default (all decode hot loops dispatch through [`run_chunks`]);
//! disable with `OXIDIZE_SPINPOOL=0` (falls back to rayon).

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

struct Topology {
    /// All online logical CPUs, core-first: the first `cores` entries are the
    /// first SMT sibling of each physical core, the rest are the remaining
    /// siblings. Pinning worker `i` to `order[i]` spreads the first `cores`
    /// workers across whole cores; an identity map does not (Linux enumerates
    /// sibling pairs adjacently on AMD, so identity stacks pairs of workers
    /// onto half the cores).
    order: Vec<usize>,
    cores: usize,
}

#[cfg(target_os = "linux")]
fn parse_cpu_list(s: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in s.trim().split(',') {
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                cpus.extend(a..=b);
            }
        } else if let Ok(v) = part.parse::<usize>() {
            cpus.push(v);
        }
    }
    cpus
}

#[cfg(target_os = "linux")]
fn read_topology() -> Option<Topology> {
    let online = std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    let cpus = parse_cpu_list(&online);
    let mut order = Vec::with_capacity(cpus.len());
    let mut rest = Vec::new();
    for &cpu in &cpus {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let siblings = std::fs::read_to_string(&path).ok()?;
        let first = parse_cpu_list(&siblings).into_iter().min()?;
        if first == cpu {
            order.push(cpu);
        } else {
            rest.push(cpu);
        }
    }
    if order.is_empty() {
        return None;
    }
    let cores = order.len();
    order.extend(rest);
    Some(Topology { order, cores })
}

fn topology() -> &'static Topology {
    static TOPOLOGY: OnceLock<Topology> = OnceLock::new();
    TOPOLOGY.get_or_init(|| {
        #[cfg(target_os = "linux")]
        if let Some(t) = read_topology() {
            return t;
        }
        let n = std::thread::available_parallelism().map_or(1, usize::from);
        Topology {
            order: (0..n).collect(),
            cores: n,
        }
    })
}

/// Number of physical cores (logical CPUs when the SMT topology is
/// unreadable). Decode GEMV is DRAM-bound and saturates with one worker per
/// core — SMT siblings only split issue slots — so thread-count defaults use
/// this rather than `available_parallelism`.
pub fn physical_core_count() -> usize {
    topology().cores
}

/// Pin the calling thread to the `slot`-th CPU in core-first order (one
/// physical core per slot until cores run out, then the remaining SMT
/// siblings). Stable placement keeps each worker's weight stream on one
/// core's prefetcher and, on NUMA hosts, on one node. No-op with
/// `OXIDIZE_NO_PIN=1` or off Linux.
#[cfg(target_os = "linux")]
pub fn pin_to_slot(slot: usize) {
    if std::env::var_os("OXIDIZE_NO_PIN").is_some() {
        return;
    }
    let order = &topology().order;
    let cpu = order[slot % order.len()];
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_to_slot(_slot: usize) {}

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
        // Pin the submitting thread to slot 0 (workers own slots 1..P). An
        // unpinned submitter floats onto cores where workers are spinning and
        // timeshares against them — all the serial glue between regions (and
        // the submitter's own chunk range) then runs at half speed.
        thread_local! {
            static PINNED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        PINNED.with(|pinned| {
            if !pinned.get() {
                pin_to_slot(0);
                pinned.set(true);
            }
        });
        let s = self.shared;
        if n_chunks == 1
            || s.busy
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
    // Pin like the rayon workers (core-first order, submitter-adjacent
    // slots). The spin workers are never active at the same time as a rayon
    // GEMV region, so sharing cores is fine; OXIDIZE_NO_PIN=1 disables.
    pin_to_slot(worker_idx + 1);

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
                    let (_guard, timeout) = s
                        .idle_cv
                        .wait_timeout(guard, std::time::Duration::from_millis(50))
                        .unwrap();
                    // Only a notify means a region is imminent; a timeout on
                    // an idle pool must NOT re-enter the spin phase, or every
                    // idle worker burns a few ms of CPU per 50ms — poisonous
                    // when other processes share these cores.
                    if timeout.timed_out() {
                        spins = SPIN_BUDGET;
                        continue;
                    }
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
        // Default on: with every decode hot loop dispatched through
        // run_chunks (GEMV fused regions + attention heads), the resident
        // workers beat rayon's sleep/wake handoff on single-socket parts too
        // (11.8 vs 10.9 tok/s, Ryzen 6850H) — but only with the submitter
        // pinned to slot 0 and no nested/concurrent regions, which would run
        // inline-serial. OXIDIZE_SPINPOOL=0 falls back to rayon.
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
            // Static block partitioning, like the spin pool: one contiguous
            // chunk range per worker. Decode GEMV chunks are ~1-10us each;
            // letting rayon schedule hundreds of them individually buries
            // the kernels in steal/join overhead (a 9728x2560 Q4_K GEMV
            // measured 21 GB/s with per-chunk tasks vs ~36 GB/s for shapes
            // with coarser chunks). Chunks are uniform, so blocks balance
            // within one chunk of ideal.
            let tasks = rayon::current_num_threads().min(n_chunks);
            if tasks <= 1 {
                for i in 0..n_chunks {
                    f(i);
                }
                return;
            }
            (0..tasks).into_par_iter().for_each(|t| {
                let start = t * n_chunks / tasks;
                let end = (t + 1) * n_chunks / tasks;
                for i in start..end {
                    f(i);
                }
            });
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
                assert_eq!(
                    c.load(Ordering::Relaxed),
                    round + 1,
                    "chunk {i} round {round}"
                );
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

    #[test]
    fn topology_pin_order_covers_each_cpu_once() {
        let t = topology();
        assert!(t.cores >= 1);
        assert!(t.cores <= t.order.len());
        let mut seen = t.order.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), t.order.len(), "pin order must not repeat CPUs");
        let logical = std::thread::available_parallelism().map_or(1, usize::from);
        assert_eq!(t.order.len(), logical);
    }
}
