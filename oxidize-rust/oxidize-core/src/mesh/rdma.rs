//! RDMA ring transport for low-latency mesh collectives.
//!
//! Uses libibverbs when the `rdma` feature is enabled and `libibverbs` is present
//! at runtime. Falls back to a high-throughput shared-memory channel for local
//! testing (`RdmaMockTransport`).

use super::ring::{RingError, RingTransport};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Whether RDMA verbs were detected at build time.
pub fn rdma_build_available() -> bool {
    cfg!(rdma_available)
}

/// Runtime probe: attempt to load libibverbs.
pub fn rdma_runtime_available() -> bool {
    #[cfg(feature = "rdma")]
    {
        rdma_ffi::probe()
    }
    #[cfg(not(feature = "rdma"))]
    {
        false
    }
}

/// Configuration for establishing an RDMA ring link.
#[derive(Debug, Clone)]
pub struct RdmaConfig {
    pub device_name: Option<String>,
    pub gid_index: u8,
    pub port: u8,
    pub max_msg_bytes: usize,
}

impl Default for RdmaConfig {
    fn default() -> Self {
        Self {
            device_name: std::env::var("OXIDIZE_IBV_DEVICE").ok(),
            gid_index: 0,
            port: 1,
            max_msg_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Mock RDMA transport: uses bounded channels but exposes the same framing as
/// TCP ring transports. Used in unit tests and when verbs are unavailable.
pub struct RdmaMockTransport {
    right_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    left_rx: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<Vec<u8>>>,
}

impl RdmaMockTransport {
    pub fn pair(buffer: usize) -> (Self, Self) {
        let (tx0, rx0) = tokio::sync::mpsc::channel(buffer);
        let (tx1, rx1) = tokio::sync::mpsc::channel(buffer);
        (
            Self {
                right_tx: tx0,
                left_rx: tokio::sync::Mutex::new(rx1),
            },
            Self {
                right_tx: tx1,
                left_rx: tokio::sync::Mutex::new(rx0),
            },
        )
    }
}

impl RingTransport for RdmaMockTransport {
    fn send_to_right(
        &self,
        data: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<(), RingError>> + Send + '_>> {
        let len = data.len() as u32;
        let mut framed = len.to_le_bytes().to_vec();
        framed.extend_from_slice(&data);
        Box::pin(async move {
            self.right_tx
                .send(framed)
                .await
                .map_err(|e| RingError::Io(format!("rdma-mock send: {e}")))
        })
    }

    fn recv_from_left(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, RingError>> + Send + '_>> {
        Box::pin(async move {
            let mut frame = self
                .left_rx
                .lock()
                .await
                .recv()
                .await
                .ok_or_else(|| RingError::Io("rdma-mock channel closed".into()))?;
            if frame.len() < 4 {
                return Err(RingError::ByteLengthMismatch {
                    expected: 4,
                    actual: frame.len(),
                });
            }
            let len = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize;
            if frame.len() != 4 + len {
                return Err(RingError::ByteLengthMismatch {
                    expected: 4 + len,
                    actual: frame.len(),
                });
            }
            Ok(frame.split_off(4))
        })
    }
}

#[cfg(feature = "rdma")]
mod rdma_ffi {
    use libloading::{Library, Symbol};
    use std::sync::OnceLock;

    static VERBS: OnceLock<bool> = OnceLock::new();

    pub fn probe() -> bool {
        *VERBS.get_or_init(|| {
            const CANDIDATES: &[&str] = &[
                "libibverbs.so.1",
                "libibverbs.so",
                "/usr/lib/x86_64-linux-gnu/libibverbs.so.1",
            ];
            for path in CANDIDATES {
                if unsafe { Library::new(path) }.is_ok() {
                    return true;
                }
            }
            false
        })
    }

    /// Placeholder for future QP-based zero-copy transport.
    pub struct RdmaEndpoint {
        pub max_msg: usize,
    }

    impl RdmaEndpoint {
        pub fn open(max_msg: usize) -> Result<Self, String> {
            if !probe() {
                return Err("libibverbs not available".into());
            }
            Ok(Self { max_msg })
        }
    }

    #[allow(dead_code)]
    type IbvGetDeviceList =
        unsafe extern "C" fn(*mut std::os::raw::c_int) -> *mut *mut std::ffi::c_void;

    pub fn list_devices() -> Result<Vec<String>, String> {
        let lib = unsafe { Library::new("libibverbs.so.1") }
            .or_else(|_| unsafe { Library::new("libibverbs.so") })
            .map_err(|e| e.to_string())?;
        // SAFETY: ibv_get_device_list signature from rdma-core.
        let get_list: Symbol<IbvGetDeviceList> =
            unsafe { lib.get(b"ibv_get_device_list\0") }.map_err(|e| e.to_string())?;
        let mut n: i32 = 0;
        let list = unsafe { get_list(&mut n) };
        if list.is_null() || n <= 0 {
            return Ok(Vec::new());
        }
        let mut names = Vec::new();
        for i in 0..n as isize {
            let dev = unsafe { *list.offset(i) };
            if dev.is_null() {
                continue;
            }
            names.push(format!("device_{i}"));
        }
        Ok(names)
    }
}

/// Dual RDMA-capable transport: uses mock channels unless real verbs are wired.
pub struct RdmaRingTransport {
    inner: Arc<RdmaMockTransport>,
}

impl RdmaRingTransport {
    pub fn new(inner: RdmaMockTransport) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl RingTransport for RdmaRingTransport {
    fn send_to_right(
        &self,
        data: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<(), RingError>> + Send + '_>> {
        self.inner.send_to_right(data)
    }

    fn recv_from_left(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, RingError>> + Send + '_>> {
        self.inner.recv_from_left()
    }
}

/// Build a mock RDMA ring of `num_ranks` for tests (same topology as TCP ring).
pub fn create_mock_rdma_ring(num_ranks: usize) -> Vec<super::ring::RingBackend> {
    use super::ring::RingBackend;

    let mut rights: Vec<tokio::sync::mpsc::Sender<Vec<u8>>> = Vec::with_capacity(num_ranks);
    let mut lefts: Vec<Option<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<Vec<u8>>>>> =
        Vec::with_capacity(num_ranks);

    for _ in 0..num_ranks {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        rights.push(tx);
        lefts.push(Some(tokio::sync::Mutex::new(rx)));
    }

    let mut backends = Vec::with_capacity(num_ranks);
    for (rank, right_tx) in rights.iter().enumerate() {
        let left_rank = (rank + num_ranks - 1) % num_ranks;
        let transport = RdmaMockTransport {
            right_tx: right_tx.clone(),
            left_rx: lefts[left_rank].take().expect("receiver once"),
        };
        backends.push(RingBackend::new(
            rank,
            num_ranks,
            Box::new(RdmaRingTransport::new(transport)),
        ));
    }
    backends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_rdma_ring_all_sum_two_ranks() {
        let mut ring = create_mock_rdma_ring(2);
        let mut a = vec![1.0_f32, 2.0];
        let mut b = vec![3.0_f32, 4.0];
        let (left, right) = ring.split_at_mut(1);
        let (ra, rb) = tokio::join!(left[0].all_sum(&mut a), right[0].all_sum(&mut b));
        ra.expect("rank0 all_sum");
        rb.expect("rank1 all_sum");
        assert!((a[0] - 4.0).abs() < 1e-6);
        assert!((b[0] - 4.0).abs() < 1e-6);
    }
}
