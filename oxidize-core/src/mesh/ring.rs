//! TCP ring backend for distributed collectives.
//!
//! Implements ring all-reduce (all_sum) and ring all-gather over an
//! abstract ring transport.  A mock channel transport is provided for
//! fast unit tests; a TCP transport is provided for real mesh usage.

use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Errors raised by ring operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RingError {
    Io(String),
    Timeout,
    MismatchedRankCount { expected: usize, actual: usize },
    WrongChunkSize { expected: usize, actual: usize },
    NotConnected,
}

impl std::fmt::Display for RingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RingError::Io(s) => write!(f, "ring io error: {s}"),
            RingError::Timeout => write!(f, "ring operation timed out"),
            RingError::MismatchedRankCount { expected, actual } => {
                write!(f, "expected {expected} ranks, got {actual}")
            }
            RingError::WrongChunkSize { expected, actual } => {
                write!(f, "expected chunk size multiple of {expected}, got remainder {actual}")
            }
            RingError::NotConnected => write!(f, "ring transport not connected"),
        }
    }
}

impl std::error::Error for RingError {}

/// Abstract ring transport.  Each rank sends to its right neighbour and
/// receives from its left neighbour.
///
/// Methods take `&self` so that send and receive futures can be created
/// concurrently without violating Rust's aliasing rules.  Implementations
/// use interior mutability (e.g. [`tokio::sync::Mutex`]) where needed.
pub trait RingTransport: Send {
    fn send_to_right(
        &self,
        data: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<(), RingError>> + Send + '_>>;

    fn recv_from_left(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, RingError>> + Send + '_>>;
}

/// Mock channel transport for unit tests.
pub struct ChannelTransport {
    pub right_tx: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
    pub left_rx: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>>,
}

impl RingTransport for ChannelTransport {
    fn send_to_right(
        &self,
        data: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<(), RingError>> + Send + '_>> {
        Box::pin(async move {
            self.right_tx
                .send(data)
                .map_err(|e| RingError::Io(format!("channel send: {e}")))
        })
    }

    fn recv_from_left(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, RingError>> + Send + '_>> {
        Box::pin(async move {
            self.left_rx
                .lock()
                .await
                .recv()
                .await
                .ok_or_else(|| RingError::Io("channel closed".to_string()))
        })
    }
}

/// TCP transport with length-prefixed framing using a single bidirectional
/// stream.  Works because TCP is full-duplex.
pub struct TcpTransport {
    stream: tokio::sync::Mutex<TcpStream>,
}

impl TcpTransport {
    pub fn new(stream: TcpStream) -> Self {
        Self {
            stream: tokio::sync::Mutex::new(stream),
        }
    }
}

impl RingTransport for TcpTransport {
    fn send_to_right(
        &self,
        data: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<(), RingError>> + Send + '_>> {
        Box::pin(async move {
            let len = data.len() as u32;
            let mut s = self.stream.lock().await;
            s.write_all(&len.to_le_bytes())
                .await
                .map_err(|e| RingError::Io(e.to_string()))?;
            s.write_all(&data)
                .await
                .map_err(|e| RingError::Io(e.to_string()))?;
            Ok(())
        })
    }

    fn recv_from_left(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, RingError>> + Send + '_>> {
        Box::pin(async move {
            let mut len_bytes = [0u8; 4];
            let mut s = self.stream.lock().await;
            s.read_exact(&mut len_bytes)
                .await
                .map_err(|e| RingError::Io(e.to_string()))?;
            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut buf = vec![0u8; len];
            s.read_exact(&mut buf)
                .await
                .map_err(|e| RingError::Io(e.to_string()))?;
            Ok(buf)
        })
    }
}

/// Dual-socket TCP transport: send on one stream, receive on another.
/// Needed when the ring is wired with separate outbound / inbound sockets.
pub struct DualTcpTransport {
    send_stream: tokio::sync::Mutex<TcpStream>,
    recv_stream: tokio::sync::Mutex<TcpStream>,
}

impl DualTcpTransport {
    pub fn new(send_stream: TcpStream, recv_stream: TcpStream) -> Self {
        Self {
            send_stream: tokio::sync::Mutex::new(send_stream),
            recv_stream: tokio::sync::Mutex::new(recv_stream),
        }
    }
}

impl RingTransport for DualTcpTransport {
    fn send_to_right(
        &self,
        data: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<(), RingError>> + Send + '_>> {
        Box::pin(async move {
            let len = data.len() as u32;
            let mut s = self.send_stream.lock().await;
            s.write_all(&len.to_le_bytes())
                .await
                .map_err(|e| RingError::Io(e.to_string()))?;
            s.write_all(&data)
                .await
                .map_err(|e| RingError::Io(e.to_string()))?;
            Ok(())
        })
    }

    fn recv_from_left(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, RingError>> + Send + '_>> {
        Box::pin(async move {
            let mut len_bytes = [0u8; 4];
            let mut s = self.recv_stream.lock().await;
            s.read_exact(&mut len_bytes)
                .await
                .map_err(|e| RingError::Io(e.to_string()))?;
            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut buf = vec![0u8; len];
            s.read_exact(&mut buf)
                .await
                .map_err(|e| RingError::Io(e.to_string()))?;
            Ok(buf)
        })
    }
}

/// Ring collective backend.
pub struct RingBackend {
    pub rank: usize,
    pub num_ranks: usize,
    pub transport: Box<dyn RingTransport>,
}

impl RingBackend {
    pub fn new(rank: usize, num_ranks: usize, transport: Box<dyn RingTransport>) -> Self {
        Self {
            rank,
            num_ranks,
            transport,
        }
    }

    /// Ring all-reduce: sum `data` in-place so every rank ends up with the
    /// same element-wise sum.
    ///
    /// `data.len()` must be evenly divisible by `num_ranks`.
    pub async fn all_sum(&mut self, data: &mut [f32]) -> Result<(), RingError> {
        if self.num_ranks == 1 {
            return Ok(());
        }
        if !data.len().is_multiple_of(self.num_ranks) {
            return Err(RingError::WrongChunkSize {
                expected: self.num_ranks,
                actual: data.len() % self.num_ranks,
            });
        }

        let chunk_size = data.len() / self.num_ranks;
        let mut send_buf = vec![0.0_f32; chunk_size];
        let mut recv_f32 = vec![0.0_f32; chunk_size];

        // ---------- Scatter-reduce: N-1 steps ----------
        // At step s, rank i sends chunk (i - s) mod N and receives into
        // chunk (i - s - 1) mod N from the left.
        let mut send_chunk = self.rank;
        let mut recv_chunk = (self.rank + self.num_ranks - 1) % self.num_ranks;

        for _step in 0..(self.num_ranks - 1) {
            let send_off = send_chunk * chunk_size;
            send_buf.copy_from_slice(&data[send_off..send_off + chunk_size]);
            let send_bytes = f32_slice_to_bytes(&send_buf);

            let send_fut = self.transport.send_to_right(send_bytes);
            let recv_fut = self.transport.recv_from_left();
            let (send_res, recv_res) = tokio::join!(send_fut, recv_fut);
            send_res?;
            let recv_bytes = recv_res?;
            bytes_to_f32_slice_into(&recv_bytes, &mut recv_f32)?;

            let recv_off = recv_chunk * chunk_size;
            for i in 0..chunk_size {
                data[recv_off + i] += recv_f32[i];
            }

            send_chunk = (send_chunk + self.num_ranks - 1) % self.num_ranks;
            recv_chunk = (recv_chunk + self.num_ranks - 1) % self.num_ranks;
        }

        // ---------- All-gather: N-1 steps ----------
        // At step s, rank i sends the chunk it acquired at step s-1
        // (starting with its fully-reduced chunk at index (i+1) mod N)
        // and receives into the next missing slot.
        let mut send_chunk = (self.rank + 1) % self.num_ranks;
        let mut recv_chunk = self.rank;

        for _step in 0..(self.num_ranks - 1) {
            let send_off = send_chunk * chunk_size;
            send_buf.copy_from_slice(&data[send_off..send_off + chunk_size]);
            let send_bytes = f32_slice_to_bytes(&send_buf);

            let send_fut = self.transport.send_to_right(send_bytes);
            let recv_fut = self.transport.recv_from_left();
            let (send_res, recv_res) = tokio::join!(send_fut, recv_fut);
            send_res?;
            let recv_bytes = recv_res?;
            bytes_to_f32_slice_into(&recv_bytes, &mut recv_f32)?;

            let recv_off = recv_chunk * chunk_size;
            data[recv_off..recv_off + chunk_size].copy_from_slice(&recv_f32);

            send_chunk = (send_chunk + self.num_ranks - 1) % self.num_ranks;
            recv_chunk = (recv_chunk + self.num_ranks - 1) % self.num_ranks;
        }

        Ok(())
    }

    /// Ring all-gather: each rank contributes one chunk and receives the
    /// concatenation of all chunks into `out`.
    ///
    /// `data.len()` must equal `chunk_size`; `out.len()` must equal
    /// `chunk_size * num_ranks`.
    pub async fn all_gather(&mut self, data: &[f32], out: &mut [f32]) -> Result<(), RingError> {
        if self.num_ranks == 1 {
            out.copy_from_slice(data);
            return Ok(());
        }
        let chunk_size = data.len();
        if out.len() != chunk_size * self.num_ranks {
            return Err(RingError::WrongChunkSize {
                expected: chunk_size * self.num_ranks,
                actual: out.len(),
            });
        }

        let mut send_buf = vec![0.0_f32; chunk_size];
        let mut recv_f32 = vec![0.0_f32; chunk_size];

        // Place local chunk at the correct offset.
        let local_offset = self.rank * chunk_size;
        out[local_offset..local_offset + chunk_size].copy_from_slice(data);

        // Each rank initially holds its own chunk.
        let mut send_chunk = self.rank;
        let mut recv_chunk = (self.rank + self.num_ranks - 1) % self.num_ranks;

        for _step in 0..(self.num_ranks - 1) {
            let send_off = send_chunk * chunk_size;
            send_buf.copy_from_slice(&out[send_off..send_off + chunk_size]);
            let send_bytes = f32_slice_to_bytes(&send_buf);

            let send_fut = self.transport.send_to_right(send_bytes);
            let recv_fut = self.transport.recv_from_left();
            let (send_res, recv_res) = tokio::join!(send_fut, recv_fut);
            send_res?;
            let recv_bytes = recv_res?;
            bytes_to_f32_slice_into(&recv_bytes, &mut recv_f32)?;

            let recv_off = recv_chunk * chunk_size;
            out[recv_off..recv_off + chunk_size].copy_from_slice(&recv_f32);

            send_chunk = (send_chunk + self.num_ranks - 1) % self.num_ranks;
            recv_chunk = (recv_chunk + self.num_ranks - 1) % self.num_ranks;
        }

        Ok(())
    }
}

/// Build a mock ring of `num_ranks` backends connected via channels.
pub fn create_mock_ring(num_ranks: usize) -> Vec<RingBackend> {
    let mut right_txs: Vec<tokio::sync::mpsc::UnboundedSender<Vec<u8>>> =
        Vec::with_capacity(num_ranks);
    let mut left_rxs: Vec<Option<tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>>>> =
        Vec::with_capacity(num_ranks);
    for _ in 0..num_ranks {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        right_txs.push(tx);
        left_rxs.push(Some(tokio::sync::Mutex::new(rx)));
    }

    let mut backends = Vec::with_capacity(num_ranks);
    for (rank, right_tx) in right_txs.iter().enumerate() {
        let left_rank = (rank + num_ranks - 1) % num_ranks;
        let transport = ChannelTransport {
            right_tx: right_tx.clone(),
            left_rx: left_rxs[left_rank].take().expect("receiver consumed once"),
        };
        backends.push(RingBackend::new(rank, num_ranks, Box::new(transport)));
    }
    backends
}

/// Helper: bind a TCP listener on an ephemeral localhost port and return
/// the socket address.
pub async fn tcp_bind_ephemeral() -> Result<(TcpListener, std::net::SocketAddr), RingError> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| RingError::Io(e.to_string()))?;
    let addr = listener.local_addr().map_err(|e| RingError::Io(e.to_string()))?;
    Ok((listener, addr))
}

/// Spawn a TCP ring of `num_ranks` backends on localhost ephemeral ports.
///
/// Every rank opens an outbound connection to its right neighbour.  The
/// *inbound* connection from the left neighbour is accepted on a listener
/// bound at the rank's own address.  Each rank therefore ends up with two
/// sockets: `send_stream` (to right) and `recv_stream` (from left).  These
/// are wrapped in a [`DualTcpTransport`].
pub async fn create_tcp_ring(num_ranks: usize) -> Result<Vec<RingBackend>, RingError> {
    let mut listeners = Vec::with_capacity(num_ranks);
    let mut addrs = Vec::with_capacity(num_ranks);
    for _ in 0..num_ranks {
        let (listener, addr) = tcp_bind_ephemeral().await?;
        listeners.push(listener);
        addrs.push(addr);
    }

    // Spawn accept tasks first so connections don't deadlock.
    let mut accept_tasks = Vec::with_capacity(num_ranks);
    for rank in 0..num_ranks {
        let listener = listeners.remove(0);
        accept_tasks.push(tokio::spawn(async move {
            let (stream, _peer) = listener
                .accept()
                .await
                .map_err(|e| RingError::Io(e.to_string()))?;
            Ok::<_, RingError>((rank, stream))
        }));
    }

    // Give accept tasks a moment to start listening.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Each rank connects to its right neighbour.
    let mut connect_tasks = Vec::with_capacity(num_ranks);
    for rank in 0..num_ranks {
        let right_addr = addrs[(rank + 1) % num_ranks];
        connect_tasks.push(tokio::spawn(async move {
            let stream = TcpStream::connect(right_addr)
                .await
                .map_err(|e| RingError::Io(e.to_string()))?;
            Ok::<_, RingError>((rank, stream))
        }));
    }

    let mut recv_streams = std::collections::HashMap::new();
    for task in accept_tasks {
        let (rank, stream) = task.await.map_err(|e| RingError::Io(e.to_string()))??;
        recv_streams.insert(rank, stream);
    }

    let mut send_streams = std::collections::HashMap::new();
    for task in connect_tasks {
        let (rank, stream) = task.await.map_err(|e| RingError::Io(e.to_string()))??;
        send_streams.insert(rank, stream);
    }

    let mut backends = Vec::with_capacity(num_ranks);
    for rank in 0..num_ranks {
        let recv = recv_streams.remove(&rank).ok_or(RingError::NotConnected)?;
        let send = send_streams.remove(&rank).ok_or(RingError::NotConnected)?;
        let transport = DualTcpTransport::new(send, recv);
        backends.push(RingBackend::new(rank, num_ranks, Box::new(transport)));
    }

    Ok(backends)
}

// ---- byte conversion helpers ----

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bytes_to_f32_slice_into(bytes: &[u8], out: &mut [f32]) -> Result<(), RingError> {
    if bytes.len() != out.len() * 4 {
        return Err(RingError::WrongChunkSize {
            expected: out.len() * 4,
            actual: bytes.len(),
        });
    }
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ring_all_reduce_two_ranks() {
        let mut backends = create_mock_ring(2);
        let mut data0 = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut data1 = vec![10.0_f32, 20.0, 30.0, 40.0];

        let (mut b0, mut b1) = (backends.remove(0), backends.remove(0));
        let (res0, res1) = tokio::join!(
            b0.all_sum(&mut data0),
            b1.all_sum(&mut data1),
        );
        res0.expect("rank 0 all_sum should succeed");
        res1.expect("rank 1 all_sum should succeed");

        let expected = vec![11.0_f32, 22.0, 33.0, 44.0];
        assert_eq!(data0, expected);
        assert_eq!(data1, expected);
    }

    #[tokio::test]
    async fn ring_all_reduce_four_ranks_identity() {
        let backends = create_mock_ring(4);
        let mut handles = Vec::with_capacity(4);
        for (rank, mut backend) in backends.into_iter().enumerate() {
            let data: Vec<f32> = (0..8).map(|i| (rank * 10 + i) as f32).collect();
            handles.push(tokio::spawn(async move {
                let mut d = data;
                backend.all_sum(&mut d).await?;
                Ok::<_, RingError>(d)
            }));
        }

        let mut results = Vec::new();
        for h in handles {
            results.push(h.await.unwrap().unwrap());
        }

        let expected: Vec<f32> = (0..8)
            .map(|i| (0..4).map(|r| (r * 10 + i) as f32).sum::<f32>())
            .collect();
        for r in &results {
            assert_eq!(r, &expected);
        }
    }

    #[tokio::test]
    async fn ring_all_reduce_single_rank_noop() {
        let mut backend = create_mock_ring(1).into_iter().next().unwrap();
        let mut data = vec![1.0_f32, 2.0, 3.0];
        backend.all_sum(&mut data).await.unwrap();
        assert_eq!(data, vec![1.0_f32, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn ring_all_gather_four_ranks() {
        let backends = create_mock_ring(4);
        let chunk_size = 4;
        let mut handles = Vec::with_capacity(4);
        for (rank, mut backend) in backends.into_iter().enumerate() {
            let data: Vec<f32> = (0..chunk_size).map(|i| (rank * 10 + i) as f32).collect();
            let mut out = vec![0.0_f32; chunk_size * 4];
            handles.push(tokio::spawn(async move {
                backend.all_gather(&data, &mut out).await?;
                Ok::<_, RingError>(out)
            }));
        }

        let mut results = Vec::new();
        for h in handles {
            results.push(h.await.unwrap().unwrap());
        }

        let expected: Vec<f32> = (0..4)
            .flat_map(|r| (0..chunk_size).map(move |i| (r * 10 + i) as f32))
            .collect();
        for r in &results {
            assert_eq!(r, &expected);
        }
    }

    #[tokio::test]
    async fn ring_all_gather_single_rank() {
        let mut backend = create_mock_ring(1).into_iter().next().unwrap();
        let data = vec![1.0_f32, 2.0, 3.0];
        let mut out = vec![0.0_f32; 3];
        backend.all_gather(&data, &mut out).await.unwrap();
        assert_eq!(out, vec![1.0_f32, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn ring_all_reduce_1mib_tcp_under_5s() {
        // 1 MiB = 262_144 f32 values.  Must be divisible by 4.
        let n_elems = 262_144;
        let num_ranks = 4;
        let backends = create_tcp_ring(num_ranks).await.expect("tcp ring should bind");

        let mut handles = Vec::with_capacity(num_ranks);
        for (rank, mut backend) in backends.into_iter().enumerate() {
            let data: Vec<f32> = (0..n_elems).map(|i| (rank as f32) * (i as f32)).collect();
            handles.push(tokio::spawn(async move {
                let mut d = data;
                let start = std::time::Instant::now();
                backend.all_sum(&mut d).await?;
                let elapsed = start.elapsed();
                Ok::<_, RingError>((d, elapsed))
            }));
        }

        let mut results = Vec::new();
        for h in handles {
            results.push(h.await.unwrap().unwrap());
        }

        let expected: Vec<f32> = (0..n_elems)
            .map(|i| (0..num_ranks).map(|r| (r as f32) * (i as f32)).sum::<f32>())
            .collect();
        for (data, elapsed) in &results {
            assert_eq!(data, &expected);
            assert!(
                elapsed.as_secs_f64() < 5.0,
                "all_reduce took {:.2}s, expected < 5s",
                elapsed.as_secs_f64()
            );
        }
    }
}
