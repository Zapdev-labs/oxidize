//! Model sharding engine and distributed parallelism helpers.
//!
//! Provides:
//! - `ShardPlan` broadcast via GossipSub COMMANDS.
//! - Pipeline parallelism (layer ranges with activation send/recv).
//! - Tensor parallelism (weight splits with all_sum over the ring).

use serde::{Deserialize, Serialize};

use super::ring::{RingBackend, RingError};
use super::topology::TopologyGraph;

/// A shard assignment for a single worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardAssignment {
    /// Pipeline stage: contiguous layer range [start, end).
    Pipeline { start_layer: usize, end_layer: usize },
    /// Tensor-parallel shard: column or row split index.
    Tensor { split_index: usize, total_splits: usize },
}

/// Full sharding plan broadcast by the master.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardPlan {
    pub model_id: String,
    pub total_layers: usize,
    pub strategy: ParallelismStrategy,
    /// Worker ID -> assignment.
    pub assignments: std::collections::HashMap<String, ShardAssignment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelismStrategy {
    Pipeline,
    Tensor,
}

/// Compute a shard plan from the topology graph.
///
/// If `strategy` is `Pipeline`, layers are split contiguously across peers.
/// If `strategy` is `Tensor`, each layer is split by the number of peers.
///
/// The local node is included as a worker if it is marked `can_shard`.
pub fn compute_shard_plan(
    topology: &TopologyGraph,
    model_id: String,
    total_layers: usize,
    strategy: ParallelismStrategy,
) -> ShardPlan {
    let mut peers: Vec<String> = topology
        .nodes
        .iter()
        .filter(|(_, n)| n.capabilities.can_shard)
        .map(|(id, _)| id.clone())
        .collect();

    // Include local node if it can shard.
    if let Some(local) = &topology.local_peer_id
        && !peers.contains(local)
    {
        peers.push(local.clone());
    }

    peers.sort();
    let num_workers = peers.len().max(1);
    let mut assignments = std::collections::HashMap::with_capacity(num_workers);

    match strategy {
        ParallelismStrategy::Pipeline => {
            let base = total_layers / num_workers;
            let rem = total_layers % num_workers;
            let mut start = 0usize;
            for (i, peer_id) in peers.iter().enumerate() {
                let width = base + usize::from(i < rem);
                let end = (start + width).min(total_layers);
                assignments.insert(
                    peer_id.clone(),
                    ShardAssignment::Pipeline {
                        start_layer: start,
                        end_layer: end,
                    },
                );
                start = end;
            }
        }
        ParallelismStrategy::Tensor => {
            for peer_id in &peers {
                assignments.insert(
                    peer_id.clone(),
                    ShardAssignment::Tensor {
                        split_index: 0,
                        total_splits: num_workers,
                    },
                );
            }
        }
    }

    ShardPlan {
        model_id,
        total_layers,
        strategy,
        assignments,
    }
}

/// Identify the local shard assignment from a plan.
pub fn local_assignment<'a>(
    plan: &'a ShardPlan,
    local_peer_id: &str,
) -> Option<&'a ShardAssignment> {
    plan.assignments.get(local_peer_id)
}

/// Pipeline stage result: activations passed to the next node.
#[allow(dead_code)]
pub struct PipelineStageResult {
    pub activations: Vec<f32>,
}

/// Tensor-parallel partial result.
#[allow(dead_code)]
pub struct TensorParallelResult {
    pub partial_output: Vec<f32>,
}

/// Send activations to the next pipeline stage (right neighbour in the
/// pipeline ordering).
///
/// Uses the ring transport for the data plane.
pub async fn pipeline_send(
    ring: &mut RingBackend,
    activations: Vec<f32>,
) -> Result<(), RingError> {
    let bytes = f32_slice_to_bytes(&activations);
    ring.transport.send_to_right(bytes).await
}

/// Receive activations from the previous pipeline stage (left neighbour).
pub async fn pipeline_recv(ring: &mut RingBackend, num_floats: usize) -> Result<Vec<f32>, RingError> {
    let bytes = ring.transport.recv_from_left().await?;
    let mut out = vec![0.0_f32; num_floats];
    bytes_to_f32_slice_into(&bytes, &mut out)?;
    Ok(out)
}

/// Perform a tensor-parallel all_sum over the ring.
///
/// Each rank holds a partial output; after `all_sum` every rank has the
/// same full output.
pub async fn tensor_parallel_all_sum(
    ring: &mut RingBackend,
    partial: &mut [f32],
) -> Result<(), RingError> {
    ring.all_sum(partial).await
}

/// Gather outputs from all ranks so every rank has the full concatenation.
pub async fn tensor_parallel_all_gather(
    ring: &mut RingBackend,
    partial: &[f32],
    out: &mut [f32],
) -> Result<(), RingError> {
    ring.all_gather(partial, out).await
}

// ---- byte helpers (duplicated from ring.rs to keep module self-contained) ----

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
    use crate::mesh::topology::TopologyGraph;
    use crate::mesh::node::NodeCapabilities;
    use std::collections::HashMap;

    fn dummy_caps(can_shard: bool) -> NodeCapabilities {
        NodeCapabilities {
            device_type: "cpu".to_string(),
            memory_bytes: 8_000_000_000,
            cpu_threads: 8,
            can_shard,
            tags: HashMap::new(),
        }
    }

    fn make_topology_with_local(local: &str, peers: &[&str]) -> TopologyGraph {
        let mut graph = TopologyGraph::new();
        graph.local_peer_id = Some(local.to_string());
        graph.add_or_update_node(local, dummy_caps(true));
        for peer in peers {
            graph.add_or_update_node(peer, dummy_caps(true));
        }
        graph
    }

    #[test]
    fn pipeline_plan_splits_contiguous_layers() {
        let graph = make_topology_with_local("a", &["b", "c"]);
        let plan = compute_shard_plan(&graph, "m".to_string(), 9, ParallelismStrategy::Pipeline);
        assert_eq!(plan.strategy, ParallelismStrategy::Pipeline);
        assert_eq!(plan.assignments.len(), 3);
        assert_eq!(
            plan.assignments.get("a"),
            Some(&ShardAssignment::Pipeline {
                start_layer: 0,
                end_layer: 3,
            })
        );
        assert_eq!(
            plan.assignments.get("b"),
            Some(&ShardAssignment::Pipeline {
                start_layer: 3,
                end_layer: 6,
            })
        );
        assert_eq!(
            plan.assignments.get("c"),
            Some(&ShardAssignment::Pipeline {
                start_layer: 6,
                end_layer: 9,
            })
        );
    }

    #[test]
    fn pipeline_plan_balances_remainder() {
        let graph = make_topology_with_local("a", &["b"]);
        let plan = compute_shard_plan(&graph, "m".to_string(), 5, ParallelismStrategy::Pipeline);
        assert_eq!(
            plan.assignments.get("a"),
            Some(&ShardAssignment::Pipeline {
                start_layer: 0,
                end_layer: 3,
            })
        );
        assert_eq!(
            plan.assignments.get("b"),
            Some(&ShardAssignment::Pipeline {
                start_layer: 3,
                end_layer: 5,
            })
        );
    }

    #[test]
    fn tensor_plan_creates_tensor_assignments() {
        let graph = make_topology_with_local("a", &["b"]);
        let plan = compute_shard_plan(&graph, "m".to_string(), 4, ParallelismStrategy::Tensor);
        assert_eq!(plan.strategy, ParallelismStrategy::Tensor);
        assert_eq!(
            plan.assignments.get("a"),
            Some(&ShardAssignment::Tensor {
                split_index: 0,
                total_splits: 2,
            })
        );
    }

    #[test]
    fn local_assignment_finds_worker_shard() {
        let graph = make_topology_with_local("a", &["b"]);
        let plan = compute_shard_plan(&graph, "m".to_string(), 4, ParallelismStrategy::Pipeline);
        assert_eq!(
            local_assignment(&plan, "a"),
            Some(&ShardAssignment::Pipeline {
                start_layer: 0,
                end_layer: 2,
            })
        );
        assert!(local_assignment(&plan, "unknown").is_none());
    }

    #[tokio::test]
    async fn pipeline_send_recv_roundtrip() {
        use crate::mesh::ring::create_mock_ring;
        let mut rings = create_mock_ring(2);
        let activations = vec![1.0_f32, 2.0, 3.0];

        let (mut r0, mut r1) = (rings.remove(0), rings.remove(0));
        let (send_res, recv_res) = tokio::join!(
            pipeline_send(&mut r0, activations.clone()),
            pipeline_recv(&mut r1, 3),
        );
        send_res.expect("send should succeed");
        let received = recv_res.expect("recv should succeed");
        assert_eq!(received, activations);
    }

    #[tokio::test]
    async fn tensor_parallel_all_sum_produces_identical_outputs() {
        use crate::mesh::ring::create_mock_ring;
        let rings = create_mock_ring(2);
        let mut handles = Vec::with_capacity(2);
        for (rank, mut ring) in rings.into_iter().enumerate() {
            let data: Vec<f32> = (0..4).map(|i| (rank * 10 + i) as f32).collect();
            handles.push(tokio::spawn(async move {
                let mut d = data;
                tensor_parallel_all_sum(&mut ring, &mut d).await?;
                Ok::<_, RingError>(d)
            }));
        }
        let mut results = Vec::new();
        for h in handles {
            results.push(h.await.unwrap().unwrap());
        }
        let expected = vec![10.0_f32, 12.0, 14.0, 16.0];
        for r in &results {
            assert_eq!(r, &expected);
        }
    }

    #[tokio::test]
    async fn all_gather_pipeline_end() {
        use crate::mesh::ring::create_mock_ring;
        let rings = create_mock_ring(4);
        let chunk_size = 4;
        let mut handles = Vec::with_capacity(4);
        for (rank, mut ring) in rings.into_iter().enumerate() {
            let data: Vec<f32> = (0..chunk_size).map(|i| (rank * 10 + i) as f32).collect();
            let mut out = vec![0.0_f32; chunk_size * 4];
            handles.push(tokio::spawn(async move {
                tensor_parallel_all_gather(&mut ring, &data, &mut out).await?;
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
}
