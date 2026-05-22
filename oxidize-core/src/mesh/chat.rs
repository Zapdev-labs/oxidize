//! Distributed chat engine for mesh nodes.
//!
//! Provides message types and the [`MeshChatEngine`] that orchestrates
//! prompt broadcasting, simulated distributed forward passes, and token
//! streaming across the mesh.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

use super::gossip::MeshEnvelope;
use super::ring::{RingBackend, RingError};
use super::sharding::{ShardAssignment, ShardPlan, local_assignment, pipeline_send, pipeline_recv, tensor_parallel_all_sum, tensor_parallel_all_gather};

/// A chat prompt broadcast by a client (CLI or HTTP) to the mesh master
/// via the `COMMANDS` topic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshChatPrompt {
    pub request_id: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}

/// A single streaming token broadcast by the master on `GLOBAL_EVENTS`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshChatToken {
    pub request_id: String,
    pub token: String,
    pub index: usize,
    pub is_final: bool,
}

/// A complete response broadcast when generation finishes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshChatResponse {
    pub request_id: String,
    pub content: String,
    pub finish_reason: String,
    pub tokens_generated: usize,
}

/// Command variants sent on the mesh `COMMANDS` topic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum MeshCommand {
    ChatPrompt(MeshChatPrompt),
    Shutdown(super::fault_tolerance::ShutdownTask),
    ShardPlan(super::sharding::ShardPlan),
}

/// Distributed chat engine embedded in the mesh node event loop.
///
/// - **Master** receives [`MeshChatPrompt`]s on `COMMANDS` (or from the
///   local CLI via [`prompt_rx`]), runs a simulated distributed forward
///   pass through pipeline/tensor stages, and broadcasts tokens on
///   `GLOBAL_EVENTS`.
/// - **Workers** participate in the distributed forward pass when they
///   receive the prompt (or when the master tells them to via the
///   pipeline/tensor protocol).
///
/// In the current implementation the forward pass is *simulated* using
/// synthetic activations passed through the real ring collectives.  This
/// validates end-to-end wiring without requiring a loaded model.
#[derive(Debug)]
pub struct MeshChatEngine {
    /// If true, this node is the elected master.
    pub is_master: bool,
    /// Local peer id string.
    pub local_peer_id: String,
    /// Current election clock (for session validation).
    pub clock: u64,
    /// Active shard plan, if any.
    pub shard_plan: Option<ShardPlan>,
    /// Token stream receivers per request (CLI side).
    pub token_sinks: Arc<Mutex<HashMap<String, mpsc::UnboundedSender<MeshChatToken>>>>,
    /// Ring backend for data-plane collectives.
    pub ring: Option<RingBackend>,
    /// Receiver for prompts injected by the local CLI.
    pub prompt_rx: Option<mpsc::UnboundedReceiver<MeshChatPrompt>>,
    /// Sender for streaming tokens back to the local CLI.
    pub token_tx: Option<mpsc::UnboundedSender<MeshChatToken>>,
}

impl MeshChatEngine {
    pub fn new(is_master: bool, local_peer_id: String, clock: u64) -> Self {
        Self {
            is_master,
            local_peer_id,
            clock,
            shard_plan: None,
            token_sinks: Arc::new(Mutex::new(HashMap::new())),
            ring: None,
            prompt_rx: None,
            token_tx: None,
        }
    }

    /// Register a token sink so the CLI can receive streaming tokens.
    pub async fn register_sink(
        &self,
        request_id: &str,
        tx: mpsc::UnboundedSender<MeshChatToken>,
    ) {
        let mut sinks = self.token_sinks.lock().await;
        sinks.insert(request_id.to_string(), tx);
    }

    /// Unregister a token sink.
    pub async fn unregister_sink(&self, request_id: &str) {
        let mut sinks = self.token_sinks.lock().await;
        sinks.remove(request_id);
    }

    /// Handle an inbound [`MeshChatToken`] (received on `GLOBAL_EVENTS`).
    /// Forwards it to any locally-registered sink and to the local CLI
    /// `token_tx` if present.
    pub async fn handle_token(&self, token: MeshChatToken) {
        let sinks = self.token_sinks.lock().await;
        if let Some(tx) = sinks.get(&token.request_id) {
            let _ = tx.send(token.clone());
        }
        if let Some(ref tx) = self.token_tx {
            let _ = tx.send(token);
        }
    }

    /// Handle a [`MeshChatPrompt`] — master starts generation, workers
    /// participate in the distributed forward pass.
    ///
    /// Returns a sequence of tokens that the caller (master) should
    /// broadcast on `GLOBAL_EVENTS`.
    pub async fn handle_prompt(
        &mut self,
        prompt: &MeshChatPrompt,
    ) -> Vec<MeshChatToken> {
        let request_id = prompt.request_id.clone();
        let max_tokens = prompt.max_tokens;

        if self.is_master {
            // Simulate a distributed forward pass:
            // 1. Pipeline stages pass activations through the ring.
            // 2. Tensor parallelism all-sums partial outputs.
            // 3. Sample tokens deterministically from the prompt.
            let mut tokens = Vec::with_capacity(max_tokens);
            let words: Vec<&str> = prompt.prompt.split_whitespace().collect();
            for i in 0..max_tokens {
                let token_text = if words.is_empty() {
                    format!("tok-{i}")
                } else {
                    words.get(i % words.len()).unwrap_or(&"tok").to_string()
                };

                // --- simulated pipeline stage ---
                if let Some(ref mut ring) = self.ring {
                    let plan_ref = self.shard_plan.as_ref();
                    let _ = Self::run_pipeline_stage(ring, plan_ref, &self.local_peer_id, i).await;
                }

                let is_final = i == max_tokens.saturating_sub(1);
                tokens.push(MeshChatToken {
                    request_id: request_id.clone(),
                    token: token_text,
                    index: i,
                    is_final,
                });
            }
            tokens
        } else {
            // Worker: participate in the forward pass if we have a ring.
            if let Some(ref mut ring) = self.ring {
                let plan_ref = self.shard_plan.as_ref();
                for i in 0..max_tokens {
                    let _ = Self::run_pipeline_stage(ring, plan_ref, &self.local_peer_id, i).await;
                }
            }
            Vec::new()
        }
    }

    /// Simulate one pipeline stage using the real ring backend.
    ///
    /// If this node holds the first pipeline stage, it sends synthetic
    /// activations to the right.  If it holds the last stage, it receives
    /// from the left and then `all_gather`s the result back.  Intermediate
    /// stages receive from left and send to right.
    async fn run_pipeline_stage(
        ring: &mut RingBackend,
        shard_plan: Option<&ShardPlan>,
        local_peer_id: &str,
        _step: usize,
    ) -> Result<(), RingError> {
        let assignment = shard_plan
            .and_then(|plan| local_assignment(plan, local_peer_id));

        match assignment {
            Some(ShardAssignment::Pipeline { start_layer, end_layer }) => {
                // Determine whether we are first, last, or middle stage.
                let num_workers = ring.num_ranks;
                let stage_index = ring.rank;
                let is_first = stage_index == 0;
                let is_last = stage_index == num_workers.saturating_sub(1);

                // Small synthetic activation vector (4 floats).
                let synthetic: Vec<f32> = vec![
                    stage_index as f32,
                    (*start_layer) as f32,
                    (*end_layer) as f32,
                    1.0,
                ];

                if is_first {
                    pipeline_send(ring, synthetic).await?;
                } else if is_last {
                    let received = pipeline_recv(ring, 4).await?;
                    // all_gather back to every rank so all have the result.
                    let mut gathered = vec![0.0_f32; 4 * num_workers];
                    tensor_parallel_all_gather(ring, &received, &mut gathered).await?;
                } else {
                    let received = pipeline_recv(ring, 4).await?;
                    pipeline_send(ring, received).await?;
                }
                Ok(())
            }
            Some(ShardAssignment::Tensor { .. }) => {
                // Tensor parallelism: all_sum a small synthetic partial.
                let partial = vec![ring.rank as f32; 4];
                let mut buf = partial.clone();
                tensor_parallel_all_sum(ring, &mut buf).await?;
                Ok(())
            }
            None => {
                // No assignment — nothing to do.
                Ok(())
            }
        }
    }

    /// Build deterministic tokens for a prompt without a real model.
    ///
    /// Used by the master when no ring is available (single-node mesh).
    pub fn generate_tokens_local(&self, prompt: &MeshChatPrompt) -> Vec<MeshChatToken> {
        let words: Vec<&str> = prompt.prompt.split_whitespace().collect();
        let mut tokens = Vec::with_capacity(prompt.max_tokens);
        for i in 0..prompt.max_tokens {
            let token_text = if words.is_empty() {
                format!("tok-{i}")
            } else {
                words.get(i % words.len()).unwrap_or(&"tok").to_string()
            };
            tokens.push(MeshChatToken {
                request_id: prompt.request_id.clone(),
                token: token_text,
                index: i,
                is_final: i == prompt.max_tokens.saturating_sub(1),
            });
        }
        tokens
    }
}

/// Serialize a [`MeshCommand`] into a [`GossipMessage`] payload.
pub fn encode_mesh_command(cmd: &MeshCommand, clock: u64) -> Result<Vec<u8>, serde_json::Error> {
    MeshEnvelope::pack(clock, cmd)
}

/// Deserialize a [`MeshCommand`] from raw GossipSub message data.
pub fn decode_mesh_command(data: &[u8]) -> Result<(u64, MeshCommand), serde_json::Error> {
    let (clock, inner) = MeshEnvelope::unpack(data)?;
    let cmd: MeshCommand = serde_json::from_slice(&inner)?;
    Ok((clock, cmd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ring::create_mock_ring;
    use crate::mesh::sharding::{compute_shard_plan, ParallelismStrategy};
    use crate::mesh::topology::TopologyGraph;
    use crate::mesh::node::NodeCapabilities;

    fn dummy_caps() -> NodeCapabilities {
        NodeCapabilities {
            device_type: "cpu".to_string(),
            memory_bytes: 8_000_000_000,
            cpu_threads: 8,
            can_shard: true,
            tags: HashMap::new(),
        }
    }

    #[test]
    fn mesh_chat_prompt_serializes() {
        let prompt = MeshChatPrompt {
            request_id: "req-1".into(),
            prompt: "hello mesh".into(),
            max_tokens: 5,
            temperature: 0.7,
            top_p: 0.9,
        };
        let json = serde_json::to_string(&prompt).unwrap();
        let back: MeshChatPrompt = serde_json::from_str(&json).unwrap();
        assert_eq!(prompt, back);
    }

    #[test]
    fn mesh_chat_token_serializes() {
        let token = MeshChatToken {
            request_id: "req-1".into(),
            token: "world".into(),
            index: 0,
            is_final: false,
        };
        let json = serde_json::to_string(&token).unwrap();
        let back: MeshChatToken = serde_json::from_str(&json).unwrap();
        assert_eq!(token, back);
    }

    #[test]
    fn mesh_command_serializes_roundtrip() {
        let cmd = MeshCommand::ChatPrompt(MeshChatPrompt {
            request_id: "r".into(),
            prompt: "p".into(),
            max_tokens: 3,
            temperature: 0.5,
            top_p: 0.8,
        });
        let encoded = encode_mesh_command(&cmd, 1).unwrap();
        let (clock, back) = decode_mesh_command(&encoded).unwrap();
        assert_eq!(clock, 1);
        assert_eq!(back, cmd);
    }

    #[test]
    fn engine_generates_tokens_local() {
        let engine = MeshChatEngine::new(true, "me".into(), 1);
        let prompt = MeshChatPrompt {
            request_id: "r".into(),
            prompt: "one two three".into(),
            max_tokens: 5,
            temperature: 0.0,
            top_p: 0.0,
        };
        let tokens = engine.generate_tokens_local(&prompt);
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].token, "one");
        assert_eq!(tokens[1].token, "two");
        assert_eq!(tokens[2].token, "three");
        assert_eq!(tokens[3].token, "one");
        assert_eq!(tokens[4].token, "two");
        assert!(tokens[4].is_final);
    }

    #[test]
    fn engine_generates_tokens_for_empty_prompt() {
        let engine = MeshChatEngine::new(true, "me".into(), 1);
        let prompt = MeshChatPrompt {
            request_id: "r".into(),
            prompt: "".into(),
            max_tokens: 3,
            temperature: 0.0,
            top_p: 0.0,
        };
        let tokens = engine.generate_tokens_local(&prompt);
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].token, "tok-0");
        assert_eq!(tokens[1].token, "tok-1");
        assert_eq!(tokens[2].token, "tok-2");
    }

    #[tokio::test]
    async fn engine_handle_prompt_as_master_with_ring() {
        let mut rings = create_mock_ring(2);
        let mut graph = TopologyGraph::new();
        graph.local_peer_id = Some("rank-0".into());
        graph.add_or_update_node("rank-0", dummy_caps());
        graph.add_or_update_node("rank-1", dummy_caps());
        let plan = compute_shard_plan(&graph, "m".into(), 4, ParallelismStrategy::Pipeline);

        let mut engine = MeshChatEngine::new(true, "rank-0".into(), 1);
        engine.shard_plan = Some(plan);
        engine.ring = Some(rings.remove(0));

        let prompt = MeshChatPrompt {
            request_id: "r".into(),
            prompt: "hello world".into(),
            max_tokens: 3,
            temperature: 0.0,
            top_p: 0.0,
        };
        let tokens = engine.handle_prompt(&prompt).await;
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].token, "hello");
        assert_eq!(tokens[1].token, "world");
        assert_eq!(tokens[2].token, "hello");
    }

    #[tokio::test]
    async fn engine_handle_prompt_as_worker_no_tokens() {
        let mut graph = TopologyGraph::new();
        graph.local_peer_id = Some("rank-1".into());
        graph.add_or_update_node("rank-0", dummy_caps());
        graph.add_or_update_node("rank-1", dummy_caps());
        let plan = compute_shard_plan(&graph, "m".into(), 4, ParallelismStrategy::Pipeline);

        let mut engine = MeshChatEngine::new(false, "rank-1".into(), 1);
        engine.shard_plan = Some(plan);
        // No ring — workers do not generate tokens regardless.

        let prompt = MeshChatPrompt {
            request_id: "r".into(),
            prompt: "hello".into(),
            max_tokens: 2,
            temperature: 0.0,
            top_p: 0.0,
        };
        let tokens = engine.handle_prompt(&prompt).await;
        // Workers do not generate tokens themselves.
        assert!(tokens.is_empty());
    }

    #[tokio::test]
    async fn engine_token_sink_routing() {
        let engine = MeshChatEngine::new(true, "me".into(), 1);
        let (tx, mut rx) = mpsc::unbounded_channel();
        engine.register_sink("r", tx).await;

        let token = MeshChatToken {
            request_id: "r".into(),
            token: "hi".into(),
            index: 0,
            is_final: false,
        };
        engine.handle_token(token.clone()).await;

        let received = rx.recv().await.unwrap();
        assert_eq!(received, token);
    }

    #[tokio::test]
    async fn engine_token_sink_ignores_unknown_request() {
        let engine = MeshChatEngine::new(true, "me".into(), 1);
        let (tx, mut rx) = mpsc::unbounded_channel();
        engine.register_sink("other", tx).await;

        let token = MeshChatToken {
            request_id: "r".into(),
            token: "hi".into(),
            index: 0,
            is_final: false,
        };
        engine.handle_token(token).await;
        // No panic and no message delivered.
        assert!(rx.try_recv().is_err());
    }
}
