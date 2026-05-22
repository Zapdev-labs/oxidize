//! Mesh node state and configuration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Capability summary advertised by a mesh node during discovery.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeCapabilities {
    /// Device type string (e.g. "cpu", "mlx", "cuda").
    pub device_type: String,
    /// Approximate available memory in bytes.
    pub memory_bytes: u64,
    /// Number of CPU threads / cores.
    pub cpu_threads: usize,
    /// Whether the node can act as a model shard worker.
    pub can_shard: bool,
    /// Extra key/value tags for future extensibility.
    pub tags: HashMap<String, String>,
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            device_type: "cpu".to_string(),
            memory_bytes: std::env::var("OXIDIZE_MESH_MEMORY_BYTES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8_000_000_000),
            cpu_threads: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(8),
            can_shard: true,
            tags: HashMap::new(),
        }
    }
}

/// Configuration for a mesh node.
#[derive(Debug, Clone)]
pub struct MeshConfig {
    /// libp2p listening port (0 = ephemeral).
    pub listen_port: u16,
    /// mDNS namespace for cluster isolation.
    pub namespace: String,
    /// Capabilities advertised to peers.
    pub capabilities: NodeCapabilities,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            listen_port: 0,
            namespace: Self::default_namespace(),
            capabilities: NodeCapabilities::default(),
        }
    }
}

impl MeshConfig {
    /// Namespace from env or default.
    pub fn default_namespace() -> String {
        std::env::var("OXIDIZE_MESH_NAMESPACE")
            .or_else(|_| std::env::var("EXO_LIBP2P_NAMESPACE"))
            .unwrap_or_else(|_| "default".to_string())
    }
}

/// Local mesh node state.
#[derive(Debug)]
pub struct MeshNode {
    pub config: MeshConfig,
}

impl MeshNode {
    pub fn new(config: MeshConfig) -> Self {
        Self { config }
    }
}
