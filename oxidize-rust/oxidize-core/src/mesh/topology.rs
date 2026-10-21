//! Mesh topology graph — tracks peers, edges, and capabilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::node::NodeCapabilities;

/// A node in the mesh topology graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TopologyNode {
    pub peer_id: String,
    pub capabilities: NodeCapabilities,
    /// How many commands this node has processed (used for tie-breaking).
    pub commands_seen: u64,
    /// Monotonic join counter / seniority score.
    pub seniority: u64,
    #[serde(skip)]
    pub last_seen: Option<Instant>,
    #[serde(skip)]
    pub joined_at: Option<Instant>,
}

impl TopologyNode {
    pub fn new(peer_id: String, capabilities: NodeCapabilities) -> Self {
        Self {
            peer_id,
            capabilities,
            commands_seen: 0,
            seniority: 0,
            last_seen: Some(Instant::now()),
            joined_at: Some(Instant::now()),
        }
    }

    /// Update last_seen timestamp to now.
    pub fn heartbeat(&mut self) {
        self.last_seen = Some(Instant::now());
    }

    /// True if we have not received a heartbeat within `timeout`.
    pub fn is_stale(&self, timeout: Duration) -> bool {
        self.last_seen
            .map(|t| t.elapsed() > timeout)
            .unwrap_or(true)
    }

    /// Increment the commands-seen counter.
    pub fn inc_commands(&mut self) {
        self.commands_seen += 1;
    }
}

/// An edge (connection) between two nodes in the topology graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TopologyEdge {
    pub from: String,
    pub to: String,
    #[serde(skip)]
    pub established_at: Option<Instant>,
}

/// The mesh topology graph.
///
/// Tracks every known peer as a [`TopologyNode`] and every known
/// connection as a [`TopologyEdge`].  Provides capability queries
/// and stale-node eviction.
#[derive(Debug, Default)]
pub struct TopologyGraph {
    /// Nodes indexed by peer_id string.
    pub nodes: HashMap<String, TopologyNode>,
    /// Undirected-ish edges (stored as directed pairs; callers dedupe).
    pub edges: Vec<TopologyEdge>,
    /// Local node's peer_id, if known.
    pub local_peer_id: Option<String>,
}

impl TopologyGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or update a peer node.
    pub fn add_or_update_node(&mut self, peer_id: &str, capabilities: NodeCapabilities) {
        match self.nodes.get_mut(peer_id) {
            Some(existing) => {
                existing.capabilities = capabilities;
                existing.heartbeat();
            }
            None => {
                self.nodes.insert(
                    peer_id.to_string(),
                    TopologyNode::new(peer_id.to_string(), capabilities),
                );
            }
        }
    }

    /// Remove a node and all edges touching it.
    pub fn remove_node(&mut self, peer_id: &str) {
        self.nodes.remove(peer_id);
        self.edges.retain(|e| e.from != peer_id && e.to != peer_id);
    }

    /// Record a directed edge (both directions are usually added).
    pub fn add_edge(&mut self, from: &str, to: &str) {
        let already = self
            .edges
            .iter()
            .any(|e| (e.from == from && e.to == to) || (e.from == to && e.to == from));
        if !already {
            self.edges.push(TopologyEdge {
                from: from.to_string(),
                to: to.to_string(),
                established_at: Some(Instant::now()),
            });
        }
    }

    /// Remove all edges touching a peer (used when a peer disconnects).
    pub fn remove_edges_for(&mut self, peer_id: &str) {
        self.edges.retain(|e| e.from != peer_id && e.to != peer_id);
    }

    /// Evict nodes that have not been seen within `timeout`.
    pub fn evict_stale(&mut self, timeout: Duration) -> Vec<String> {
        let stale: Vec<String> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.is_stale(timeout))
            .map(|(id, _)| id.clone())
            .collect();
        if stale.is_empty() {
            return stale;
        }
        let stale_set: std::collections::HashSet<&str> = stale.iter().map(|s| s.as_str()).collect();
        self.nodes.retain(|id, _| !stale_set.contains(id.as_str()));
        self.edges
            .retain(|e| !stale_set.contains(e.from.as_str()) && !stale_set.contains(e.to.as_str()));
        stale
    }

    /// All currently known peer IDs (excluding local, if set).
    pub fn peer_ids(&self) -> Vec<String> {
        self.nodes
            .keys()
            .filter(|id| self.local_peer_id.as_deref() != Some(id.as_str()))
            .cloned()
            .collect()
    }

    /// Total number of known peers.
    pub fn peer_count(&self) -> usize {
        self.nodes.len()
    }

    /// Aggregate capability summary across all peers.
    pub fn aggregate_capabilities(&self) -> AggregateCapabilities {
        let mut total_memory = 0u64;
        let mut total_threads = 0usize;
        let mut can_shard_count = 0usize;
        let mut device_types = std::collections::HashSet::new();

        for node in self.nodes.values() {
            total_memory += node.capabilities.memory_bytes;
            total_threads += node.capabilities.cpu_threads;
            if node.capabilities.can_shard {
                can_shard_count += 1;
            }
            device_types.insert(node.capabilities.device_type.clone());
        }

        AggregateCapabilities {
            node_count: self.nodes.len(),
            total_memory_bytes: total_memory,
            total_cpu_threads: total_threads,
            can_shard_nodes: can_shard_count,
            device_types: device_types.into_iter().collect(),
        }
    }

    /// Lookup a peer's capabilities, if known.
    pub fn capabilities_of(&self, peer_id: &str) -> Option<&NodeCapabilities> {
        self.nodes.get(peer_id).map(|n| &n.capabilities)
    }

    /// Lookup a mutable reference to a peer node.
    pub fn node_mut(&mut self, peer_id: &str) -> Option<&mut TopologyNode> {
        self.nodes.get_mut(peer_id)
    }
}

/// Summary of capabilities across the whole mesh.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateCapabilities {
    pub node_count: usize,
    pub total_memory_bytes: u64,
    pub total_cpu_threads: usize,
    pub can_shard_nodes: usize,
    pub device_types: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn graph_tracks_nodes() {
        let mut graph = TopologyGraph::new();
        graph.add_or_update_node("peer-a", dummy_caps());
        assert_eq!(graph.peer_count(), 1);
        assert!(graph.capabilities_of("peer-a").is_some());
    }

    #[test]
    fn graph_updates_existing_node() {
        let mut graph = TopologyGraph::new();
        graph.add_or_update_node("peer-a", dummy_caps());
        let mut updated = dummy_caps();
        updated.memory_bytes = 16_000_000_000;
        graph.add_or_update_node("peer-a", updated.clone());
        assert_eq!(
            graph.capabilities_of("peer-a").unwrap().memory_bytes,
            16_000_000_000
        );
    }

    #[test]
    fn graph_removes_node_and_edges() {
        let mut graph = TopologyGraph::new();
        graph.add_or_update_node("peer-a", dummy_caps());
        graph.add_or_update_node("peer-b", dummy_caps());
        graph.add_edge("peer-a", "peer-b");
        graph.remove_node("peer-a");
        assert_eq!(graph.peer_count(), 1);
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn graph_evicts_stale_nodes() {
        let mut graph = TopologyGraph::new();
        graph.add_or_update_node("peer-a", dummy_caps());
        // Manually make the node stale
        if let Some(node) = graph.node_mut("peer-a") {
            node.last_seen = Some(Instant::now() - Duration::from_secs(100));
        }
        let evicted = graph.evict_stale(Duration::from_secs(30));
        assert_eq!(evicted, vec!["peer-a"]);
        assert_eq!(graph.peer_count(), 0);
    }

    #[test]
    fn aggregate_capabilities_sum() {
        let mut graph = TopologyGraph::new();
        graph.add_or_update_node("peer-a", dummy_caps());
        let mut caps = dummy_caps();
        caps.memory_bytes = 4_000_000_000;
        caps.device_type = "mlx".to_string();
        graph.add_or_update_node("peer-b", caps);
        let agg = graph.aggregate_capabilities();
        assert_eq!(agg.node_count, 2);
        assert_eq!(agg.total_memory_bytes, 12_000_000_000);
        assert_eq!(agg.total_cpu_threads, 16);
        assert_eq!(agg.can_shard_nodes, 2);
        assert!(agg.device_types.contains(&"cpu".to_string()));
        assert!(agg.device_types.contains(&"mlx".to_string()));
    }

    #[test]
    fn graph_dedupes_edges() {
        let mut graph = TopologyGraph::new();
        graph.add_edge("a", "b");
        graph.add_edge("a", "b");
        graph.add_edge("b", "a");
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn topology_node_tracks_commands() {
        let mut node = TopologyNode::new("p".to_string(), dummy_caps());
        node.inc_commands();
        node.inc_commands();
        assert_eq!(node.commands_seen, 2);
    }

    #[test]
    fn peer_ids_excludes_local() {
        let mut graph = TopologyGraph::new();
        graph.local_peer_id = Some("me".to_string());
        graph.add_or_update_node("me", dummy_caps());
        graph.add_or_update_node("peer-a", dummy_caps());
        let ids = graph.peer_ids();
        assert_eq!(ids, vec!["peer-a"]);
    }
}
