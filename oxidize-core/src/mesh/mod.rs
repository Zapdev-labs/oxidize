//! Distributed mesh networking layer.
//!
//! Provides peer discovery via libp2p + mDNS, GossipSub control plane,
//! and foundational types for leader election and sharding.

mod discovery;
mod gossip;
mod node;

pub use discovery::{build_swarm, generate_identity, run_mesh_node, DiscoveryEvent, DiscoveryPayload, DiscoveryService, same_namespace};
pub use gossip::{GossipMessage, GossipRouter, MeshBehaviour, MeshEvent, TopicKind};
pub use node::{MeshConfig, MeshNode, NodeCapabilities};
