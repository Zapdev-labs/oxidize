//! Distributed mesh networking layer.
//!
//! Provides peer discovery via libp2p + mDNS, GossipSub control plane,
//! leader election, topology tracking, and foundational types for sharding.

mod discovery;
mod election;
mod gossip;
mod node;
mod topology;

pub use discovery::{build_swarm, generate_identity, run_mesh_node, DiscoveryEvent, DiscoveryPayload, DiscoveryService, same_namespace};
pub use election::{BullyElection, ElectionClock, ElectionMessage, ElectionState, Priority, run_election_round};
pub use gossip::{GossipMessage, GossipRouter, MeshBehaviour, MeshEvent, TopicKind};
pub use node::{MeshConfig, MeshNode, NodeCapabilities};
pub use topology::{AggregateCapabilities, TopologyEdge, TopologyGraph, TopologyNode};
