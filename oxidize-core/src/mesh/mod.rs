//! Distributed mesh networking layer.
//!
//! Provides peer discovery via libp2p + mDNS, GossipSub control plane,
//! leader election, topology tracking, ring collectives, sharding,
//! fault tolerance, and distributed progress indicators.

mod chat;
mod discovery;
mod election;
mod fault_tolerance;
mod gossip;
mod node;
mod progress;
mod ring;
mod scrutiny;
mod sharding;
mod topology;

pub use chat::{
    MeshChatEngine, MeshChatPrompt, MeshChatResponse, MeshChatToken, MeshCommand,
    decode_mesh_command, encode_mesh_command,
};
pub use discovery::{
    DiscoveryEvent, DiscoveryPayload, DiscoveryService, broadcast_shard_plan, build_swarm,
    generate_identity, run_mesh_node, same_namespace,
};
pub use election::{
    BullyElection, ElectionClock, ElectionMessage, ElectionState, Priority, run_election_round,
};
pub use fault_tolerance::{
    DEFAULT_COLLECTIVE_TIMEOUT, RunnerStatus, RunnerStatusUpdated, ShutdownTask, TimedResult,
    eval_with_timeout, eval_with_timeout_and_notify,
};
pub use gossip::{GossipMessage, GossipRouter, MeshBehaviour, MeshEnvelope, MeshEvent, TopicKind};
pub use node::{MeshConfig, MeshNode, NodeCapabilities};
pub use progress::{
    AggregatedProgress, LoadProgressReport, aggregate_progress, render_cluster_progress_bar,
};
pub use ring::{
    ChannelTransport, DualTcpTransport, RingBackend, RingError, RingTransport, TcpTransport,
    create_mock_ring, create_tcp_ring,
};
pub use scrutiny::{
    MeshValidationReport, validate_mesh_command, validate_mesh_prompt, validate_node_capabilities,
    validate_shard_plan,
};
pub use sharding::{
    ParallelismStrategy, ShardAssignment, ShardPlan, compute_shard_plan, local_assignment,
    pipeline_recv, pipeline_send, tensor_parallel_all_gather, tensor_parallel_all_sum,
};
pub use topology::{AggregateCapabilities, TopologyEdge, TopologyGraph, TopologyNode};
