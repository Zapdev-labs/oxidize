# oxidize-core/src/mesh/

**Generated:** 2026-05-26
**Domain:** Distributed inference (libp2p mesh)

## OVERVIEW
libp2p-based distributed mesh for multi-node LLM inference with leader election, model sharding, and fault tolerance.

## STRUCTURE
```
mesh/
├── mod.rs              # Privacy boundary — all exports gated here
├── node.rs             # MeshNode, MeshConfig, NodeCapabilities
├── discovery.rs        # mDNS peer discovery, swarm builder, run_mesh_node event loop
├── election.rs         # BullyElection deterministic leader election
├── gossip.rs           # GossipSub topics (6), MeshEnvelope, GossipRouter session invalidation
├── topology.rs         # TopologyGraph peer tracking, stale eviction
├── sharding.rs         # ShardPlan, pipeline/tensor parallelism over ring
├── ring.rs             # RingTransport trait, all_sum/all_gather collectives
├── chat.rs             # Distributed chat engine, prompt/token routing
├── progress.rs         # AggregatedProgress, cluster progress bars
├── fault_tolerance.rs  # eval_with_timeout, RunnerStatus, ShutdownTask
└── scrutiny.rs         # MeshValidationReport, plan validation
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add GossipSub topic | `gossip.rs` | Update `TopicKind` enum + `all()` array |
| Change election logic | `election.rs` | Priority tuple ordering is deterministic |
| New transport backend | `ring.rs` | Implement `RingTransport` trait |
| Shard strategy | `sharding.rs` | `ParallelismStrategy::Pipeline` or `Tensor` |
| Event loop changes | `discovery.rs` | `run_mesh_node()` is the main async loop |
| Timeout / recovery | `fault_tolerance.rs` | `DEFAULT_COLLECTIVE_TIMEOUT` = 60s |

## CONVENTIONS
- **Privacy boundary**: `mod.rs` is the ONLY real mod.rs in oxidize-core. All submodules are `mod` private; public API is explicitly `pub use` re-exported.
- **Session invalidation**: Every message carries an `election_clock`. `GossipRouter::accept()` drops stale messages after new election.
- **Namespace isolation**: Topics are namespaced (`oxidize/mesh/{ns}/commands`). Cross-namespace peers are ignored.
- **Deterministic election**: Winner is max `(clock, seniority, commands_seen, peer_id)` — no randomness, all nodes converge independently.
- **Test transport**: `create_mock_ring()` for unit tests; `create_tcp_ring()` for integration.

## ANTI-PATTERNS
- `run_mesh_node()` is 400+ lines — event loop mixes discovery, election, chat, sharding, fault tolerance. Refactor candidate.
- Hardcoded parallelism strategy in discovery.rs (`ParallelismStrategy::Pipeline`) — should come from config.
- `unwrap()` in `run_mesh_node()` on serde and multiaddr parsing — should propagate errors.
- Heartbeat timeout (15s) and election timeout (3s) are magic constants — should be `MeshConfig` fields.
