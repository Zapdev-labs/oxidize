Index your code with Devin

[DeepWiki](https://deepwiki.com/)

[DeepWiki](https://deepwiki.com/)

[exo-explore/exo](https://github.com/exo-explore/exo "Open repository")

Index your code with

Devin
Edit WikiShare

Last indexed: 29 March 2026 ( [1e51dc](https://github.com/exo-explore/exo/commits/1e51dc89))

- [Overview](https://deepwiki.com/exo-explore/exo/1-overview)
- [Getting Started](https://deepwiki.com/exo-explore/exo/2-getting-started)
- [Installation and Setup](https://deepwiki.com/exo-explore/exo/2.1-installation-and-setup)
- [Running Your First Model](https://deepwiki.com/exo-explore/exo/2.2-running-your-first-model)
- [Using the macOS Application](https://deepwiki.com/exo-explore/exo/2.3-using-the-macos-application)
- [Core Architecture](https://deepwiki.com/exo-explore/exo/3-core-architecture)
- [Node Orchestration](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration)
- [Event Sourcing and State Management](https://deepwiki.com/exo-explore/exo/3.2-event-sourcing-and-state-management)
- [Master and Worker Roles](https://deepwiki.com/exo-explore/exo/3.3-master-and-worker-roles)
- [Election and Failover](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover)
- [Model Placement and Sharding](https://deepwiki.com/exo-explore/exo/4-model-placement-and-sharding)
- [Placement Engine](https://deepwiki.com/exo-explore/exo/4.1-placement-engine)
- [Sharding Strategies](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies)
- [Communication Backends](https://deepwiki.com/exo-explore/exo/4.3-communication-backends)
- [Inference Execution](https://deepwiki.com/exo-explore/exo/5-inference-execution)
- [Runner Lifecycle](https://deepwiki.com/exo-explore/exo/5.1-runner-lifecycle)
- [Task Planning and Execution](https://deepwiki.com/exo-explore/exo/5.2-task-planning-and-execution)
- [MLX Backend and Model Loading](https://deepwiki.com/exo-explore/exo/5.3-mlx-backend-and-model-loading)
- [Distributed Parallelism Implementation](https://deepwiki.com/exo-explore/exo/5.4-distributed-parallelism-implementation)
- [Text Generation Pipeline](https://deepwiki.com/exo-explore/exo/5.5-text-generation-pipeline)
- [Image Generation Pipeline](https://deepwiki.com/exo-explore/exo/5.6-image-generation-pipeline)
- [Router and Messaging](https://deepwiki.com/exo-explore/exo/6-router-and-messaging)
- [Message Topics and ForwarderEvent](https://deepwiki.com/exo-explore/exo/6.1-message-topics-and-forwarderevent)
- [Topology Discovery](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery)
- [API and User Interfaces](https://deepwiki.com/exo-explore/exo/7-api-and-user-interfaces)
- [REST API Overview](https://deepwiki.com/exo-explore/exo/7.1-rest-api-overview)
- [API Endpoints Reference](https://deepwiki.com/exo-explore/exo/7.2-api-endpoints-reference)
- [Dashboard Web Interface](https://deepwiki.com/exo-explore/exo/7.3-dashboard-web-interface)
- [macOS Application Deep Dive](https://deepwiki.com/exo-explore/exo/7.4-macos-application-deep-dive)
- [Model Management](https://deepwiki.com/exo-explore/exo/8-model-management)
- [Model Cards and Metadata](https://deepwiki.com/exo-explore/exo/8.1-model-cards-and-metadata)
- [Model Downloads from HuggingFace](https://deepwiki.com/exo-explore/exo/8.2-model-downloads-from-huggingface)
- [Shard Distribution and Download Coordination](https://deepwiki.com/exo-explore/exo/8.3-shard-distribution-and-download-coordination)
- [Benchmarking and Performance](https://deepwiki.com/exo-explore/exo/9-benchmarking-and-performance)
- [exo\_bench Benchmarking Tool](https://deepwiki.com/exo-explore/exo/9.1-exo_bench-benchmarking-tool)
- [Performance Metrics](https://deepwiki.com/exo-explore/exo/9.2-performance-metrics)
- [Development](https://deepwiki.com/exo-explore/exo/10-development)
- [Development Environment Setup](https://deepwiki.com/exo-explore/exo/10.1-development-environment-setup)
- [Build System and Release Process](https://deepwiki.com/exo-explore/exo/10.2-build-system-and-release-process)
- [Testing Infrastructure](https://deepwiki.com/exo-explore/exo/10.3-testing-infrastructure)
- [Contributing Guidelines](https://deepwiki.com/exo-explore/exo/10.4-contributing-guidelines)
- [Glossary](https://deepwiki.com/exo-explore/exo/11-glossary)

Menu

# Topology Discovery

Relevant source files

- [Cargo.lock](https://github.com/exo-explore/exo/blob/1e51dc89/Cargo.lock)
- [Cargo.toml](https://github.com/exo-explore/exo/blob/1e51dc89/Cargo.toml)
- [rust/exo\_pyo3\_bindings/Cargo.toml](https://github.com/exo-explore/exo/blob/1e51dc89/rust/exo_pyo3_bindings/Cargo.toml)
- [rust/exo\_pyo3\_bindings/exo\_pyo3\_bindings.pyi](https://github.com/exo-explore/exo/blob/1e51dc89/rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi)
- [rust/exo\_pyo3\_bindings/pyproject.toml](https://github.com/exo-explore/exo/blob/1e51dc89/rust/exo_pyo3_bindings/pyproject.toml)
- [rust/exo\_pyo3\_bindings/src/allow\_threading.rs](https://github.com/exo-explore/exo/blob/1e51dc89/rust/exo_pyo3_bindings/src/allow_threading.rs)
- [rust/exo\_pyo3\_bindings/src/lib.rs](https://github.com/exo-explore/exo/blob/1e51dc89/rust/exo_pyo3_bindings/src/lib.rs)
- [rust/exo\_pyo3\_bindings/src/networking.rs](https://github.com/exo-explore/exo/blob/1e51dc89/rust/exo_pyo3_bindings/src/networking.rs)
- [rust/networking/Cargo.toml](https://github.com/exo-explore/exo/blob/1e51dc89/rust/networking/Cargo.toml)
- [rust/networking/examples/chatroom.rs](https://github.com/exo-explore/exo/blob/1e51dc89/rust/networking/examples/chatroom.rs)
- [rust/networking/src/discovery.rs](https://github.com/exo-explore/exo/blob/1e51dc89/rust/networking/src/discovery.rs)
- [rust/networking/src/swarm.rs](https://github.com/exo-explore/exo/blob/1e51dc89/rust/networking/src/swarm.rs)
- [rust/networking/tests/bootstrap\_peers.rs](https://github.com/exo-explore/exo/blob/1e51dc89/rust/networking/tests/bootstrap_peers.rs)
- [src/exo/routing/connection\_message.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/routing/connection_message.py)
- [src/exo/routing/router.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/routing/router.py)
- [src/exo/shared/topology.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py)
- [src/exo/shared/types/profiling.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/profiling.py)
- [src/exo/shared/types/state.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/state.py)
- [src/exo/utils/info\_gatherer/info\_gatherer.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/info_gatherer.py)
- [src/exo/utils/info\_gatherer/system\_info.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/system_info.py)

## Overview

Topology Discovery is the process by which exo nodes discover each other and build a directed graph representation of the cluster's network structure. The discovery process identifies available nodes, tests network connectivity between them, detects connection types (standard TCP/IP vs RDMA-capable Thunderbolt), and constructs cycles that can be used for distributed model execution.

The topology discovery system:

- Discovers peer nodes via libp2p advertising and mDNS.
- Tests network reachability between nodes.
- Builds a directed graph using `rustworkx.PyDiGraph`.
- Detects RDMA-capable Thunderbolt connections via `ThunderboltBridgeService` and `rdma_ctl`.
- Finds cycles for pipeline and tensor parallelism placement.
- Tracks node capabilities and connection characteristics in a `TopologySnapshot`.

Sources: [src/exo/shared/topology.py19-213](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py#L19-L213) [src/exo/utils/info\_gatherer/info\_gatherer.py45-180](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/info_gatherer.py#L45-L180)

## Core Data Structures

### Topology Class

The `Topology` class wraps a `rustworkx.PyDiGraph` to represent the cluster as a directed graph. It maintains bidirectional mappings between `NodeId` strings and internal graph indices for efficient lookups.

**Graph Structure Diagram:**

```
Topology [src/exo/shared/topology.py]

indexed by

reverse lookup

edge tracking

rustworkx.PyDiGraph[NodeInfo, Connection]

_node_id_to_rx_id_map: dict[NodeId, int]

_rx_id_to_node_id_map: dict[int, NodeId]

_edge_id_to_rx_id_map: dict[Connection, int]
```

**Key Operations:**

| Operation | Description | Source |
| --- | --- | --- |
| `add_node(NodeInfo)` | Add node to graph with performance profile | [src/exo/shared/topology.py45-50](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py#L45-L50) |
| `add_connection(Connection)` | Add directed edge between nodes | [src/exo/shared/topology.py80-96](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py#L80-L96) |
| `remove_node(NodeId)` | Remove node and all its connections | [src/exo/shared/topology.py130-145](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py#L130-L145) |
| `get_cycles()` | Find all simple cycles in the graph | [src/exo/shared/topology.py154-161](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py#L154-L161) |
| `get_cycles_tb()` | Find cycles using only Thunderbolt connections | [src/exo/shared/topology.py163-182](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py#L163-L182) |
| `node_is_leaf(NodeId)` | Check if node has exactly one outgoing neighbor | [src/exo/shared/topology.py52-56](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py#L52-L56) |

Sources: [src/exo/shared/topology.py19-213](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py#L19-L213)

### State Serialization

The `State` class handles the persistence of the cluster topology. It uses a `TopologySnapshot` to serialize the `rustworkx` graph into a format suitable for the event log.

```
# src/exo/shared/types/state.py:64-84
@field_serializer("topology", mode="plain")
def _encode_topology(self, value: Topology) -> TopologySnapshot:
    return value.to_snapshot()

@field_validator("topology", mode="before")
@classmethod
def _deserialize_topology(cls, value: object) -> Topology:
    if isinstance(value, Mapping):
        snapshot = TopologySnapshot(**value)
        return Topology.from_snapshot(snapshot)
    return value
```

Sources: [src/exo/shared/types/state.py64-84](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/state.py#L64-L84) [src/exo/shared/topology.py12-43](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py#L12-L43)

## Peer Discovery via libp2p

Exo uses a Rust-based networking stack (via `exo_pyo3_bindings`) to handle peer discovery. The `Router` class manages `GossipSub` topics and receives peer events from the libp2p swarm.

**Discovery Flow:**

```
"Node Orchestrator""Router [src/exo/routing/router.py]""NetworkingHandle [rust/exo_pyo3_bindings/src/networking.rs]""libp2p Swarm (Rust)""Node Orchestrator""Router [src/exo/routing/router.py]""NetworkingHandle [rust/exo_pyo3_bindings/src/networking.rs]""libp2p Swarm (Rust)""Trigger TopologyEdgeCreated event""FromSwarm::Discovered { peer_id }""PyFromSwarm.Connection(peer_id, connected=True)""CONNECTION_MESSAGES topic update"
```

The `NetworkingHandle` in Rust translates libp2p swarm events into `PyFromSwarm` objects for the Python layer.

Sources: [rust/exo\_pyo3\_bindings/src/networking.rs142-171](https://github.com/exo-explore/exo/blob/1e51dc89/rust/exo_pyo3_bindings/src/networking.rs#L142-L171) [src/exo/routing/router.py188-210](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/routing/router.py#L188-L210)

## Thunderbolt and RDMA Detection

Exo performs deep inspection of hardware to identify high-speed Thunderbolt bridges, which are used for RDMA-accelerated inference via the `mlx_jaccl` backend.

### ThunderboltBridgeService

On macOS, the `InfoGatherer` uses `networksetup` and `ifconfig` to map physical Thunderbolt ports to network bridge interfaces.

**Detection Logic:**

1. `_get_thunderbolt_devices()`: Parses `networksetup -listallhardwareports` to find devices labeled "thunderbolt" [src/exo/utils/info\_gatherer/info\_gatherer.py45-75](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/info_gatherer.py#L45-L75)
2. `_get_bridge_services()`: Maps bridge devices (e.g., `bridge0`) to service names [src/exo/utils/info\_gatherer/info\_gatherer.py78-122](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/info_gatherer.py#L78-L122)
3. `_find_thunderbolt_bridge()`: Checks `ifconfig` for bridge members to see if any are Thunderbolt devices [src/exo/utils/info\_gatherer/info\_gatherer.py147-158](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/info_gatherer.py#L147-L158)

### RDMA Control (`rdma_ctl`)

The system checks for RDMA capability by attempting to call the `rdma_ctl` utility. This status is captured in `RdmaCtlStatus` and stored in the global `State`.

```
# src/exo/utils/info_gatherer/info_gatherer.py:212-225
class RdmaCtlStatus(TaggedModel):
    enabled: bool

    @classmethod
    async def gather(cls) -> Self | None:
        if not IS_DARWIN or shutil.which("rdma_ctl") is None:
            return None
        # ... execution of rdma_ctl status ...
```

Sources: [src/exo/utils/info\_gatherer/info\_gatherer.py212-225](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/info_gatherer.py#L212-L225) [src/exo/shared/types/state.py59](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/state.py#L59-L59)

## Cycle Detection for Placement

The placement engine relies on `Topology.get_cycles()` to identify groups of nodes that can form a closed loop for pipeline parallelism.

**Cycle Filtering:**

1. **Memory Filtering**: `filter_cycles_by_memory` removes cycles where the aggregate memory is less than the model requirements.
2. **Thunderbolt Preference**: `get_cycles_tb()` identifies cycles where every edge is a Thunderbolt connection [src/exo/shared/topology.py163-182](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py#L163-L182)
3. **Smallest Cycle**: `get_smallest_cycles` prioritizes cycles with the fewest nodes to minimize communication overhead.

**Thunderbolt Cycle Validation:**

```
Thunderbolt (169.254.x.x)

Thunderbolt (169.254.x.x)

Thunderbolt (169.254.x.x)

is_thunderbolt_cycle check

Check Connection.is_thunderbolt() for all edges

Node A

Node B

Node C
```

Sources: [src/exo/shared/topology.py163-182](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/topology.py#L163-L182) [src/exo/shared/types/topology.py28-31](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/topology.py#L28-L31)

## Reachability and Networking Profiles

Before adding a node to the active topology for placement, exo may verify reachability using `net_profile`.

**NodeNetworkInfo**:
Each node reports its available interfaces, including IP addresses and MTU settings. This is stored in `State.node_network`.

```
# src/exo/shared/types/state.py:56-56
node_network: Mapping[NodeId, NodeNetworkInfo] = {}
```

The system uses these profiles to determine the `send_back_multiaddr` for a `Connection`, ensuring that the master can instruct workers on which specific IP to use for inter-node communication.

Sources: [src/exo/shared/types/state.py51-60](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/state.py#L51-L60) [src/exo/shared/types/topology.py17-26](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/topology.py#L17-L26)

Dismiss

Refresh this wiki

Enter email to refresh

### On this page

- [Topology Discovery](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery#topology-discovery)
- [Overview](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery#overview)
- [Core Data Structures](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery#core-data-structures)
- [Topology Class](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery#topology-class)
- [State Serialization](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery#state-serialization)
- [Peer Discovery via libp2p](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery#peer-discovery-via-libp2p)
- [Thunderbolt and RDMA Detection](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery#thunderbolt-and-rdma-detection)
- [ThunderboltBridgeService](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery#thunderboltbridgeservice)
- [RDMA Control (\`rdma\_ctl\`)](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery#rdma-control-rdma_ctl)
- [Cycle Detection for Placement](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery#cycle-detection-for-placement)
- [Reachability and Networking Profiles](https://deepwiki.com/exo-explore/exo/6.2-topology-discovery#reachability-and-networking-profiles)

Ask Devin about exo-explore/exo

Fast