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

# Communication Backends

Relevant source files

- [src/exo/master/placement.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement.py)
- [src/exo/master/placement\_utils.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement_utils.py)
- [src/exo/master/tests/conftest.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/tests/conftest.py)
- [src/exo/master/tests/test\_placement.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/tests/test_placement.py)
- [src/exo/master/tests/test\_placement\_utils.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/tests/test_placement_utils.py)
- [src/exo/master/tests/test\_topology.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/tests/test_topology.py)
- [src/exo/shared/types/chunks.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/chunks.py)
- [src/exo/shared/types/profiling.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/profiling.py)
- [src/exo/shared/types/state.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/state.py)
- [src/exo/shared/types/worker/instances.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/worker/instances.py)
- [src/exo/shared/types/worker/runner\_response.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/worker/runner_response.py)
- [src/exo/utils/channels.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/channels.py)
- [src/exo/utils/info\_gatherer/info\_gatherer.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/info_gatherer.py)
- [src/exo/utils/info\_gatherer/net\_profile.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/net_profile.py)
- [src/exo/utils/info\_gatherer/system\_info.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/system_info.py)
- [src/exo/worker/engines/mlx/generator/generate.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/generator/generate.py)
- [src/exo/worker/engines/mlx/utils\_mlx.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/utils_mlx.py)
- [src/exo/worker/plan.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/plan.py)
- [src/exo/worker/runner/bootstrap.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/runner/bootstrap.py)
- [src/exo/worker/runner/runner\_supervisor.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/runner/runner_supervisor.py)
- [src/exo/worker/tests/unittests/test\_runner/test\_event\_ordering.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/tests/unittests/test_runner/test_event_ordering.py)

## Purpose and Scope

This document describes the communication backends used by `exo` to enable data transfer between nodes during distributed inference. Communication backends handle the low-level transport of activations between model shards running on different nodes.

For information about how models are partitioned across nodes, see [Sharding Strategies](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies). For details on the inference execution flow, see [Inference Execution](https://deepwiki.com/exo-explore/exo/5-inference-execution).

* * *

## Overview

Exo supports two communication backends for distributed inference, both leveraging the `mlx.distributed` framework. The backend selection is handled by the `Master` during the placement phase in `src/exo/master/placement.py`.

| Backend | Transport | Use Case | Code Entity |
| --- | --- | --- | --- |
| **MLX Ring** | TCP/IP via `MLX_HOSTFILE` | Standard network connectivity (WiFi/Ethernet/Thunderbolt) | `MlxRingInstance` |
| **MLX Jaccl** | RDMA via `MLX_JACCL_COORDINATOR` | Low-latency clusters with RDMA hardware (e.g., RoCE) | `MlxJacclInstance` |

**Sources:** [src/exo/master/placement.py41-48](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement.py#L41-L48) [src/exo/worker/engines/mlx/utils\_mlx.py108-146](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/utils_mlx.py#L108-L146)

* * *

## Backend Type Hierarchy

The communication backend configuration is encapsulated within `Instance` subclasses. These objects are generated by the `Master` and sent to `Workers` to initialize the distributed environment.

```

«abstract»

Instance

+InstanceId instance_id

+ShardAssignments shard_assignments

MlxRingInstance

+dict hosts_by_node

+int ephemeral_port

MlxJacclInstance

+list jaccl_devices

+dict jaccl_coordinators

ShardAssignments

+ModelId model_id

+dict node_to_runner

+dict runner_to_shard
```

**Sources:** [src/exo/shared/types/worker/instances.py19-51](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/worker/instances.py#L19-L51) [src/exo/master/placement.py41-47](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement.py#L41-L47)

* * *

## Initialization Flow

When a `Runner` starts, it calls `initialize_mlx()` which triggers the distributed setup. This process translates the high-level `Instance` configuration into environment variables required by the MLX C++ runtime.

```
Backend Branching

No

Yes

MlxRingInstance

MlxJacclInstance

initialize_mlx(bound_instance)

Is world_size > 1?

Skip Distributed Init

mlx_distributed_init()

Instance Type?

Create MLX_HOSTFILE (JSON)

Set MLX_RANK & MLX_RING_VERBOSE

mx.distributed.init(backend='ring')

Create MLX_IBV_DEVICES (JSON)

Set MLX_JACCL_COORDINATOR

mx.distributed.init(backend='jaccl')

Return Group
```

**Sources:** [src/exo/worker/engines/mlx/utils\_mlx.py94-163](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/utils_mlx.py#L94-L163)

### MLX Ring (TCP/IP) Implementation

The Ring backend utilizes a "hostfile" to define the topology. `exo` dynamically generates a temporary JSON file for each rank containing the list of IPs it should connect to or listen on.

- **Key Functions**: `get_mlx_ring_hosts_by_node` in `placement_utils.py` generates the IP lists.
- **Data Flow**:

1. `Master` identifies a cycle in the `Topology`.
2. For each node, it calculates neighbors (rank-1, rank+1).
3. It builds a `HostList` where the node's own rank is `0.0.0.0` and neighbors are their specific cluster IPs.
4. `Worker` writes this to a `tmpdir` and sets `os.environ["MLX_HOSTFILE"]`.

**Sources:** [src/exo/worker/engines/mlx/utils\_mlx.py109-123](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/utils_mlx.py#L109-L123) [src/exo/master/placement\_utils.py270-318](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement_utils.py#L270-L318)

### MLX Jaccl (RDMA) Implementation

Jaccl (Just Another Collective Communication Library) is the RDMA-enabled backend. It requires an all-to-all connectivity matrix.

- **Configuration**:

  - `MLX_IBV_DEVICES`: A matrix where `matrix[i][j]` is the name of the InfiniBand/RDMA device on node `i` used to reach node `j`.
  - `MLX_JACCL_COORDINATOR`: The IP:Port of rank 0, used for initial handshake.
- **rdma\_ctl integration**: The `InfoGatherer` detects RDMA capability by checking for the `rdma_ctl` binary on macOS.

**Sources:** [src/exo/worker/engines/mlx/utils\_mlx.py125-146](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/utils_mlx.py#L125-L146) [src/exo/utils/info\_gatherer/info\_gatherer.py212-225](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/info_gatherer.py#L212-L225)

* * *

## Topology and Connectivity Detection

The `Master` uses the `Topology` graph to decide which backend is viable.

### Thunderbolt Topology Detection

Exo specifically detects Thunderbolt bridges on macOS to prioritize high-bandwidth connections. The `InfoGatherer` parses `networksetup` and `ifconfig` to identify `bridge` interfaces containing Thunderbolt members.

```
ifconfignetworksetupInfoGathererifconfignetworksetupInfoGathererThunderbolt Bridge Detected-listallhardwareportsHardware Port: Thunderbolt 1, Device: en2-listnetworkserviceorder(1) Thunderbolt Bridge, Device: bridge0bridge0member: en2
```

**Sources:** [src/exo/utils/info\_gatherer/info\_gatherer.py45-158](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/info_gatherer.py#L45-L158)

### IP Prioritization

When building hostfiles, `exo` prioritizes network interfaces to ensure the fastest path is used:

1. Ethernet
2. WiFi
3. Thunderbolt (often treated as a bridge)

**Sources:** [src/exo/master/placement\_utils.py243-268](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement_utils.py#L243-L268)

* * *

## Distributed Execution Mechanisms

Once the backend is initialized, `exo` employs several techniques to optimize data flow:

### Staggered Pipeline Prefill

To prevent network congestion and maximize GPU utilization during the prefill phase of pipeline parallelism, `exo` uses `pipeline_parallel_prefill`. This staggered execution allows ranks to overlap computation and communication.

```
Time T1

Rank 0: Chunk 1

Rank 1: Chunk 0

Time T0

Rank 0: Chunk 0

Rank 1: Idle
```

**Sources:** [src/exo/worker/engines/mlx/generator/generate.py76-140](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/generator/generate.py#L76-L140)

### Barriers and Synchronization

Distributed coordination is maintained using `mx_barrier(group)`, ensuring all nodes reach the same execution point before proceeding to sensitive operations like model loading or KV cache rotation.

**Sources:** [src/exo/worker/engines/mlx/utils\_mlx.py236-241](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/utils_mlx.py#L236-L241) [src/exo/worker/engines/mlx/generator/generate.py55-57](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/generator/generate.py#L55-L57)

* * *

## Error Handling and Failover

The communication backends are monitored by the `RunnerSupervisor`. If a distributed initialization fails (e.g., a node is unreachable), the following occurs:

1. **Timeout**: `initialize_mlx` is wrapped in `eval_with_timeout`.
2. **Failure Propagation**: The `Runner` emits a `RunnerStatusUpdated` with `RunnerFailed`.
3. **Master Re-planning**: The `Master` detects the failed runner in its `plan()` loop and issues a `Shutdown` command for the entire instance to prevent deadlocks.

**Sources:** [src/exo/worker/runner/runner\_supervisor.py198-210](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/runner/runner_supervisor.py#L198-L210) [src/exo/worker/plan.py68-89](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/plan.py#L68-L89) [src/exo/worker/engines/mlx/utils\_mlx.py166-210](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/utils_mlx.py#L166-L210)

Dismiss

Refresh this wiki

Enter email to refresh

### On this page

- [Communication Backends](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#communication-backends)
- [Purpose and Scope](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#purpose-and-scope)
- [Overview](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#overview)
- [Backend Type Hierarchy](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#backend-type-hierarchy)
- [Initialization Flow](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#initialization-flow)
- [MLX Ring (TCP/IP) Implementation](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#mlx-ring-tcpip-implementation)
- [MLX Jaccl (RDMA) Implementation](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#mlx-jaccl-rdma-implementation)
- [Topology and Connectivity Detection](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#topology-and-connectivity-detection)
- [Thunderbolt Topology Detection](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#thunderbolt-topology-detection)
- [IP Prioritization](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#ip-prioritization)
- [Distributed Execution Mechanisms](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#distributed-execution-mechanisms)
- [Staggered Pipeline Prefill](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#staggered-pipeline-prefill)
- [Barriers and Synchronization](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#barriers-and-synchronization)
- [Error Handling and Failover](https://deepwiki.com/exo-explore/exo/4.3-communication-backends#error-handling-and-failover)

Ask Devin about exo-explore/exo

Fast