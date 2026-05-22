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

# Election and Failover

Relevant source files

- [src/exo/download/coordinator.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/download/coordinator.py)
- [src/exo/main.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py)
- [src/exo/master/main.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/main.py)
- [src/exo/master/tests/test\_master.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/tests/test_master.py)
- [src/exo/shared/apply.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/apply.py)
- [src/exo/shared/election.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py)
- [src/exo/shared/tests/test\_election.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/tests/test_election.py)
- [src/exo/shared/types/commands.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/commands.py)
- [src/exo/shared/types/events.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/events.py)
- [src/exo/shared/types/tasks.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/tasks.py)
- [src/exo/worker/main.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/main.py)

## Purpose and Scope

This page explains the election protocol used to select a master node and the failover mechanisms that maintain cluster availability when nodes disconnect or fail. The elected master coordinates the cluster by indexing events and performing instance placement decisions. For details on node orchestration, see [Node Orchestration](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration). For state synchronization across nodes, see [Event Sourcing and State Management](https://deepwiki.com/exo-explore/exo/3.2-event-sourcing-and-state-management).

* * *

## Election Protocol Overview

The EXO cluster uses a **seniority-based election protocol** to automatically select a master node from all connected peers. Every node participates in elections, as any node should be able to become a master if no dedicated master candidates are present [src/exo/main.py36-38](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L36-L38) Elections occur when:

- A new node joins the cluster or an existing node disconnects (via `ConnectionMessage`) [src/exo/shared/election.py160-172](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L160-L172)
- An existing node receives an `ElectionMessage` with a higher clock [src/exo/shared/election.py136-150](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L136-L150)
- The node initially starts up [src/exo/shared/election.py94-98](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L94-L98)

The protocol ensures all nodes converge on the same master selection within a bounded time window defined by `DEFAULT_ELECTION_TIMEOUT` (3 seconds) [src/exo/shared/election.py18](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L18-L18)

* * *

## Election Algorithm Components

### Election Message Structure

Each node broadcasts an `ElectionMessage` during an election round containing metadata used for comparison:

| Field | Type | Description |
| --- | --- | --- |
| `clock` | int | Round number; higher clock values trigger new elections [src/exo/shared/election.py22](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L22-L22) |
| `seniority` | int | Priority level; higher for nodes that have won before [src/exo/shared/election.py23](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L23-L23) |
| `proposed_session` | SessionId | Proposed master (`node_id` \+ `election_clock`) [src/exo/shared/election.py24](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L24-L24) |
| `commands_seen` | int | Total commands processed by this node (tiebreaker) [src/exo/shared/election.py25](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L25-L25) |

**Sources:** [src/exo/shared/election.py21-25](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L21-L25)

### Winner Selection Logic

The protocol selects a winner by comparing `ElectionMessage` instances using the `__lt__` operator. The comparison logic follows this priority:

1. **Clock**: Higher clock always wins [src/exo/shared/election.py29-30](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L29-L30)
2. **Seniority**: Higher seniority wins [src/exo/shared/election.py31-32](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L31-L32)
3. **Commands Seen**: Node that has seen more commands wins [src/exo/shared/election.py33-34](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L33-L34)
4. **Node ID**: Lexicographical comparison of `node_id` as a final tiebreaker [src/exo/shared/election.py36-39](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L36-L39)

```
def __lt__(self, other: Self) -> bool:
    if self.clock != other.clock:
        return self.clock < other.clock
    if self.seniority != other.seniority:
        return self.seniority < other.seniority
    elif self.commands_seen != other.commands_seen:
        return self.commands_seen < other.commands_seen
    else:
        return (
            self.proposed_session.master_node_id
            < other.proposed_session.master_node_id
        )
```

**Sources:** [src/exo/shared/election.py28-40](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L28-L40)

* * *

## Election State Machine

### Election Class Architecture

The `Election` class manages the lifecycle of master selection by listening to specific topics via the `Router`.

```

```

**Sources:** [src/exo/shared/election.py48-84](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L48-L84) [src/exo/main.py117-128](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L117-L128)

* * *

## Election Triggers and Flow

### Trigger 1: Topology Changes

When the `Router` detects a new connection or a disconnection, it emits a `ConnectionMessage`. The `Election` component listens for these, waits briefly (0.2s) for network symmetry, increments the local clock, and starts a new campaign [src/exo/shared/election.py160-176](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L160-L176)

### Trigger 2: Higher Clock Reception

If an `ElectionMessage` arrives with `message.clock > self.clock`, the node updates its clock to match and joins the new round by starting its own campaign [src/exo/shared/election.py136-150](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L136-L150)

* * *

## Campaign Execution

### The \_campaign Lifecycle

A campaign is a time-limited period where a node collects votes (`ElectionMessage`s) from peers.

1. **Broadcast**: The node sends its own `ElectionMessage` to the `ELECTION_MESSAGES` topic [src/exo/shared/election.py214-222](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L214-L222)
2. **Wait**: It sleeps for `DEFAULT_ELECTION_TIMEOUT` (3s) while `_election_receiver` populates the `_candidates` list [src/exo/shared/election.py224-227](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L224-L227)
3. **Resolve**: After the timeout, it calculates `elected = max(candidates)` [src/exo/shared/election.py230](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L230-L230)
4. **Update Seniority**: If the node won, it sets its seniority to the number of participants in the election [src/exo/shared/election.py234-241](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L234-L241)
5. **Emit Result**: An `ElectionResult` is sent to the `Node` orchestrator [src/exo/shared/election.py246-252](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L246-L252)

**Campaign Cancellation**: If a new campaign is triggered (higher clock) while one is running, the old campaign is cancelled using an `anyio.CancelScope` [src/exo/shared/election.py202-211](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L202-L211)

**Sources:** [src/exo/shared/election.py195-261](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L195-L261)

* * *

## Seniority Accumulation

### Seniority Update Rules

Seniority is a persistent value that rewards nodes for staying in the cluster.

- **Initial**: Nodes start with seniority 0. If `--force-master` is used, seniority is set to 1,000,000 [src/exo/main.py120](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L120-L120)
- **Non-Candidates**: Nodes can be initialized with `is_candidate=False`, setting seniority to -1 [src/exo/shared/election.py64](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L64-L64)
- **Winning**: On winning an election, `seniority = max(current_seniority, len(participants))` [src/exo/shared/election.py234-241](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L234-L241)

* * *

## Failover and Role Transitions

### Node Role Management

When a node receives an `ElectionResult` indicating a master change (`is_new_master=True`), it must reset its internal components to align with the new cluster state [src/exo/main.py183-186](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L183-L186)

```

```

### SessionId Invalidation

The `SessionId` contains the `master_node_id` and the `election_clock` [src/exo/shared/types/common.py49](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/common.py#L49-L49) When a new master is elected, the `election_clock` increments. This invalidates existing sessions in the `EventRouter`, preventing stale events from an old master from being processed [src/exo/main.py185-200](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L185-L200)

**Sources:** [src/exo/main.py183-205](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L183-L205) [src/exo/shared/election.py42-46](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L42-L46)

* * *

## Failover Mechanisms

### Runner and Task Recovery

Failover is not just about the Master. The `Worker` also monitors the health of runners. If a node detects a failure in its own runner or a peer runner within the same instance, the `plan()` function triggers a `Shutdown` task for that instance [src/exo/worker/plan.py57-78](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/plan.py#L57-L78)

1. **Worker Plan**: `plan()` checks `state.runners` for `RunnerFailed` status [src/exo/worker/main.py147-156](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/main.py#L147-L156)
2. **Cleanup**: The `Worker` issues a `Shutdown` task to stop the `RunnerSupervisor` [src/exo/worker/main.py218-223](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/main.py#L218-L223)
3. **Master Re-planning**: Once the `InstanceDeleted` event is indexed by the Master [src/exo/shared/apply.py193-198](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/apply.py#L193-L198) the Master's `_plan` loop will eventually re-run `place_instance` to recover the model on available nodes [src/exo/master/main.py404-410](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/main.py#L404-L410)

**Sources:** [src/exo/worker/main.py144-223](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/main.py#L144-L223) [src/exo/shared/apply.py193-198](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/apply.py#L193-L198) [src/exo/master/main.py404-410](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/main.py#L404-L410)

* * *

## Implementation Summary

### Key Files and Classes

| Class | File | Responsibility |
| --- | --- | --- |
| `Election` | [src/exo/shared/election.py48](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L48-L48) | Implements the seniority-based voting logic. |
| `ElectionMessage` | [src/exo/shared/election.py21](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L21-L21) | Data structure for votes with comparison logic. |
| `Node` | [src/exo/main.py31](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L31-L31) | Handles component restarts on `ElectionResult`. |
| `Worker` | [src/exo/worker/main.py54](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/main.py#L54-L54) | Monitors runner health and performs local cleanup. |
| `Master` | [src/exo/master/main.py72](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/main.py#L72-L72) | Re-plans model placement after node/runner failure. |

**Sources:** [src/exo/shared/election.py1-274](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L1-L274) [src/exo/main.py1-210](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L1-L210) [src/exo/worker/main.py1-230](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/main.py#L1-L230)

Dismiss

Refresh this wiki

Enter email to refresh

### On this page

- [Election and Failover](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#election-and-failover)
- [Purpose and Scope](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#purpose-and-scope)
- [Election Protocol Overview](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#election-protocol-overview)
- [Election Algorithm Components](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#election-algorithm-components)
- [Election Message Structure](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#election-message-structure)
- [Winner Selection Logic](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#winner-selection-logic)
- [Election State Machine](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#election-state-machine)
- [Election Class Architecture](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#election-class-architecture)
- [Election Triggers and Flow](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#election-triggers-and-flow)
- [Trigger 1: Topology Changes](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#trigger-1-topology-changes)
- [Trigger 2: Higher Clock Reception](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#trigger-2-higher-clock-reception)
- [Campaign Execution](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#campaign-execution)
- [The \_campaign Lifecycle](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#the-_campaign-lifecycle)
- [Seniority Accumulation](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#seniority-accumulation)
- [Seniority Update Rules](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#seniority-update-rules)
- [Failover and Role Transitions](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#failover-and-role-transitions)
- [Node Role Management](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#node-role-management)
- [SessionId Invalidation](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#sessionid-invalidation)
- [Failover Mechanisms](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#failover-mechanisms)
- [Runner and Task Recovery](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#runner-and-task-recovery)
- [Implementation Summary](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#implementation-summary)
- [Key Files and Classes](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover#key-files-and-classes)

Ask Devin about exo-explore/exo

Fast