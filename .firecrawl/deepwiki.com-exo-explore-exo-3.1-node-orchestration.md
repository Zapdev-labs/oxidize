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

# Node Orchestration

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

The `Node` class in `src/exo/main.py` serves as the central orchestrator for an exo instance. It manages the lifecycle of all major system components: `Router`, `Master`, `Worker`, `API`, `Election`, and `DownloadCoordinator`. The `Node` is responsible for initializing these components, wiring their communication via topic-based messaging, and managing dynamic role transitions (promoting or demoting the local `Master` instance) based on cluster-wide election results.

For information about the event sourcing architecture that nodes participate in, see [Event Sourcing and State Management](https://deepwiki.com/exo-explore/exo/3.2-event-sourcing-and-state-management). For details about Master and Worker responsibilities, see [Master and Worker Roles](https://deepwiki.com/exo-explore/exo/3.3-master-and-worker-roles). For election protocol specifics, see [Election and Failover](https://deepwiki.com/exo-explore/exo/3.4-election-and-failover).

## The Node Class Structure

The `Node` is a dataclass that composes the primary functional units of the system. While every node participates in the `Election` and `Router` systems, other components like the `Worker` or `API` can be disabled via configuration.

**Sources:** [src/exo/main.py30-43](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L30-L43)

Node Orchestration Component Map:

```
Functional Roles

Core Infrastructure

Node
(src/exo/main.py)

router: Router
(src/exo/routing/router.py)

event_router: EventRouter
(src/exo/routing/event_router.py)

download_coordinator:
DownloadCoordinator | None

worker: Worker | None

election: Election

master: Master | None

api: API | None

election_result_receiver:
Receiver[ElectionResult]
```

**Component Roles and Optionality:**

| Component | Required | Controlled By | Purpose |
| --- | --- | --- | --- |
| `router` | Yes | Always | Low-level libp2p transport and topic management [src/exo/main.py50-54](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L50-L54) |
| `event_router` | Yes | Always | Manages `OrderedBuffer` and session-based event sequencing [src/exo/main.py61-66](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L61-L66) |
| `election` | Yes | Always | Implements the Bully-style leader election protocol [src/exo/main.py117-128](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L117-L128) |
| `download_coordinator` | No | `--no-downloads` | Manages HF downloads and shard verification [src/exo/main.py71-78](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L71-L78) |
| `worker` | No | `--no-worker` | Manages local `RunnerSupervisor` and inference execution [src/exo/main.py94-103](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L94-L103) |
| `master` | No | Election Result | Processes `Command` objects into `IndexedEvent` logs [src/exo/main.py106-114](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L106-L114) |
| `api` | No | `--spawn-api` | FastAPI server for OpenAI/Ollama compatibility [src/exo/main.py82-92](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L82-L92) |

**Sources:** [src/exo/main.py31-42](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L31-L42) [src/exo/main.py46-141](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L46-L141)

## Component Initialization and Topic Registration

The `Node.create()` method performs the wiring of the system. It registers the node on specific libp2p topics and establishes the data flow between components using `Sender` and `Receiver` channels.

### Topic-Based Messaging Backbone

The `Router` registers six core topics that define the cluster's communication surface:

| Topic Name | Code Symbol | Description |
| --- | --- | --- |
| `GLOBAL_EVENTS` | `topics.GLOBAL_EVENTS` | Master-indexed events broadcast to all nodes [src/exo/main.py55](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L55-L55) |
| `LOCAL_EVENTS` | `topics.LOCAL_EVENTS` | Un-indexed events sent from Workers to the Master [src/exo/main.py56](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L56-L56) |
| `COMMANDS` | `topics.COMMANDS` | External requests (inference, placement) sent to the Master [src/exo/main.py57](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L57-L57) |
| `ELECTION_MESSAGES` | `topics.ELECTION_MESSAGES` | Protocol messages for the `Election` component [src/exo/main.py58](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L58-L58) |
| `CONNECTION_MESSAGES` | `topics.CONNECTION_MESSAGES` | Peer discovery and topology updates [src/exo/main.py59](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L59-L59) |
| `DOWNLOAD_COMMANDS` | `topics.DOWNLOAD_COMMANDS` | Commands targeting the `DownloadCoordinator` [src/exo/main.py60](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L60-L60) |

**Sources:** [src/exo/main.py55-60](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L55-L60) [src/exo/routing/topics.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/routing/topics.py)

### Component Wiring Diagram

This diagram maps the data flow between the orchestrator's components and the underlying messaging topics.

```
Node Components

External/Network

send(TextGeneration)

send(NodeGatheredInfo)

receive()

receive()

send(IndexedEvent)

receive()

yield(IndexedEvent)

yield(IndexedEvent)

send(StartDownload)

receive()

GLOBAL_EVENTS

LOCAL_EVENTS

COMMANDS

DOWNLOAD_COMMANDS

API

Worker

Master

DownloadCoordinator

EventRouter
```

**Sources:** [src/exo/main.py61-114](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L61-L114) [src/exo/master/main.py72-83](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/main.py#L72-L83) [src/exo/worker/main.py54-65](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/main.py#L54-L65)

## Component Lifecycle and `_elect_loop`

The `Node.run()` method starts all components concurrently within an `anyio.TaskGroup` [src/exo/main.py144-158](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L144-L158) A critical part of this lifecycle is the `_elect_loop`, which reacts to `ElectionResult` messages to manage the node's role in the cluster.

### Election Lifecycle Flow

The `_elect_loop` handles the transition of a node between "Follower" and "Master" states. When a new master is elected (indicated by `result.is_new_master`), the node must reset its internal state to align with the new session.

```
WorkerMasterEventRouterNode._elect_loopElectionWorkerMasterEventRouterNode._elect_loopElectionNew Master detectedI am the new MasterOther node is Masteralt[result.master_node_id == self.node_id]ElectionResult(is_new_master=True, session_id=S2)shutdown()Re-create EventRouter(S2)Start Master componentshutdown() (if running)shutdown()Re-create Worker(S2)start_soon(worker.run)
```

**Sources:** [src/exo/main.py168-249](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L168-L249) [src/exo/shared/election.py42-46](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/election.py#L42-L46)

### Key Lifecycle Behaviors:

1. **Event Consistency:** On a new master election, the `EventRouter` is replaced to ensure the `OrderedBuffer` is cleared and ready for a new sequence of indices starting from the new `SessionId` [src/exo/main.py185-194](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L185-L194)
2. **Master Promotion/Demotion:** If the local node is elected master, it instantiates the `Master` component [src/exo/main.py196-200](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L196-L200) If another node wins, any existing local `Master` is shut down [src/exo/main.py202-206](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L202-L206)
3. **Worker Reset:** The `Worker` is always re-created on a new master election to ensure it synchronizes its local `State` against the new master's event log [src/exo/main.py218-228](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L218-L228)
4. **API Synchronization:** The `API` is reset or unpaused depending on whether the master is entirely new or if the current master simply renewed its lease [src/exo/main.py244-248](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L244-L248)

**Sources:** [src/exo/main.py183-249](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L183-L249)

## Graceful Shutdown

The `Node` manages a clean exit by capturing `SIGINT` and `SIGTERM` signals. The `shutdown()` method triggers the cancellation of the primary `TaskGroup`, which propagates to all sub-components [src/exo/main.py145-146](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L145-L146)

- **Worker Shutdown:** Closes its event and command senders and shuts down all managed `RunnerSupervisor` processes [src/exo/worker/main.py98-106](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/main.py#L98-L106)
- **Master Shutdown:** Closes the `DiskEventLog` and all event/command channels [src/exo/master/main.py108-113](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/main.py#L108-L113)
- **DownloadCoordinator Shutdown:** Cancels all active download tasks [src/exo/download/coordinator.py137-139](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/download/coordinator.py#L137-L139)

**Sources:** [src/exo/main.py160-167](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/main.py#L160-L167) [src/exo/worker/main.py98-107](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/main.py#L98-L107) [src/exo/master/main.py108-113](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/main.py#L108-L113)

Dismiss

Refresh this wiki

Enter email to refresh

### On this page

- [Node Orchestration](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration#node-orchestration)
- [Purpose and Scope](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration#purpose-and-scope)
- [The Node Class Structure](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration#the-node-class-structure)
- [Component Initialization and Topic Registration](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration#component-initialization-and-topic-registration)
- [Topic-Based Messaging Backbone](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration#topic-based-messaging-backbone)
- [Component Wiring Diagram](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration#component-wiring-diagram)
- [Component Lifecycle and \`\_elect\_loop\`](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration#component-lifecycle-and-_elect_loop)
- [Election Lifecycle Flow](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration#election-lifecycle-flow)
- [Key Lifecycle Behaviors:](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration#key-lifecycle-behaviors)
- [Graceful Shutdown](https://deepwiki.com/exo-explore/exo/3.1-node-orchestration#graceful-shutdown)

Ask Devin about exo-explore/exo

Fast