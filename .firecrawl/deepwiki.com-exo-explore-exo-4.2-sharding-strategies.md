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

# Sharding Strategies

Relevant source files

- [.github/workflows/pipeline.yml](https://github.com/exo-explore/exo/blob/1e51dc89/.github/workflows/pipeline.yml)
- [bench/src/exo\_bench/\_\_init\_\_.py](https://github.com/exo-explore/exo/blob/1e51dc89/bench/src/exo_bench/__init__.py)
- [flake.lock](https://github.com/exo-explore/exo/blob/1e51dc89/flake.lock)
- [flake.nix](https://github.com/exo-explore/exo/blob/1e51dc89/flake.nix)
- [justfile](https://github.com/exo-explore/exo/blob/1e51dc89/justfile)
- [nix/mlx.nix](https://github.com/exo-explore/exo/blob/1e51dc89/nix/mlx.nix)
- [pyproject.toml](https://github.com/exo-explore/exo/blob/1e51dc89/pyproject.toml)
- [python/parts.nix](https://github.com/exo-explore/exo/blob/1e51dc89/python/parts.nix)
- [rust/parts.nix](https://github.com/exo-explore/exo/blob/1e51dc89/rust/parts.nix)
- [src/exo/master/placement.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement.py)
- [src/exo/master/placement\_utils.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement_utils.py)
- [src/exo/master/tests/conftest.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/tests/conftest.py)
- [src/exo/master/tests/test\_placement.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/tests/test_placement.py)
- [src/exo/master/tests/test\_placement\_utils.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/tests/test_placement_utils.py)
- [src/exo/master/tests/test\_topology.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/tests/test_topology.py)
- [src/exo/shared/models/model\_cards.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/models/model_cards.py)
- [src/exo/shared/types/worker/instances.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/worker/instances.py)
- [src/exo/utils/channels.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/channels.py)
- [src/exo/utils/info\_gatherer/net\_profile.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/utils/info_gatherer/net_profile.py)
- [src/exo/worker/engines/mlx/auto\_parallel.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/auto_parallel.py)
- [src/exo/worker/plan.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/plan.py)
- [src/exo/worker/runner/bootstrap.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/runner/bootstrap.py)
- [src/exo/worker/runner/runner\_supervisor.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/runner/runner_supervisor.py)
- [src/exo/worker/tests/unittests/test\_runner/test\_event\_ordering.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/tests/unittests/test_runner/test_event_ordering.py)
- [uv.lock](https://github.com/exo-explore/exo/blob/1e51dc89/uv.lock)

## Purpose and Scope

This document explains the sharding strategies used by exo to distribute model inference across multiple devices: **Pipeline Parallelism**, **Tensor Parallelism**, and **CFG Parallelism**. It covers how each strategy partitions model computation, the implementation mechanics, model-specific adaptations, and performance trade-offs.

For information about how the Placement Engine selects nodes and generates shard assignments, see [Placement Engine](https://deepwiki.com/exo-explore/exo/4.1-placement-engine). For details on the communication backends, see [Communication Backends](https://deepwiki.com/exo-explore/exo/4.3-communication-backends).

## Sharding Strategy Overview

Exo supports three fundamental approaches to distributing model computation, defined by the `Sharding` enum:

| Strategy | Partitioning Unit | Distribution Pattern | Communication Pattern |
| --- | --- | --- | --- |
| **Pipeline** | Contiguous layer ranges | Different layers per node | Sequential activation passing |
| **Tensor** | Weight tensors within each layer | All layers per node | All-reduce for gradients, all-gather for results |
| **CFG** | Prompt branches (Positive vs Negative) | Different branches per node | Parallel execution of diffusion branches |

**Sources:** [src/exo/shared/types/worker/shards.py10-14](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/worker/shards.py#L10-L14) [src/exo/master/placement.py130-156](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement.py#L130-L156) [src/exo/worker/engines/mlx/auto\_parallel.py150-186](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/auto_parallel.py#L150-L186)

## Pipeline Parallelism

### Concept

Pipeline parallelism divides a model's layers into contiguous ranges, assigning each range to a different node. Activations flow sequentially through the pipeline: Node 0 processes layers 0-N, passes activations to Node 1, which processes layers N+1-M, and so on.

```

mx.distributed.send

mx.distributed.send

mx.distributed.all_gather

Input Tokens

Node 0 (Rank 0)
Layers 0-10

Node 1 (Rank 1)
Layers 11-20

Node 2 (Rank 2)
Layers 21-31

Output Logits
```

**Diagram: Pipeline Parallelism Data Flow**

**Sources:** [src/exo/worker/engines/mlx/auto\_parallel.py150-186](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/auto_parallel.py#L150-L186)

### Layer Allocation

Layer ranges are allocated proportionally to each node's available memory using the `allocate_layers_proportionally` function. It calculates the fraction of total memory each node contributes and distributes layers accordingly, handling remainders to ensure the full model is covered.

**Sources:** [src/exo/master/placement\_utils.py46-74](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement_utils.py#L46-L74) [src/exo/master/placement\_utils.py77-144](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement_utils.py#L77-L144)

### Implementation

The `pipeline_auto_parallel()` function wraps the first and last layers of each node's shard with custom implementations:

- **`PipelineFirstLayer`**: Receives activations from the previous rank via `mx.distributed.recv_like()`. Rank 0 skips this and uses input data directly. [src/exo/worker/engines/mlx/auto\_parallel.py150-169](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/auto_parallel.py#L150-L169)
- **`PipelineLastLayer`**: Processes its assigned layers, then sends the resulting activations to the next rank via `mx.distributed.send()`. [src/exo/worker/engines/mlx/auto\_parallel.py173-186](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/auto_parallel.py#L173-L186)

#### PipelineShardMetadata

Each node receives a `PipelineShardMetadata` object specifying its layer range:

```
PipelineShardMetadata(
    device_rank=0,
    world_size=3,
    start_layer=0,
    end_layer=11,
    n_layers=32
)
```

**Sources:** [src/exo/shared/types/worker/shards.py17-23](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/worker/shards.py#L17-L23)

#### Deadlock Prevention

During prefill, `pipeline_parallel_prefill` manages staggered execution. To prevent GPU hangs, `eval_with_timeout` is used during MLX item evaluation. If an operation takes longer than the `timeout_seconds` (default 60s), it terminates the process to avoid cluster-wide deadlocks.

**Sources:** [src/exo/worker/engines/mlx/auto\_parallel.py85-116](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/auto_parallel.py#L85-L116) [src/exo/worker/runner/runner\_supervisor.py48-49](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/runner/runner_supervisor.py#L48-L49)

## Tensor Parallelism

### Concept

Tensor parallelism splits weight tensors within each layer across multiple nodes. All nodes process all layers, but each node only stores a fraction of the weights (e.g., 1/N of attention heads).

```
All Nodes Process All Layers

Layer N - Attention

Node 0
Q/K/V: heads 0-15

Input (B, T, H)

Node 1
Q/K/V: heads 16-31

Node 0
O proj: shard 0

Node 1
O proj: shard 1

mx.distributed.all_sum()

Output (B, T, H)
```

**Diagram: Tensor Parallelism Weight Sharding**

**Sources:** [src/exo/worker/engines/mlx/auto\_parallel.py293-393](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/auto_parallel.py#L293-L393)

### Implementation Strategy

The `tensor_auto_parallel()` function selects a model-specific sharding strategy based on the architecture. It patches linear layers using `shard_linear` from the MLX library.

**Supported Architectures:**

- **Llama/Mistral**: `LlamaShardingStrategy` shards Q/K/V/O projections and MLP gate/up/down projections. [src/exo/worker/engines/mlx/auto\_parallel.py421-446](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/auto_parallel.py#L421-L446)
- **DeepSeek V3/V3.2**: `DeepseekShardingStrategy` handles LoRA-compressed attention and MoE expert weights. [src/exo/worker/engines/mlx/auto\_parallel.py475-522](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/auto_parallel.py#L475-L522)
- **Qwen/GLM MoE**: `QwenShardingStrategy` shards sparse MoE blocks and wraps them in `ShardedQwenMoE` to perform `all_sum` on expert outputs. [src/exo/worker/engines/mlx/auto\_parallel.py654-692](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/auto_parallel.py#L654-L692)

#### TensorShardMetadata

Nodes in a tensor parallel group receive `TensorShardMetadata`. Unlike pipeline shards, these metadata objects typically cover the full layer range (`start_layer=0`, `end_layer=n_layers`) but assign unique `device_rank` values for weight partitioning.

**Sources:** [src/exo/shared/types/worker/shards.py26-32](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/worker/shards.py#L26-L32)

### Constraints

- **Model Card Flag**: The `ModelCard` must have `supports_tensor=True`. [src/exo/shared/models/model\_cards.py97](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/models/model_cards.py#L97-L97)
- **Divisibility**: The `hidden_size` and `num_key_value_heads` must be divisible by the number of nodes in the cycle. [src/exo/master/placement.py136-142](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement.py#L136-L142)

## CFG Parallelism (Image Models)

For diffusion models like FLUX, exo supports CFG (Classifier-Free Guidance) parallelism. Since image generation often involves two parallel prompt branches (conditional/positive and unconditional/negative), exo can shard these branches across different nodes.

- **`CfgShardMetadata`**: Defines which branch (positive or negative) a runner handles. [src/exo/shared/types/worker/shards.py35-39](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/worker/shards.py#L35-L39)
- **Implementation**: The `DistributedImageModel` coordinates execution between the branches. [src/exo/worker/runner/image\_inference/diffusion\_runner.py](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/runner/image_inference/diffusion_runner.py) (implied by metadata usage).

## Performance Implications

| Strategy | Memory Efficiency | Latency (Prefill) | Latency (Decode) | Network Requirement |
| --- | --- | --- | --- | --- |
| **Pipeline** | High (Sharded weights) | Sequential (Slow) | High (Bubble) | Moderate (Activations) |
| **Tensor** | Low (Full weights\*) | Parallel (Fast) | Low (Fast) | Very High (All-Reduce) |
| **CFG** | Moderate | Parallel | N/A | Low (Final Image) |

_\*Note: While Tensor Parallelism shards weights, every node still requires significant memory for the full KV cache and base model overhead._

**Sources:** [src/exo/master/placement.py130-156](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/master/placement.py#L130-L156) [src/exo/worker/engines/mlx/auto\_parallel.py97-105](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/worker/engines/mlx/auto_parallel.py#L97-L105)

## Code Entity Map

```

defines

«enumeration»

Sharding

Pipeline

Tensor

CFG

«abstract»

ShardMetadata

+ModelCard model_card

PipelineShardMetadata

+int start_layer

+int end_layer

TensorShardMetadata

+int device_rank

CfgShardMetadata

+bool is_positive

ShardAssignments

+ModelId model_id

+dict runner_to_shard
```

**Sources:** [src/exo/shared/types/worker/shards.py1-40](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/worker/shards.py#L1-L40) [src/exo/shared/types/worker/runners.py41](https://github.com/exo-explore/exo/blob/1e51dc89/src/exo/shared/types/worker/runners.py#L41-L41)

Dismiss

Refresh this wiki

Enter email to refresh

### On this page

- [Sharding Strategies](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#sharding-strategies)
- [Purpose and Scope](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#purpose-and-scope)
- [Sharding Strategy Overview](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#sharding-strategy-overview)
- [Pipeline Parallelism](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#pipeline-parallelism)
- [Concept](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#concept)
- [Layer Allocation](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#layer-allocation)
- [Implementation](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#implementation)
- [PipelineShardMetadata](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#pipelineshardmetadata)
- [Deadlock Prevention](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#deadlock-prevention)
- [Tensor Parallelism](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#tensor-parallelism)
- [Concept](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#concept-1)
- [Implementation Strategy](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#implementation-strategy)
- [TensorShardMetadata](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#tensorshardmetadata)
- [Constraints](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#constraints)
- [CFG Parallelism (Image Models)](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#cfg-parallelism-image-models)
- [Performance Implications](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#performance-implications)
- [Code Entity Map](https://deepwiki.com/exo-explore/exo/4.2-sharding-strategies#code-entity-map)

Ask Devin about exo-explore/exo

Fast