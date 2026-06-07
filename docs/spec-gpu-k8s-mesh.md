# Technical Specification: Distributed GPU Clusters and Kubernetes Orchestration

## 1. Introduction
This document outlines the technical design for implementing multi-node distributed GPU clusters and Kubernetes (K8s) orchestration within the Oxidize framework. The goal is to scale LLM inference from single-device/single-node to massive, elastic GPU clusters.

## 2. Architecture Overview
The system follows a hierarchical control-plane/data-plane split:
- **Control Plane**: Kubernetes Operator managing stateful sets of Oxidize workers.
- **Data Plane**: Rust-based core engine with high-performance networking (NCCL/gRPC) and memory management (PagedKV).
- **Interface Layer**: Go and Python bindings for high-level cluster coordination and application integration.

## 3. Rust Core Implementation (`oxidize-core`)

### 3.1. Tensor Parallelism (TP)
- Implement Intra-node TP using **NCCL** for NVIDIA GPUs and **RCCL** for AMD.
- Abstract communication primitives via a `CommGroup` trait to support multiple backends.
- Use a split-weight strategy for linear layers (RowParallelLinear, ColumnParallelLinear) to minimize synchronization overhead.

### 3.2. PagedKV Distributed Cache
- Extend the current `PagedKV` implementation to support **Remote Direct Memory Access (RDMA)**.
- Implement a global block manager that tracks KV-cache page locations across the cluster.
- Use a "request-steering" load balancer to route prompts to nodes containing relevant prefix cache blocks.

### 3.3. Networking: gRPC & GPUDirect RDMA
- **Control**: Use gRPC for inter-node health checks, metadata synchronization, and request routing.
- **Data**: Support **GPUDirect RDMA** for zero-copy tensor transfers between GPUs on different nodes, bypassing the CPU/Host memory bottleneck.
- Fallback to standard TCP for environments without InfiniBand/RoCE support.

## 4. Go Implementation (`oxidize-golang`)

### 4.1. CGO FFI Wrappers
- Provide thread-safe Go wrappers around the Rust distributed runtime.
- Implement a `ClusterHandle` that manages the lifecycle of local worker processes and their connection to the mesh.

### 4.2. GOMAXPROCS and Multi-worker Scaling
- Optimize Go's scheduler (`GOMAXPROCS`) to prevent contention between the gRPC control plane and the high-throughput Rust data plane.
- Implement a worker registry in Go that interacts with K8s `Service` and `Endpoints` for discovery.

### 4.3. Cluster Coordination
- Use Go to implement the consensus logic for leader election in multi-head setups (if not using K8s native leader election).

## 5. Python Implementation (`oxidize-py`)

### 5.1. FFI Bindings
- Expose the distributed configuration through Python classes (PyO3).
- Allow users to define cluster topology (e.g., `ClusterConfig(nodes=["node1", "node2"], gpus_per_node=8)`).

### 5.2. Multi-GPU Model Loaders
- Extend Python model loaders to handle sharded weights.
- Implement automated model sharding during the loading phase to distribute weights across the defined cluster topology.

### 5.3. Scaling Configuration
- Provide a high-level Python API for elastic scaling, allowing the backend to hot-plug new workers into the mesh without restarting the entire inference service.

## 6. Kubernetes Orchestration

### 6.1. Custom Resource Definitions (CRDs)
- **OxidizeCluster**: Defines the desired state (replicas, GPU requirements, model ID).
- **OxidizeWorker**: Internal resource representing an individual worker node.

### 6.2. GPU Scheduling
- Integrate with `nvidia-device-plugin` for GPU resource allocation.
- Support Kubernetes **Topology-Aware Scheduling** to ensure workers in a TP group are scheduled on nodes with high-speed interconnects (NVLink).

### 6.3. Monitoring and Observability
- Export Prometheus metrics for cluster-wide GPU utilization, NCCL throughput, and PagedKV cache hit rates.

## 7. Performance Targets
- **Intra-node scaling**: 95%+ efficiency on up to 8 GPUs.
- **Inter-node scaling**: 80%+ efficiency on up to 32 nodes using GPUDirect RDMA.
- **Latency**: < 50ms overhead for inter-node communication in standard configurations.
