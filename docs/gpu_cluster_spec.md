# Oxidize GPU Cluster Technical Specification

**Project:** zapdev-labs/oxidize  
**Version:** 1.0.0  
**Date:** 2026-06-07  
**Status:** Draft  

---

## 1. Executive Summary

This specification defines the architecture, configuration, and operational requirements for deploying and managing a heterogeneous NVIDIA GPU cluster within Kubernetes (k8s) and K9s environments for the Oxidize project. The cluster targets three distinct NVIDIA GPU tiers: HPC/AI training (B200), datacenter inference/training (A100), and professional visualization/workstation (RTX Pro 6000).

---

## 2. Target GPU Hardware

### 2.1 NVIDIA B200 (Blackwell Architecture)

| Attribute | Specification |
|-----------|--------------|
| Architecture | Blackwell |
| Memory | 192 GB HBM3e |
| TDP | 1000W |
| Form Factor | HGX B200 (NVLink) or PCIe |
| NVLink | NVLink 5th Gen, 900 GB/s |
| FP8 Tensor Core | Yes |
| Transformer Engine | 2nd Gen |
| Target Workloads | LLM training, large-scale HPC, multi-modal AI |

**Node Constraints:**
- Requires NVLink switch infrastructure for multi-GPU scaling
- Power delivery: 1200W+ per socket capability
- Cooling: Direct liquid cooling (DLC) recommended
- Network: NDR 400 InfiniBand or 400GbE for scale-out

### 2.2 NVIDIA A100 (Ampere Architecture)

| Attribute | Specification |
|-----------|--------------|
| Architecture | Ampere |
| Memory | 40 GB or 80 GB HBM2e |
| TDP | 400W (SXM4) / 250W (PCIe) |
| Form Factor | SXM4 or PCIe Gen4 |
| NVLink | NVLink 3.0, 600 GB/s |
| MIG Support | Yes (up to 7 instances per GPU) |
| Target Workloads | ML training, inference at scale, scientific computing |

**Node Constraints:**
- SXM4 requires NVIDIA HGX A100 baseboard
- MIG requires A100 40GB or 80GB SKU
- PCIe variants suitable for mainstream servers

### 2.3 NVIDIA RTX Pro 6000 (Workstation/Professional)

| Attribute | Specification |
|-----------|--------------|
| Architecture | Ada Lovelace / Blackwell class |
| Memory | 48-96 GB GDDR6 |
| TDP | 300W |
| Form Factor | Dual-slot PCIe |
| NVLink | No (single GPU deployments) |
| Target Workloads | CAD/CAE, media rendering, AI development, edge inference |

**Node Constraints:**
- Standard ATX/EATX chassis compatible
- No NVLink; multi-GPU uses PCIe P2P only
- Display outputs present (headless mode required in k8s)

---

## 3. Kubernetes Cluster Architecture

### 3.1 Node Pools by GPU Tier

```yaml
# Conceptual node pool layout
nodePools:
  b200-training:
    instanceType: custom-hgx-b200
    count: 8
    gpuPerNode: 8
    labels:
      oxidize.io/gpu-family: b200
      oxidize.io/gpu-arch: blackwell
      oxidize.io/workload-type: training
      oxidize.io/network-class: infiniband
    taints:
      - key: oxidize.io/gpu
        value: b200
        effect: NoSchedule

  a100-mixed:
    instanceType: custom-hgx-a100
    count: 16
    gpuPerNode: 8
    labels:
      oxidize.io/gpu-family: a100
      oxidize.io/gpu-arch: ampere
      oxidize.io/workload-type: mixed
      oxidize.io/mig-capable: "true"
    taints:
      - key: oxidize.io/gpu
        value: a100
        effect: NoSchedule

  rtx-pro6000:
    instanceType: custom-workstation
    count: 4
    gpuPerNode: 2
    labels:
      oxidize.io/gpu-family: rtx-pro-6000
      oxidize.io/gpu-arch: ada-lovelace
      oxidize.io/workload-type: workstation
      oxidize.io/display-capable: "true"
    taints:
      - key: oxidize.io/gpu
        value: rtx-pro-6000
        effect: NoSchedule
```

### 3.2 Node Labeling Strategy

All GPU nodes MUST carry the following label taxonomy:

| Label Key | Example Value | Description |
|-----------|--------------|-------------|
| `nvidia.com/gpu.present` | `true` | Managed by GPU Operator / Device Plugin |
| `nvidia.com/gpu.product` | `NVIDIA-B200` / `NVIDIA-A100-SXM4-80GB` / `NVIDIA-RTX-Pro-6000` | Exact product name from NVML |
| `nvidia.com/gpu.count` | `8` | Physical GPUs per node |
| `nvidia.com/mig.capable` | `true` / `false` | MIG availability |
| `oxidize.io/gpu-family` | `b200` / `a100` / `rtx-pro-6000` | Oxidize custom grouping |
| `oxidize.io/gpu-generation` | `blackwell` / `ampere` / `ada` | Architecture shorthand |
| `oxidize.io/network-class` | `infiniband` / `ethernet` | Interconnect type |

### 3.3 Taints and Tolerations

GPU nodes use dedicated taints to prevent non-GPU workloads from scheduling.

**Node Taint (applied via kubelet or cluster provisioning):**
```yaml
spec:
  taints:
    - key: oxidize.io/gpu
      value: "<gpu-family>"
      effect: NoSchedule
    - key: nvidia.com/gpu
      value: "true"
      effect: NoSchedule
```

**Pod Toleration (required for all GPU workloads):**
```yaml
tolerations:
  - key: "oxidize.io/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
```

**Soft Anti-Affinity for Mixed Clusters:**
If a job requests specific GPU types, node affinity should be preferred over required to allow fallback:
```yaml
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
            - key: oxidize.io/gpu-family
              operator: In
              values:
                - b200
```

---

## 4. NVIDIA Device Plugin for Kubernetes

### 4.1 Overview

The NVIDIA Device Plugin is the foundational component that exposes NVIDIA GPUs to the Kubernetes scheduler as `nvidia.com/gpu` extended resources.

**Version Target:** v0.16.0+ (supports Time-Slicing, MIG Strategy, ConfigMap-based configuration)

### 4.2 Deployment via DaemonSet

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
        - key: oxidize.io/gpu
          operator: Exists
          effect: NoSchedule
      priorityClassName: system-node-critical
      containers:
        - name: nvidia-device-plugin-ctr
          image: nvcr.io/nvidia/k8s-device-plugin:v0.16.0
          args: ["--config-file", "/etc/nvidia-device-plugin/config.yaml"]
          env:
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: DEVICE_LIST_STRATEGY
              value: "envvar"  # or "volume-mounts"
            - name: DEVICE_SHARING_STRATEGY
              value: "none"    # overridden by config per node
            - name: NVIDIA_MIG_MONITOR_DEVICES
              value: "all"
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
          volumeMounts:
            - name: device-plugin
              mountPath: /var/lib/kubelet/device-plugins
            - name: nvidia-device-plugin-config
              mountPath: /etc/nvidia-device-plugin
      volumes:
        - name: device-plugin
          hostPath:
            path: /var/lib/kubelet/device-plugins
        - name: nvidia-device-plugin-config
          configMap:
            name: nvidia-device-plugin-config
```

### 4.3 Time-Slicing Configuration (RTX Pro 6000 / A100 Inference)

For inference workloads that do not fully saturate a GPU, time-slicing allows multiple Pods to share a single GPU via temporal multiplexing.

```yaml
# ConfigMap: nvidia-device-plugin-config
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvidia-device-plugin-config
  namespace: kube-system
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        renameByDefault: false
        failRequestsGreaterThanOne: true
        resources:
          - name: nvidia.com/gpu
            replicas: 4
    # Node-specific overrides
    nodes:
      - match:
          - key: oxidize.io/gpu-family
            operator: In
            values:
              - rtx-pro-6000
        sharing:
          timeSlicing:
            renameByDefault: true
            resources:
              - name: nvidia.com/gpu
                replicas: 8
      - match:
          - key: nvidia.com/gpu.product
            operator: In
            values:
              - NVIDIA-A100-SXM4-80GB
              - NVIDIA-A100-PCIe-80GB
        sharing:
          timeSlicing:
            renameByDefault: true
            resources:
              - name: nvidia.com/gpu
                replicas: 2
```

**Behavior:**
- RTX Pro 6000: Each physical GPU appears as 8 allocatable `nvidia.com/gpu` units
- A100 80GB: Each physical GPU appears as 2 allocatable units (conservative for mixed workloads)
- B200: No time-slicing; full GPU allocation only (`failRequestsGreaterThanOne: true`)

### 4.4 MIG Strategy Configuration (A100 Only)

MIG (Multi-Instance GPU) partitions an A100 into up to 7 isolated instances with dedicated compute and memory.

```yaml
# ConfigMap override for A100 nodes with MIG enabled
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvidia-mig-config
  namespace: kube-system
data:
  config.yaml: |
    version: v1
    flags:
      migStrategy: mixed  # options: none, single, mixed
    nodes:
      - match:
          - key: nvidia.com/gpu.product
            operator: In
            values:
              - NVIDIA-A100-SXM4-80GB
        mig:
          strategy: mixed
          # Example MIG geometries
          # 1g.10gb x 7 + 2g.20gb x 1 + 3g.40gb x 1 (varies by SKU)
```

**MIG Profile Recommendations by Workload:**

| Profile | GPU Partition | Memory | Compute Units | Best For |
|---------|--------------|--------|---------------|----------|
| 1g.10gb | 1/7 GPU | 10 GB | 14 SMs | Light inference, micro-services |
| 2g.20gb | 2/7 GPU | 20 GB | 28 SMs | Medium inference, small training |
| 3g.40gb | 3/7 GPU | 40 GB | 42 SMs | Large model inference, fine-tuning |
| 4g.40gb | 4/7 GPU | 40 GB | 56 SMs | Heavy inference, data processing |
| 7g.80gb | Full GPU | 80 GB | 108 SMs | Large training (disable MIG) |

**MIG Resource Naming in Kubernetes:**
When `migStrategy: mixed`, the device plugin advertises:
- `nvidia.com/mig-1g.10gb`
- `nvidia.com/mig-2g.20gb`
- `nvidia.com/mig-3g.40gb`
- etc.

---

## 5. NVIDIA GPU Operator

### 5.1 Overview

The NVIDIA GPU Operator automates the deployment and lifecycle management of:
- NVIDIA drivers
- Container Toolkit (nvidia-container-runtime)
- Device Plugin
- DCGM (Data Center GPU Manager)
- DCGM Exporter (Prometheus metrics)
- Node Feature Discovery (NFD) + NVIDIA Device Feature Discovery (DFD)
- GPU Feature Discovery (GFD)
- Validator + Cleanup utilities

**Version Target:** v24.3.0+ (Blackwell support begins in this series)

### 5.2 Installation via Helm

```bash
# Add the NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator for Oxidize
cat > oxidize-gpu-operator-values.yaml << 'EOF'
# --- Driver Configuration ---
driver:
  enabled: true
  version: "550.54.15"  # Minimum for Blackwell/B200; adjust as needed
  usePrecompiled: false
  repository: nvcr.io/nvidia
  image: driver
  upgradePolicy:
    autoUpgrade: true
    drain:
      enable: true
      force: false
      podSelector: "app!=critical"

# --- Container Toolkit ---
toolkit:
  enabled: true
  env:
    - name: NVIDIA_CONTAINER_RUNTIME_MODES_CDI_ENABLED
      value: "true"

# --- Device Plugin ---
devicePlugin:
  enabled: true
  config:
    name: nvidia-device-plugin-config
    default: ""

# --- DCGM and DCGM Exporter ---
dcgm:
  enabled: true
  resources:
    limits:
      nvidia.com/gpu: 0  # DCGM does not need a GPU on the node

dcgmExporter:
  enabled: true
  serviceMonitor:
    enabled: true  # Requires Prometheus Operator
  resources:
    limits:
      nvidia.com/gpu: 0

# --- Feature Discovery ---
node-feature-discovery:
  enabled: true
  worker:
    sources:
      pci:
        deviceClassWhitelist:
          - "0200"
          - "0300"
          - "0302"
      usb:
        deviceClassWhitelist: []
      custom:
        - name: "oxidize.gpu.labels"
          labels:
            oxidize.io/gpu-ready: "true"
          matchFeatures:
            - feature: pci.device
              matchExpressions:
                vendor: { op: In, value: ["10de"] }

gfd:
  enabled: true
  config:
    name: ""
    default: ""
  # Generate node labels for GPU properties
  nodeSelector:
    nvidia.com/gpu.present: "true"
EOF

# Deploy
helm upgrade --install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  -f oxidize-gpu-operator-values.yaml \
  --wait
```

### 5.3 Component-Specific Configuration

#### 5.3.1 NVIDIA Driver Container

For B200 support, ensure the driver container image includes Blackwell-compatible drivers (550.x or newer).

```yaml
driver:
  enabled: true
  version: "550.54.15"
  useOpenKernelModules: true  # Required for newer kernels and Blackwell
  kernelModuleConfig:
    name: ""
  env:
    - name: NVIDIA_DRIVER_CAPABILITIES
      value: "compute,utility"
```

**Kernel Requirements:**
- Linux kernel 5.14+ (recommended: 6.2+ for full Blackwell support)
- `nouveau` MUST be blacklisted
- `nvidia` and `nvidia_drm` modules loaded at boot

#### 5.3.2 NVIDIA Container Toolkit

Enables GPU containers via the `nvidia` runtime class.

```yaml
runtime:
  enabled: true
  className: nvidia
  env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: "all"
    - name: NVIDIA_DRIVER_CAPABILITIES
      value: "compute,utility"
```

All Oxidize GPU Pods MUST specify:
```yaml
spec:
  runtimeClassName: nvidia
  containers:
    - resources:
        limits:
          nvidia.com/gpu: 1
```

#### 5.3.3 GPU Feature Discovery (GFD)

Generates detailed GPU labels for scheduling and affinity.

Generated labels include:
- `nvidia.com/gpu.product=NVIDIA-B200`
- `nvidia.com/gpu.memory=196608` (MiB for B200)
- `nvidia.com/gpu.count=8`
- `nvidia.com/mig.capable=true` (A100 only)
- `nvidia.com/gpu.family=nvidia-blackwell`
- `nvidia.com/cuda.driver.major=550`

---

## 6. Scheduling and Resource Management

### 6.1 Extended Resource Requests

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: oxidize-training-job
spec:
  runtimeClassName: nvidia
  tolerations:
    - key: oxidize.io/gpu
      operator: Exists
      effect: NoSchedule
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: oxidize.io/gpu-family
                operator: In
                values: ["b200"]
  containers:
    - name: trainer
      image: zapdev-labs/oxidize-train:1.2.0
      resources:
        limits:
          nvidia.com/gpu: 8  # Request entire B200 node
          memory: "1.5Ti"
          cpu: "128"
        requests:
          nvidia.com/gpu: 8
          memory: "1.5Ti"
          cpu: "128"
      env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3,4,5,6,7"
```

### 6.2 Multi-GPU Scheduling with NVLink

For B200 and A100 (SXM) nodes with NVLink, use `extended` resources and topology awareness.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: oxidize-multi-gpu
  annotations:
    nvidia.com/requirements: "nvidia.com/gpu.present=true, topology-aware"
spec:
  schedulerName: default-scheduler  # or volcano/yunikorn for gang scheduling
  containers:
    - name: workload
      resources:
        limits:
          nvidia.com/gpu: 4
```

**Topology Awareness:**
The NVIDIA Device Plugin does not natively enforce local NVLink topology. For strict topology (e.g., 4 GPUs within the same NVLink domain), use the [NVIDIA GPU Scheduler Extender](https://github.com/NVIDIA/k8s-scheduler-extender) or a scheduler supporting topology constraints.

### 6.3 Gang Scheduling for Distributed Training

Distributed training across multiple B200 nodes requires gang scheduling to avoid deadlocks.

**Recommended:** [Volcano](https://volcano.sh/) or [Run:ai Scheduler](https://www.run.ai/)

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: oxidize-b200-training
spec:
  minMember: 4          # 4 Pods
  queue: oxidize-gpu
  priorityClassName: high-priority-gpu
  minResources:
    nvidia.com/gpu: 32  # 4 nodes x 8 GPUs
    memory: "6Ti"
    cpu: "512"
```

---

## 7. Networking for GPU Clusters

### 7.1 InfiniBand / NDR for B200

B200 clusters require high-bandwidth, low-latency interconnect for distributed training.

**Requirements:**
- NVIDIA ConnectX-7 / BlueField-3 DPUs
- NDR 400 Gbps InfiniBand or RoCE
- Subnet Manager (OpenSM) running on the fabric

**Kubernetes NetworkAttachmentDefinition:**
```yaml
apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ib0-b200
  namespace: default
spec:
  config: |
    {
      "cniVersion": "0.3.1",
      "type": "ib-sriov",
      "pfNames": ["ib0"],
      "ipam": { "type": "host-local", "subnet": "192.168.100.0/24" }
    }
```

**Pod Annotation for Multi-Homed GPU Pods:**
```yaml
metadata:
  annotations:
    k8s.v1.cni.cncf.io/networks: "ib0-b200"
```

### 7.2 GPUDirect RDMA

Enable GPUDirect RDMA for zero-copy GPU memory transfer over InfiniBand.

**Node Requirements:**
- `nvidia_peermem` or `nv_p2p` kernel module loaded
- `nvidia-fabricmanager` service running on NVSwitch-equipped nodes (DGX/HGX)

```bash
# Verify GPUDirect on node
lsmod | grep nvidia_peermem
# or for newer drivers
lsmod | grep nvidia_p2p

# Fabric manager for NVSwitch (required on B200 DGX nodes)
systemctl status nvidia-fabricmanager
```

---

## 8. Monitoring, Observability, and DCGM

### 8.1 DCGM Metrics via Prometheus

DCGM Exporter exposes GPU telemetry as Prometheus metrics.

**Key Metrics for Oxidize:**

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `dcgm_gpu_temp` | GPU temperature (C) | > 85C |
| `dcgm_power_usage` | Power draw (W) | > 95% TDP |
| `dcgm_gpu_utilization` | Compute utilization (%) | < 10% for > 1h (waste) |
| `dcgm_fb_free` / `dcgm_fb_used` | Frame buffer usage | > 95% alloc |
| `dcgm_pcie_tx_bytes` / `dcgm_pcie_rx_bytes` | PCIe traffic | Baseline anomaly |
| `dcgm_nvlink_bandwidth_total` | NVLink traffic | For topology validation |
| `dcgm_nvlink_replay_error_count_total` | NVLink errors | > 0 |

**PrometheusRule for Oxidize:**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: oxidize-gpu-alerts
spec:
  groups:
    - name: gpu-health
      rules:
        - alert: GPUHighTemperature
          expr: dcgm_gpu_temp > 85
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "GPU {{ $labels.gpu }} on {{ $labels.hostname }} overheating"

        - alert: GPUMemoryNearExhaustion
          expr: dcgm_fb_used / (dcgm_fb_free + dcgm_fb_used) > 0.95
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "GPU memory nearly exhausted on {{ $labels.hostname }}"

        - alert: NVLinkError
          expr: dcgm_nvlink_replay_error_count_total > 0
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "NVLink errors detected on {{ $labels.hostname }}"
```

### 8.2 Grafana Dashboards

Recommended dashboards:
- **NVIDIA DCGM Exporter Dashboard** (ID: 12239)
- **Oxidize Training Job Dashboard** (custom): per-job GPU utilization, memory, NVLink bandwidth, checkpoint I/O
- **Oxidize Cluster Capacity Dashboard**: allocatable vs requested GPUs by family and node

---

## 9. Benchmarking Considerations

### 9.1 Baseline Benchmarks by GPU

Establish baseline performance metrics for capacity planning and regression detection.

| Benchmark | Tool | Target Metric | B200 Expected | A100-80GB Expected | RTX Pro 6000 Expected |
|-----------|------|---------------|---------------|-------------------|----------------------|
| FP8 Matmul | cuBLAS / Transformer Engine | TFLOPS | ~4500 | N/A | ~330 (FP16) |
| FP16 Matmul | cuBLAS | TFLOPS | ~2200 | 312 | 90-100 |
| Memory BW | nvbandwidth / deviceQuery | GB/s | 8000 | 2000 | 960 |
| All-Reduce (NVLink) | NCCL Tests | Bus BW | ~900 GB/s | ~600 GB/s | N/A |
| All-Reduce (IB) | NCCL Tests | Bus BW | ~50 GB/s | ~50 GB/s | N/A |
| LLM Inference | vLLM / TensorRT-LLM | tok/sec | Baseline | Baseline | Baseline |
| MLPerf Training | MLPerf Training | Time to convergence | Benchmark | Benchmark | N/A |

### 9.2 NCCL Tests for Scale-Out Validation

```bash
# Install NCCL tests in benchmark container
apt-get update && apt-get install -y git build-essential mpi-default-dev

git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda

# Run all-reduce on 8 x B200 within a node
mpirun --allow-run-as-root -np 8 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_HCA=mlx5_0 \
  ./build/all_reduce_perf -b 8M -e 1G -f 2 -g 1
```

### 9.3 Node-Level Stress Test

```bash
# GPU burn test (thermal and stability)
docker run --rm --gpus all zapdev-labs/oxidize-gpu-burn 600  # 10 minutes

# Memory bandwidth validation
./deviceQuery && ./bandwidthTest --device=all

# MIG validation (A100)
nvidia-smi mig -lgip    # List GPU instance profiles
nvidia-smi mig -lgi     # List created GPU instances
nvidia-smi mig -lcip    # List compute instance profiles
```

### 9.4 Continuous Benchmarking Pipeline

Propose integrating benchmarks into CI/CD:

```yaml
# GitHub Actions / Argo Workflows stub
name: oxidize-gpu-benchmark
on:
  schedule:
    - cron: "0 2 * * 0"  # Weekly Sunday 2AM
tolerations:
  - key: oxidize.io/gpu
    operator: Exists
    effect: NoSchedule
jobs:
  benchmark-b200:
    runs-on: [self-hosted, gpu-b200]
    steps:
      - run: nvidia-smi
      - run: ./scripts/benchmark-nccl.sh
      - run: ./scripts/benchmark-mlperf-inference.sh
      - uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: "customSmallerIsBetter"
          output-file-path: benchmark-results/b200.json
```

---

## 10. K9s Operational Considerations

### 10.1 K9s Resource Views for GPU Workloads

When using K9s for operational visibility:

**Recommended Plugins / Views:**
- `nvidia-smi` daemonset for real-time GPU inspection:
  ```yaml
  kubectl describe node <node> | grep -A 5 "Allocated resources"
  kubectl top node <node>
  kubectl get pods -o custom-columns=\"POD:.metadata.name,GPU:.spec.containers[*].resources.limits.nvidia.com/gpu\"
  ```

**K9s Plugin Snippet (`~/.config/k9s/plugins.yml`):**
```yaml
plugin:
  oxidize-gpu-smi:
    shortCut: Shift-G
    description: GPU Status
    scopes:
      - node
    command: kubectl
    background: false
    args:
      - exec
      - -it
      - nvidia-driver-daemonset-XXXX
      - --
      - nvidia-smi
```

### 10.2 GPU Node Maintenance

**Cordoning and Draining GPU Nodes:**
```bash
# Identify GPU job distribution
kubectl get pods -o wide --all-namespaces | grep -E "(b200|a100|rtx-pro)"

# Cordon node
kubectl cordon oxidize-b200-node-03

# Drain with GPU-aware grace period (training jobs may need long gracePeriod)
kubectl drain oxidize-b200-node-03 \
  --ignore-daemonsets \
  --delete-emptydir-data \
  --pod-selector="app=oxidize-training" \
  --grace-period=300

# Verify no GPU pods remain
kubectl get pods --field-selector spec.nodeName=oxidize-b200-node-03
```

---

## 11. Security Hardening

### 11.1 Pod Security Standards

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: oxidize-gpu-workload
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
      resources:
        limits:
          nvidia.com/gpu: 1
```

### 11.2 Network Policies for GPU Nodes

Restrict east-west traffic to GPU nodes:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gpu-node-isolation
  namespace: oxidize-gpu
spec:
  podSelector:
    matchLabels:
      oxidize.io/gpu-family: b200
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: oxidize-control
      ports:
        - protocol: TCP
          port: 22
        - protocol: TCP
          port: 6443
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: oxidize-storage
      ports:
        - protocol: TCP
          port: 2049  # NFS
```

### 11.3 CDI (Container Device Interface) Migration

Future-proof GPU access with CDI (supported in Container Toolkit v1.14+):

```yaml
spec:
  containers:
    - name: cdi-gpu-app
      resources:
        limits:
          nvidia.com/gpu: 1
      env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "nvidia.com/gpu=all"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility"
```

CDI removes the need for `runtimeClassName: nvidia` in future Kubernetes releases (target: v1.32+).

---

## 12. Troubleshooting Guide

### 12.1 Common Issues

**Pod stuck in Pending:**
```bash
kubectl describe pod <pod-name>
# Check for:
# 0/8 nodes are available: 8 Insufficient nvidia.com/gpu
# -> Check node allocatable resources
kubectl get node <node> -o jsonpath='{.status.allocatable}' | jq .
```

**nvidia-smi not found inside Pod:**
- Verify Container Toolkit is running on the node
- Check runtime class: `kubectl get pod <pod> -o jsonpath='{.spec.runtimeClassName}'`
- Validate Device Plugin pod is running in `kube-system`

**MIG not available on A100:**
```bash
# Verify MIG mode is enabled
nvidia-smi -i 0 -q | grep "MIG Mode"
# If Disabled:
sudo nvidia-smi -i 0 -mig 1
# Enable requires GPU reset (drain node first)
```

**NVLink errors on B200:**
```bash
# Check fabric manager logs
journalctl -u nvidia-fabricmanager -n 100

# Verify NVSwitch status
nvidia-smi nvlink --status
nvidia-smi nvlink --error_counter
```

**Low GPU utilization during training:**
- Check batch size vs GPU memory (`dcgm_fb_used`)
- Verify data loader CPU bottleneck (`htop` on node)
- Profile with Nsight Systems / PyTorch Profiler

### 12.2 Log Aggregation

Forward device plugin, DCGM, and driver container logs to central logging.

```yaml
# Fluent Bit filter for GPU Operator namespace
[PARSER]
    Name        gpu_operator
    Format      json
    Time_Key    ts
    Time_Format %Y-%m-%dT%H:%M:%S.%L
    Time_Keep   On

[FILTER]
    Name                kubernetes
    Match               kube.gpu-operator.*
    Kube_URL            https://kubernetes.default.svc:443
    Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
```

---

## 13. Helm Chart Summary for Oxidize

```
gpu-operator/
├── values-oxidize.yaml          # Base configuration
├── values-b200.yaml             # B200-specific overrides
├── values-a100.yaml             # A100-specific overrides (MIG strategies)
├── values-rtx-pro6000.yaml      # RTX Pro 6000 overrides (time-slicing)
└── templates/
    ├── configmap-device-plugin.yaml
    ├── configmap-mig.yaml
    ├── networkattachment-ib.yaml
    └── prometheus-rules.yaml
```

---

## 14. Future Considerations

1. **Dynamic MIG Reconfiguration:** Kubernetes-native MIG geometry changes without node reboot (requires NVIDIA MIG Partition Editor + custom operator logic)
2. **GPU Overcommitment:** Time-slicing evolution into true QoS tiers (best-effort vs guaranteed)
3. **GPUDirect Storage:** Direct GPU-to-storage (GDS) for checkpoint loading; requires Magnum IO and compatible storage backend
4. **Confidential Computing:** NVIDIA Confidential Computing on B200 for encrypted AI training workloads
5. **K8s Autoscaling:** Cluster API + NVIDIA GPU Operator for GPU node auto-scaling (CA with GPU-aware node templates)

---

## 15. References

- [NVIDIA Device Plugin Documentation](https://github.com/NVIDIA/k8s-device-plugin)
- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)
- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/index.html)
- [Kubernetes Extended Resources](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#extended-resources)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
- [NCCL Tests Repository](https://github.com/NVIDIA/nccl-tests)
- [NGC Catalog](https://catalog.ngc.nvidia.com/)
- [NVIDIA Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/dgx-b200/) (when available)

---

**Document Control:**  
**Author:** Oxidize Platform Team  
**Reviewer:** Infrastructure / SRE  
**Approval:** Engineering Lead
