from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

BYTES_PER_GIB = 1_073_741_824


class K8sPlanError(ValueError):
    pass


@dataclass(slots=True)
class ModelSource:
    id: str
    format: str
    revision: str
    quantization: str


@dataclass(slots=True)
class ServingSpec:
    min_replicas: int
    max_replicas: int
    openai_compatible: bool
    realtime_websocket: bool


@dataclass(slots=True)
class MeshK8sSpec:
    namespace: str
    strategy: str
    listen_port: int
    collective_timeout_seconds: int


@dataclass(slots=True)
class GPUPlacement:
    required: bool
    resource_name: str
    count_per_pod: int
    min_memory_gib: int
    require_rdma: bool


@dataclass(slots=True)
class RolloutPolicy:
    max_unavailable: int
    max_surge: int
    drain_timeout_seconds: int


@dataclass(slots=True)
class OxidizeClusterSpec:
    name: str
    namespace: str
    uid: str
    model: ModelSource
    serving: ServingSpec
    mesh: MeshK8sSpec
    gpu: GPUPlacement
    rollout: RolloutPolicy


class PlannedPhase(StrEnum):
    PENDING = "Pending"
    READY = "Ready"
    DEGRADED = "Degraded"


class PlannedConditionType(StrEnum):
    READY = "Ready"
    MESH_CONVERGED = "MeshConverged"
    DEGRADED = "Degraded"


@dataclass(frozen=True, slots=True)
class PlannedCondition:
    condition_type: PlannedConditionType
    status: bool
    reason: str
    message: str


@dataclass(frozen=True, slots=True)
class PlannedClusterStatus:
    phase: PlannedPhase
    leader_peer_id: str
    peers_ready: int
    peers_desired: int
    strategy: str
    conditions: tuple[PlannedCondition, ...]


@dataclass(frozen=True, slots=True)
class PlannedNodeCapabilities:
    device_type: str
    memory_bytes: int
    can_shard: bool
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlannedMeshConfig:
    discovery_url: str
    bind_addr: str
    namespace: str


@dataclass(frozen=True, slots=True)
class K8sMeshPlan:
    mesh_config: PlannedMeshConfig
    pod_env: dict[str, str]
    capabilities: PlannedNodeCapabilities
    status: PlannedClusterStatus


def plan_k8s_mesh(
    spec: OxidizeClusterSpec,
    *,
    ready_peers: int,
    leader_peer_id: str,
) -> K8sMeshPlan:
    _validate_spec(spec)
    mesh_namespace = f"{spec.mesh.namespace}-{spec.uid}"
    return K8sMeshPlan(
        mesh_config=PlannedMeshConfig(
            discovery_url=f"k8s://{spec.namespace}/{spec.name}",
            bind_addr=f":{spec.mesh.listen_port}",
            namespace=mesh_namespace,
        ),
        pod_env={
            "OXIDIZE_MESH_NAMESPACE": mesh_namespace,
            "OXIDIZE_MODEL_ID": spec.model.id,
            "OXIDIZE_CLUSTER_UID": spec.uid,
            "OXIDIZE_MODEL_CACHE_DIR": "/var/lib/oxidize/model-cache",
        },
        capabilities=_planned_capabilities(spec),
        status=_planned_status(spec, ready_peers=ready_peers, leader_peer_id=leader_peer_id),
    )


def _validate_spec(spec: OxidizeClusterSpec) -> None:
    if spec.name.strip() == "":
        raise K8sPlanError("cluster name is empty")
    if spec.uid.strip() == "":
        raise K8sPlanError("cluster uid is empty")
    if spec.model.id.strip() == "":
        raise K8sPlanError("model id is empty")
    if (
        spec.serving.min_replicas < 0
        or spec.serving.max_replicas < 0
        or spec.serving.min_replicas > spec.serving.max_replicas
    ):
        raise K8sPlanError("serving min replicas exceeds max replicas")
    if spec.mesh.collective_timeout_seconds <= 0:
        raise K8sPlanError("collective timeout must be greater than zero")
    if spec.mesh.listen_port < 0 or spec.mesh.listen_port > 65_535:
        raise K8sPlanError("listen port must be between 0 and 65535")
    if spec.gpu.required and spec.gpu.count_per_pod <= 0:
        raise K8sPlanError("gpu count per pod must be greater than zero when gpu is required")
    if spec.gpu.min_memory_gib < 0:
        raise K8sPlanError("gpu memory must be greater than or equal to zero")


def _planned_capabilities(spec: OxidizeClusterSpec) -> PlannedNodeCapabilities:
    device_type = "cuda" if spec.gpu.required else "cpu"
    memory_bytes = max(spec.gpu.min_memory_gib * BYTES_PER_GIB, 8_000_000_000)
    tags = {
        "k8s.cluster": spec.name,
        "k8s.namespace": spec.namespace,
        "k8s.uid": spec.uid,
    }
    if spec.gpu.required:
        tags.update(
            {
                "gpu.vendor": _gpu_vendor(spec.gpu.resource_name),
                "gpu.resource": spec.gpu.resource_name,
                "gpu.count": str(spec.gpu.count_per_pod),
                "gpu.memory_bytes": str(spec.gpu.min_memory_gib * BYTES_PER_GIB),
                "fabric.rdma": str(spec.gpu.require_rdma).lower(),
                "backend.cuda": "true",
            }
        )
    return PlannedNodeCapabilities(
        device_type=device_type,
        memory_bytes=memory_bytes,
        can_shard=True,
        tags=tags,
    )


def _planned_status(
    spec: OxidizeClusterSpec,
    *,
    ready_peers: int,
    leader_peer_id: str,
) -> PlannedClusterStatus:
    peers_desired = spec.serving.min_replicas
    if ready_peers >= peers_desired:
        return PlannedClusterStatus(
            phase=PlannedPhase.READY,
            leader_peer_id=leader_peer_id,
            peers_ready=ready_peers,
            peers_desired=peers_desired,
            strategy=spec.mesh.strategy,
            conditions=(
                PlannedCondition(
                    condition_type=PlannedConditionType.MESH_CONVERGED,
                    status=True,
                    reason="MeshConverged",
                    message="all desired mesh peers are ready",
                ),
            ),
        )
    return PlannedClusterStatus(
        phase=PlannedPhase.DEGRADED,
        leader_peer_id=leader_peer_id,
        peers_ready=ready_peers,
        peers_desired=peers_desired,
        strategy=spec.mesh.strategy,
        conditions=(
            PlannedCondition(
                condition_type=PlannedConditionType.DEGRADED,
                status=True,
                reason="InsufficientReadyPeers",
                message=f"{ready_peers} of {peers_desired} desired peers are ready",
            ),
        ),
    )


def _gpu_vendor(resource_name: str) -> str:
    if resource_name.startswith("nvidia.com/"):
        return "nvidia"
    if resource_name.startswith("amd.com/"):
        return "amd"
    return "unknown"
