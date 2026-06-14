from __future__ import annotations

import pytest

from oxidize_python.core.mesh.k8s import (
    GPUPlacement,
    K8sPlanError,
    MeshK8sSpec,
    ModelSource,
    OxidizeClusterSpec,
    PlannedPhase,
    RolloutPolicy,
    ServingSpec,
    plan_k8s_mesh,
)


def llama_cluster_spec() -> OxidizeClusterSpec:
    return OxidizeClusterSpec(
        name="llama-70b",
        namespace="inference",
        uid="uid-123",
        model=ModelSource(
            id="meta-llama/Llama-3.1-70B-Instruct",
            format="gguf",
            revision="main",
            quantization="Q4_K_M",
        ),
        serving=ServingSpec(
            min_replicas=2,
            max_replicas=8,
            openai_compatible=True,
            realtime_websocket=True,
        ),
        mesh=MeshK8sSpec(
            namespace="llama-70b",
            strategy="Pipeline",
            listen_port=0,
            collective_timeout_seconds=60,
        ),
        gpu=GPUPlacement(
            required=True,
            resource_name="nvidia.com/gpu",
            count_per_pod=1,
            min_memory_gib=40,
            require_rdma=False,
        ),
        rollout=RolloutPolicy(max_unavailable=0, max_surge=1, drain_timeout_seconds=120),
    )


def test_plan_k8s_mesh_derives_namespace_env_and_gpu_tags() -> None:
    plan = plan_k8s_mesh(llama_cluster_spec(), ready_peers=4, leader_peer_id="12D3KooW")

    assert plan.mesh_config.discovery_url == "k8s://inference/llama-70b"
    assert plan.pod_env["OXIDIZE_MESH_NAMESPACE"] == "llama-70b-uid-123"
    assert plan.pod_env["OXIDIZE_MODEL_ID"] == "meta-llama/Llama-3.1-70B-Instruct"
    assert plan.capabilities.device_type == "cuda"
    assert plan.capabilities.tags["gpu.memory_bytes"] == "42949672960"
    assert plan.status.phase is PlannedPhase.READY


def test_plan_k8s_mesh_degrades_when_required_capacity_is_missing() -> None:
    plan = plan_k8s_mesh(llama_cluster_spec(), ready_peers=1, leader_peer_id="")

    assert plan.status.phase is PlannedPhase.DEGRADED
    assert plan.status.conditions[0].reason == "InsufficientReadyPeers"


def test_plan_k8s_mesh_rejects_malformed_cluster_specs() -> None:
    spec = llama_cluster_spec()
    spec.mesh.collective_timeout_seconds = 0

    with pytest.raises(K8sPlanError, match="collective timeout"):
        plan_k8s_mesh(spec, ready_peers=2, leader_peer_id="")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_replicas", -1, "replicas"),
        ("max_replicas", -1, "replicas"),
        ("collective_timeout_seconds", -1, "collective timeout"),
        ("listen_port", -1, "listen port"),
        ("count_per_pod", -1, "gpu count"),
        ("min_memory_gib", -1, "gpu memory"),
    ],
)
def test_plan_k8s_mesh_rejects_negative_boundary_values(
    field: str,
    value: int,
    message: str,
) -> None:
    spec = llama_cluster_spec()
    match field:
        case "min_replicas":
            spec.serving.min_replicas = value
        case "max_replicas":
            spec.serving.max_replicas = value
        case "collective_timeout_seconds":
            spec.mesh.collective_timeout_seconds = value
        case "listen_port":
            spec.mesh.listen_port = value
        case "count_per_pod":
            spec.gpu.count_per_pod = value
        case "min_memory_gib":
            spec.gpu.min_memory_gib = value
        case unreachable:
            raise AssertionError(unreachable)

    with pytest.raises(K8sPlanError, match=message):
        plan_k8s_mesh(spec, ready_peers=2, leader_peer_id="")
