use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{MeshConfig, NodeCapabilities, ParallelismStrategy};

const BYTES_PER_GIB: u64 = 1_073_741_824;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSource {
    pub id: String,
    pub format: String,
    pub revision: String,
    pub quantization: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServingSpec {
    pub min_replicas: usize,
    pub max_replicas: usize,
    pub openai_compatible: bool,
    pub realtime_websocket: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshK8sSpec {
    pub namespace: String,
    pub strategy: ParallelismStrategy,
    pub listen_port: u16,
    pub collective_timeout_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuPlacement {
    pub required: bool,
    pub resource_name: String,
    pub count_per_pod: u32,
    pub min_memory_gib: u64,
    pub require_rdma: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutPolicy {
    pub max_unavailable: usize,
    pub max_surge: usize,
    pub drain_timeout_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OxidizeClusterSpec {
    pub name: String,
    pub namespace: String,
    pub uid: String,
    pub model: ModelSource,
    pub serving: ServingSpec,
    pub mesh: MeshK8sSpec,
    pub gpu: GpuPlacement,
    pub rollout: RolloutPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlannedPhase {
    Pending,
    Ready,
    Degraded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlannedConditionType {
    Ready,
    MeshConverged,
    Degraded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedCondition {
    pub condition_type: PlannedConditionType,
    pub status: bool,
    pub reason: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedClusterStatus {
    pub phase: PlannedPhase,
    pub leader_peer_id: Option<String>,
    pub peers_ready: usize,
    pub peers_desired: usize,
    pub strategy: ParallelismStrategy,
    pub conditions: Vec<PlannedCondition>,
}

pub type PlannedPodEnv = HashMap<String, String>;

#[derive(Debug, Clone)]
pub struct K8sMeshPlan {
    pub mesh_config: MeshConfig,
    pub pod_env: PlannedPodEnv,
    pub capabilities: NodeCapabilities,
    pub status: PlannedClusterStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum K8sPlanError {
    #[error("cluster name is empty")]
    EmptyClusterName,
    #[error("cluster uid is empty")]
    EmptyClusterUid,
    #[error("model id is empty")]
    EmptyModelId,
    #[error("serving min replicas exceeds max replicas")]
    InvalidReplicaRange,
    #[error("collective timeout must be greater than zero")]
    InvalidCollectiveTimeout,
    #[error("gpu count per pod must be greater than zero when gpu is required")]
    InvalidGpuCount,
}

pub fn plan_k8s_mesh(
    spec: &OxidizeClusterSpec,
    ready_peers: usize,
    leader_peer_id: Option<&str>,
) -> Result<K8sMeshPlan, K8sPlanError> {
    validate_spec(spec)?;

    let mesh_namespace = format!("{}-{}", spec.mesh.namespace, spec.uid);
    let mut pod_env = HashMap::new();
    pod_env.insert("OXIDIZE_MESH_NAMESPACE".to_string(), mesh_namespace.clone());
    pod_env.insert("OXIDIZE_MODEL_ID".to_string(), spec.model.id.clone());
    pod_env.insert("OXIDIZE_CLUSTER_UID".to_string(), spec.uid.clone());
    pod_env.insert(
        "OXIDIZE_MODEL_CACHE_DIR".to_string(),
        "/var/lib/oxidize/model-cache".to_string(),
    );

    let capabilities = planned_capabilities(spec);
    let mesh_config = MeshConfig {
        listen_port: spec.mesh.listen_port,
        namespace: mesh_namespace,
        capabilities: capabilities.clone(),
    };

    let status = planned_status(spec, ready_peers, leader_peer_id);

    Ok(K8sMeshPlan {
        mesh_config,
        pod_env,
        capabilities,
        status,
    })
}

fn validate_spec(spec: &OxidizeClusterSpec) -> Result<(), K8sPlanError> {
    if spec.name.trim().is_empty() {
        return Err(K8sPlanError::EmptyClusterName);
    }
    if spec.uid.trim().is_empty() {
        return Err(K8sPlanError::EmptyClusterUid);
    }
    if spec.model.id.trim().is_empty() {
        return Err(K8sPlanError::EmptyModelId);
    }
    if spec.serving.min_replicas > spec.serving.max_replicas {
        return Err(K8sPlanError::InvalidReplicaRange);
    }
    if spec.mesh.collective_timeout_secs == 0 {
        return Err(K8sPlanError::InvalidCollectiveTimeout);
    }
    if spec.gpu.required && spec.gpu.count_per_pod == 0 {
        return Err(K8sPlanError::InvalidGpuCount);
    }
    Ok(())
}

fn planned_capabilities(spec: &OxidizeClusterSpec) -> NodeCapabilities {
    let mut tags = HashMap::new();
    let device_type = if spec.gpu.required { "cuda" } else { "cpu" };
    let memory_bytes = spec.gpu.min_memory_gib.saturating_mul(BYTES_PER_GIB);

    if spec.gpu.required {
        tags.insert(
            "gpu.vendor".to_string(),
            gpu_vendor(&spec.gpu.resource_name).to_string(),
        );
        tags.insert("gpu.resource".to_string(), spec.gpu.resource_name.clone());
        tags.insert("gpu.count".to_string(), spec.gpu.count_per_pod.to_string());
        tags.insert("gpu.memory_bytes".to_string(), memory_bytes.to_string());
        tags.insert("fabric.rdma".to_string(), spec.gpu.require_rdma.to_string());
        tags.insert("backend.cuda".to_string(), "true".to_string());
    }
    tags.insert("k8s.cluster".to_string(), spec.name.clone());
    tags.insert("k8s.namespace".to_string(), spec.namespace.clone());
    tags.insert("k8s.uid".to_string(), spec.uid.clone());

    NodeCapabilities {
        device_type: device_type.to_string(),
        memory_bytes: memory_bytes.max(8_000_000_000),
        cpu_threads: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(8),
        can_shard: true,
        tags,
    }
}

fn planned_status(
    spec: &OxidizeClusterSpec,
    ready_peers: usize,
    leader_peer_id: Option<&str>,
) -> PlannedClusterStatus {
    let peers_desired = spec.serving.min_replicas;
    let converged = ready_peers >= peers_desired;
    let phase = if converged {
        PlannedPhase::Ready
    } else {
        PlannedPhase::Degraded
    };
    let condition = if converged {
        PlannedCondition {
            condition_type: PlannedConditionType::MeshConverged,
            status: true,
            reason: "MeshConverged".to_string(),
            message: "all desired mesh peers are ready".to_string(),
        }
    } else {
        PlannedCondition {
            condition_type: PlannedConditionType::Degraded,
            status: true,
            reason: "InsufficientReadyPeers".to_string(),
            message: format!(
                "{} of {} desired peers are ready",
                ready_peers, peers_desired
            ),
        }
    };

    PlannedClusterStatus {
        phase,
        leader_peer_id: leader_peer_id.map(str::to_string),
        peers_ready: ready_peers,
        peers_desired,
        strategy: spec.mesh.strategy,
        conditions: vec![condition],
    }
}

fn gpu_vendor(resource_name: &str) -> &'static str {
    if resource_name.starts_with("nvidia.com/") {
        "nvidia"
    } else if resource_name.starts_with("amd.com/") {
        "amd"
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_cluster() -> OxidizeClusterSpec {
        OxidizeClusterSpec {
            name: "llama-70b".to_string(),
            namespace: "inference".to_string(),
            uid: "uid-123".to_string(),
            model: ModelSource {
                id: "meta-llama/Llama-3.1-70B-Instruct".to_string(),
                format: "gguf".to_string(),
                revision: "main".to_string(),
                quantization: "Q4_K_M".to_string(),
            },
            serving: ServingSpec {
                min_replicas: 2,
                max_replicas: 8,
                openai_compatible: true,
                realtime_websocket: true,
            },
            mesh: MeshK8sSpec {
                namespace: "llama-70b".to_string(),
                strategy: crate::mesh::ParallelismStrategy::Pipeline,
                listen_port: 0,
                collective_timeout_secs: 60,
            },
            gpu: GpuPlacement {
                required: true,
                resource_name: "nvidia.com/gpu".to_string(),
                count_per_pod: 1,
                min_memory_gib: 40,
                require_rdma: false,
            },
            rollout: RolloutPolicy {
                max_unavailable: 0,
                max_surge: 1,
                drain_timeout_secs: 120,
            },
        }
    }

    #[test]
    fn plan_k8s_mesh_derives_namespace_env_and_gpu_tags() {
        let plan = plan_k8s_mesh(&llama_cluster(), 4, Some("12D3KooW")).unwrap();

        assert_eq!(plan.mesh_config.namespace, "llama-70b-uid-123");
        assert_eq!(plan.mesh_config.listen_port, 0);
        assert_eq!(
            plan.pod_env
                .get("OXIDIZE_MESH_NAMESPACE")
                .map(String::as_str),
            Some("llama-70b-uid-123")
        );
        assert_eq!(
            plan.pod_env.get("OXIDIZE_MODEL_ID").map(String::as_str),
            Some("meta-llama/Llama-3.1-70B-Instruct")
        );
        assert_eq!(plan.capabilities.device_type, "cuda");
        assert_eq!(
            plan.capabilities.tags.get("gpu.vendor").map(String::as_str),
            Some("nvidia")
        );
        assert_eq!(
            plan.capabilities
                .tags
                .get("gpu.memory_bytes")
                .map(String::as_str),
            Some("42949672960")
        );
        assert_eq!(plan.status.phase, PlannedPhase::Ready);
        assert_eq!(plan.status.leader_peer_id.as_deref(), Some("12D3KooW"));
    }

    #[test]
    fn plan_k8s_mesh_degrades_when_required_gpu_capacity_is_missing() {
        let plan = plan_k8s_mesh(&llama_cluster(), 1, None).unwrap();

        assert_eq!(plan.status.phase, PlannedPhase::Degraded);
        assert!(plan.status.conditions.iter().any(|condition| {
            condition.condition_type == PlannedConditionType::Degraded
                && condition.reason == "InsufficientReadyPeers"
        }));
    }

    #[test]
    fn plan_k8s_mesh_rejects_malformed_cluster_specs() {
        let mut spec = llama_cluster();
        spec.mesh.collective_timeout_secs = 0;

        let err = plan_k8s_mesh(&spec, 2, None).unwrap_err();
        assert_eq!(err, K8sPlanError::InvalidCollectiveTimeout);
    }
}
