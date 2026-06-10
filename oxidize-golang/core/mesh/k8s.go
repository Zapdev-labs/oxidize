package mesh

import (
	"errors"
	"fmt"
	"strings"
)

const bytesPerGiB uint64 = 1_073_741_824

var (
	ErrEmptyClusterName         = errors.New("mesh: cluster name is empty")
	ErrEmptyClusterUID          = errors.New("mesh: cluster uid is empty")
	ErrEmptyModelID             = errors.New("mesh: model id is empty")
	ErrInvalidReplicaRange      = errors.New("mesh: serving min replicas exceeds max replicas")
	ErrInvalidCollectiveTimeout = errors.New("mesh: collective timeout must be greater than zero")
	ErrInvalidListenPort        = errors.New("mesh: listen port must be between 0 and 65535")
	ErrInvalidGPUCount          = errors.New("mesh: gpu count per pod must be greater than zero when gpu is required")
)

type ModelSource struct {
	ID           string
	Format       string
	Revision     string
	Quantization string
}

type ServingSpec struct {
	MinReplicas       int
	MaxReplicas       int
	OpenAICompatible  bool
	RealtimeWebSocket bool
}

type MeshK8sSpec struct {
	Namespace                string
	Strategy                 string
	ListenPort               int
	CollectiveTimeoutSeconds int
}

type GPUPlacement struct {
	Required     bool
	ResourceName string
	CountPerPod  int
	MinMemoryGiB uint64
	RequireRDMA  bool
}

type RolloutPolicy struct {
	MaxUnavailable      int
	MaxSurge            int
	DrainTimeoutSeconds int
}

type OxidizeClusterSpec struct {
	Name      string
	Namespace string
	UID       string
	Model     ModelSource
	Serving   ServingSpec
	Mesh      MeshK8sSpec
	GPU       GPUPlacement
	Rollout   RolloutPolicy
}

type PlannedPhase string

const (
	PlannedPhasePending  PlannedPhase = "Pending"
	PlannedPhaseReady    PlannedPhase = "Ready"
	PlannedPhaseDegraded PlannedPhase = "Degraded"
)

type PlannedConditionType string

const (
	PlannedConditionReady         PlannedConditionType = "Ready"
	PlannedConditionMeshConverged PlannedConditionType = "MeshConverged"
	PlannedConditionDegraded      PlannedConditionType = "Degraded"
)

type PlannedCondition struct {
	Type    PlannedConditionType
	Status  bool
	Reason  string
	Message string
}

type PlannedClusterStatus struct {
	Phase        PlannedPhase
	LeaderPeerID string
	PeersReady   int
	PeersDesired int
	Strategy     string
	Conditions   []PlannedCondition
}

type PlannedNodeCapabilities struct {
	DeviceType  string
	MemoryBytes uint64
	CanShard    bool
	Tags        map[string]string
}

type K8sMeshPlan struct {
	MeshConfig   MeshConfig
	PodEnv       map[string]string
	Capabilities PlannedNodeCapabilities
	Status       PlannedClusterStatus
}

func PlanK8sMesh(spec OxidizeClusterSpec, readyPeers int, leaderPeerID string) (K8sMeshPlan, error) {
	if err := validateK8sSpec(spec); err != nil {
		return K8sMeshPlan{}, err
	}
	meshNamespace := fmt.Sprintf("%s-%s", spec.Mesh.Namespace, spec.UID)
	capabilities := plannedNodeCapabilities(spec)
	plan := K8sMeshPlan{
		MeshConfig: MeshConfig{
			NodeID:       "",
			DiscoveryURL: fmt.Sprintf("k8s://%s/%s", spec.Namespace, spec.Name),
			BindAddr:     fmt.Sprintf(":%d", spec.Mesh.ListenPort),
		},
		PodEnv: map[string]string{
			"OXIDIZE_MESH_NAMESPACE":  meshNamespace,
			"OXIDIZE_MODEL_ID":        spec.Model.ID,
			"OXIDIZE_CLUSTER_UID":     spec.UID,
			"OXIDIZE_MODEL_CACHE_DIR": "/var/lib/oxidize/model-cache",
		},
		Capabilities: capabilities,
		Status:       plannedClusterStatus(spec, readyPeers, leaderPeerID),
	}
	return plan, nil
}

func validateK8sSpec(spec OxidizeClusterSpec) error {
	if strings.TrimSpace(spec.Name) == "" {
		return ErrEmptyClusterName
	}
	if strings.TrimSpace(spec.UID) == "" {
		return ErrEmptyClusterUID
	}
	if strings.TrimSpace(spec.Model.ID) == "" {
		return ErrEmptyModelID
	}
	if spec.Serving.MinReplicas < 0 || spec.Serving.MaxReplicas < 0 || spec.Serving.MinReplicas > spec.Serving.MaxReplicas {
		return ErrInvalidReplicaRange
	}
	if spec.Mesh.CollectiveTimeoutSeconds <= 0 {
		return ErrInvalidCollectiveTimeout
	}
	if spec.Mesh.ListenPort < 0 || spec.Mesh.ListenPort > 65535 {
		return ErrInvalidListenPort
	}
	if spec.GPU.Required && spec.GPU.CountPerPod <= 0 {
		return ErrInvalidGPUCount
	}
	return nil
}

func plannedNodeCapabilities(spec OxidizeClusterSpec) PlannedNodeCapabilities {
	deviceType := "cpu"
	memoryBytes := spec.GPU.MinMemoryGiB * bytesPerGiB
	tags := map[string]string{
		"k8s.cluster":   spec.Name,
		"k8s.namespace": spec.Namespace,
		"k8s.uid":       spec.UID,
	}
	if spec.GPU.Required {
		deviceType = "cuda"
		tags["gpu.vendor"] = gpuVendor(spec.GPU.ResourceName)
		tags["gpu.resource"] = spec.GPU.ResourceName
		tags["gpu.count"] = fmt.Sprintf("%d", spec.GPU.CountPerPod)
		tags["gpu.memory_bytes"] = fmt.Sprintf("%d", memoryBytes)
		tags["fabric.rdma"] = fmt.Sprintf("%t", spec.GPU.RequireRDMA)
		tags["backend.cuda"] = "true"
	}
	if memoryBytes < 8_000_000_000 {
		memoryBytes = 8_000_000_000
	}
	return PlannedNodeCapabilities{
		DeviceType:  deviceType,
		MemoryBytes: memoryBytes,
		CanShard:    true,
		Tags:        tags,
	}
}

func plannedClusterStatus(spec OxidizeClusterSpec, readyPeers int, leaderPeerID string) PlannedClusterStatus {
	peersDesired := spec.Serving.MinReplicas
	if readyPeers >= peersDesired {
		return PlannedClusterStatus{
			Phase:        PlannedPhaseReady,
			LeaderPeerID: leaderPeerID,
			PeersReady:   readyPeers,
			PeersDesired: peersDesired,
			Strategy:     spec.Mesh.Strategy,
			Conditions: []PlannedCondition{{
				Type:    PlannedConditionMeshConverged,
				Status:  true,
				Reason:  "MeshConverged",
				Message: "all desired mesh peers are ready",
			}},
		}
	}
	return PlannedClusterStatus{
		Phase:        PlannedPhaseDegraded,
		LeaderPeerID: leaderPeerID,
		PeersReady:   readyPeers,
		PeersDesired: peersDesired,
		Strategy:     spec.Mesh.Strategy,
		Conditions: []PlannedCondition{{
			Type:    PlannedConditionDegraded,
			Status:  true,
			Reason:  "InsufficientReadyPeers",
			Message: fmt.Sprintf("%d of %d desired peers are ready", readyPeers, peersDesired),
		}},
	}
}

func gpuVendor(resourceName string) string {
	if strings.HasPrefix(resourceName, "nvidia.com/") {
		return "nvidia"
	}
	if strings.HasPrefix(resourceName, "amd.com/") {
		return "amd"
	}
	return "unknown"
}
