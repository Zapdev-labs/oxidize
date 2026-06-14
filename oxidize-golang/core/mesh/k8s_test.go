package mesh

import "testing"

func llamaClusterSpec() OxidizeClusterSpec {
	return OxidizeClusterSpec{
		Name:      "llama-70b",
		Namespace: "inference",
		UID:       "uid-123",
		Model: ModelSource{
			ID:           "meta-llama/Llama-3.1-70B-Instruct",
			Format:       "gguf",
			Revision:     "main",
			Quantization: "Q4_K_M",
		},
		Serving: ServingSpec{MinReplicas: 2, MaxReplicas: 8, OpenAICompatible: true, RealtimeWebSocket: true},
		Mesh:    MeshK8sSpec{Namespace: "llama-70b", Strategy: "Pipeline", ListenPort: 0, CollectiveTimeoutSeconds: 60},
		GPU:     GPUPlacement{Required: true, ResourceName: "nvidia.com/gpu", CountPerPod: 1, MinMemoryGiB: 40},
		Rollout: RolloutPolicy{MaxUnavailable: 0, MaxSurge: 1, DrainTimeoutSeconds: 120},
	}
}

func TestPlanK8sMeshDerivesNamespaceEnvAndGPUTags(t *testing.T) {
	plan, err := PlanK8sMesh(llamaClusterSpec(), 4, "12D3KooW")
	if err != nil {
		t.Fatal(err)
	}
	if plan.MeshConfig.DiscoveryURL != "k8s://inference/llama-70b" {
		t.Fatalf("unexpected discovery url %q", plan.MeshConfig.DiscoveryURL)
	}
	if got := plan.PodEnv["OXIDIZE_MESH_NAMESPACE"]; got != "llama-70b-uid-123" {
		t.Fatalf("unexpected mesh namespace %q", got)
	}
	if got := plan.Capabilities.Tags["gpu.memory_bytes"]; got != "42949672960" {
		t.Fatalf("unexpected gpu memory tag %q", got)
	}
	if plan.Status.Phase != PlannedPhaseReady {
		t.Fatalf("expected ready, got %s", plan.Status.Phase)
	}
}

func TestPlanK8sMeshDegradesWhenRequiredCapacityIsMissing(t *testing.T) {
	plan, err := PlanK8sMesh(llamaClusterSpec(), 1, "")
	if err != nil {
		t.Fatal(err)
	}
	if plan.Status.Phase != PlannedPhaseDegraded {
		t.Fatalf("expected degraded, got %s", plan.Status.Phase)
	}
	if len(plan.Status.Conditions) == 0 || plan.Status.Conditions[0].Reason != "InsufficientReadyPeers" {
		t.Fatalf("expected insufficient peers condition, got %#v", plan.Status.Conditions)
	}
}

func TestPlanK8sMeshRejectsMalformedClusterSpecs(t *testing.T) {
	spec := llamaClusterSpec()
	spec.Mesh.CollectiveTimeoutSeconds = 0
	if _, err := PlanK8sMesh(spec, 2, ""); err != ErrInvalidCollectiveTimeout {
		t.Fatalf("expected ErrInvalidCollectiveTimeout, got %v", err)
	}
}

func TestPlanK8sMeshRejectsNegativeBoundaryValues(t *testing.T) {
	cases := []struct {
		name string
		edit func(*OxidizeClusterSpec)
		want error
	}{
		{
			name: "min replicas",
			edit: func(spec *OxidizeClusterSpec) { spec.Serving.MinReplicas = -1 },
			want: ErrInvalidReplicaRange,
		},
		{
			name: "max replicas",
			edit: func(spec *OxidizeClusterSpec) { spec.Serving.MaxReplicas = -1 },
			want: ErrInvalidReplicaRange,
		},
		{
			name: "collective timeout",
			edit: func(spec *OxidizeClusterSpec) { spec.Mesh.CollectiveTimeoutSeconds = -1 },
			want: ErrInvalidCollectiveTimeout,
		},
		{
			name: "listen port",
			edit: func(spec *OxidizeClusterSpec) { spec.Mesh.ListenPort = -1 },
			want: ErrInvalidListenPort,
		},
		{
			name: "gpu count",
			edit: func(spec *OxidizeClusterSpec) { spec.GPU.CountPerPod = -1 },
			want: ErrInvalidGPUCount,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spec := llamaClusterSpec()
			tc.edit(&spec)
			if _, err := PlanK8sMesh(spec, 2, ""); err != tc.want {
				t.Fatalf("expected %v, got %v", tc.want, err)
			}
		})
	}
}
