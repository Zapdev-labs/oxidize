package gpucluster

import (
	"strings"
	"testing"
)

func TestFamilySlugRoundtrips(t *testing.T) {
	for _, f := range AllFamilies() {
		got, ok := FamilyFromSlug(f.Slug())
		if !ok || got != f {
			t.Fatalf("roundtrip failed for %v: got %v ok=%v", f, got, ok)
		}
	}
	if f, ok := FamilyFromSlug("RtxPro6000"); !ok || f != RTXPro6000 {
		t.Fatalf("alias parse failed: %v %v", f, ok)
	}
	if _, ok := FamilyFromSlug("unknown"); ok {
		t.Fatal("expected unknown slug to fail")
	}
}

func TestProfilesMatchSpec(t *testing.T) {
	b := ProfileFor(B200)
	if b.MemoryMiB != 196608 || b.TDPWatts != 1000 || !b.NVLink || b.MIGCapable || b.TimeSliceReplicas != 1 {
		t.Fatalf("B200 profile mismatch: %+v", b)
	}
	a := ProfileFor(A100)
	if !a.MIGCapable || a.TimeSliceReplicas != 2 || a.Product != "NVIDIA-A100-SXM4-80GB" {
		t.Fatalf("A100 profile mismatch: %+v", a)
	}
	r := ProfileFor(RTXPro6000)
	if r.NVLink || r.TimeSliceReplicas != 8 || r.NetworkClass != "ethernet" {
		t.Fatalf("RTX profile mismatch: %+v", r)
	}
}

func TestNodePoolYAML(t *testing.T) {
	y := NodePoolYAML(NodePoolSpec{B200, 8, 8})
	for _, want := range []string{
		"b200-training:", "count: 8", "oxidize.io/gpu-family: b200",
		"oxidize.io/gpu-arch: blackwell", "key: oxidize.io/gpu", "effect: NoSchedule",
	} {
		if !strings.Contains(y, want) {
			t.Fatalf("node pool yaml missing %q:\n%s", want, y)
		}
	}
}

func TestNodePoolsYAMLListsAll(t *testing.T) {
	y := NodePoolsYAML([]NodePoolSpec{{B200, 8, 8}, {A100, 16, 8}, {RTXPro6000, 4, 2}})
	if !strings.HasPrefix(y, "nodePools:\n") {
		t.Fatal("missing nodePools header")
	}
	for _, want := range []string{"b200-training:", "a100-mixed:", "rtx-pro6000:"} {
		if !strings.Contains(y, want) {
			t.Fatalf("missing pool %q", want)
		}
	}
}

func TestNodeLabels(t *testing.T) {
	labels := NodeLabels(A100, 8)
	m := map[string]string{}
	for _, l := range labels {
		m[l.Key] = l.Value
	}
	if m["nvidia.com/gpu.product"] != "NVIDIA-A100-SXM4-80GB" ||
		m["nvidia.com/gpu.count"] != "8" ||
		m["nvidia.com/gpu.memory"] != "81920" ||
		m["nvidia.com/mig.capable"] != "true" ||
		m["oxidize.io/gpu-generation"] != "ampere" {
		t.Fatalf("labels mismatch: %v", m)
	}
}

func TestDevicePluginConfigOverrides(t *testing.T) {
	b := DevicePluginConfigYAML([]Family{B200})
	if !strings.Contains(b, "failRequestsGreaterThanOne: true") || strings.Contains(b, "nodes:") {
		t.Fatalf("B200 should emit no overrides:\n%s", b)
	}
	y := DevicePluginConfigYAML([]Family{RTXPro6000, A100})
	for _, want := range []string{"nodes:", "- rtx-pro-6000", "replicas: 8", "- a100", "replicas: 2"} {
		if !strings.Contains(y, want) {
			t.Fatalf("missing %q:\n%s", want, y)
		}
	}
}

func TestMIGConfigOnlyForA100(t *testing.T) {
	if _, ok := MIGConfigYAML(A100); !ok {
		t.Fatal("A100 should be MIG-capable")
	}
	if _, ok := MIGConfigYAML(B200); ok {
		t.Fatal("B200 should not be MIG-capable")
	}
	y, _ := MIGConfigYAML(A100)
	if !strings.Contains(y, "migStrategy: mixed") || !strings.Contains(y, "NVIDIA-A100-SXM4-80GB") {
		t.Fatalf("mig config content wrong:\n%s", y)
	}
}

func TestMIGProfiles(t *testing.T) {
	names := []string{}
	for _, p := range MIGProfiles() {
		names = append(names, p.Name)
	}
	want := "1g.10gb,2g.20gb,3g.40gb,4g.40gb,7g.80gb"
	if strings.Join(names, ",") != want {
		t.Fatalf("mig profile names = %v", names)
	}
}

func TestHelmValuesBlackwellDriver(t *testing.T) {
	b := HelmValuesYAML(B200)
	if !strings.Contains(b, "550.54.15") || !strings.Contains(b, "useOpenKernelModules: true") {
		t.Fatalf("B200 helm values wrong:\n%s", b)
	}
	if !strings.Contains(b, "migManager:\n  enabled: false") {
		t.Fatal("B200 migManager should be false")
	}
	a := HelmValuesYAML(A100)
	if !strings.Contains(a, "migManager:\n  enabled: true") {
		t.Fatal("A100 migManager should be true")
	}
}

func TestTolerationsAndPrometheus(t *testing.T) {
	if !strings.Contains(GPUTolerationsYAML(), "oxidize.io/gpu") {
		t.Fatal("tolerations missing key")
	}
	rules := PrometheusRulesYAML()
	if !strings.Contains(rules, "GPUHighTemperature") ||
		!strings.Contains(rules, "dcgm_nvlink_replay_error_count_total > 0") {
		t.Fatal("prometheus rules missing alerts")
	}
}

func TestClassifyProduct(t *testing.T) {
	cases := map[string]struct {
		fam Family
		ok  bool
	}{
		"NVIDIA B200":              {B200, true},
		"NVIDIA A100-SXM4-80GB":    {A100, true},
		"NVIDIA A100-PCIE-40GB":    {A100, true},
		"NVIDIA RTX PRO 6000":      {RTXPro6000, true},
		"Tesla V100":               {0, false},
	}
	for name, exp := range cases {
		fam, ok := ClassifyProduct(name)
		if ok != exp.ok || (ok && fam != exp.fam) {
			t.Fatalf("classify %q = (%v,%v), want (%v,%v)", name, fam, ok, exp.fam, exp.ok)
		}
	}
}

func TestParseNvidiaSMICSV(t *testing.T) {
	out := "0, NVIDIA A100-SXM4-80GB, 81920, Enabled\n" +
		"1, NVIDIA B200, 196608, Disabled\n" +
		"garbage line\n" +
		"2, Tesla V100, 16384, [N/A]\n"
	gpus := ParseNvidiaSMICSV(out)
	if len(gpus) != 3 {
		t.Fatalf("expected 3 gpus, got %d", len(gpus))
	}
	if !gpus[0].FamilyKnown || gpus[0].Family != A100 || !gpus[0].MIGEnabled {
		t.Fatalf("gpu0 wrong: %+v", gpus[0])
	}
	if !gpus[1].FamilyKnown || gpus[1].Family != B200 || gpus[1].MIGEnabled {
		t.Fatalf("gpu1 wrong: %+v", gpus[1])
	}
	if gpus[2].FamilyKnown || gpus[2].MemoryTotalMiB != 16384 {
		t.Fatalf("gpu2 wrong: %+v", gpus[2])
	}
}

func TestSummarize(t *testing.T) {
	gpus := ParseNvidiaSMICSV(
		"0, NVIDIA A100-SXM4-80GB, 81920, Disabled\n" +
			"1, NVIDIA A100-SXM4-80GB, 81920, Disabled\n" +
			"2, NVIDIA B200, 196608, Disabled\n")
	got := Summarize(gpus)
	if len(got) != 2 || got[0].Family != B200 || got[0].Count != 1 || got[1].Family != A100 || got[1].Count != 2 {
		t.Fatalf("summary wrong: %+v", got)
	}
}

func TestDetectGPUsSafeWithoutHardware(t *testing.T) {
	_ = DetectGPUs() // must not panic
}
