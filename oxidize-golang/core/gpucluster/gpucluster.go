// Package gpucluster models the Oxidize GPU cluster specification
// (docs/gpu_cluster_spec.md) as code: typed GPU tier profiles, Kubernetes/Helm
// manifest generation, and runtime GPU detection via nvidia-smi.
//
// It is a feature-parity port of the Rust oxidize_core::gpu_cluster module.
// YAML is emitted via string building to keep the package dependency-free.
package gpucluster

import (
	"os/exec"
	"strconv"
	"strings"
)

// Family is one of the three GPU tiers the cluster targets.
type Family int

const (
	B200 Family = iota
	A100
	RTXPro6000
)

// AllFamilies returns the families in spec order.
func AllFamilies() []Family { return []Family{B200, A100, RTXPro6000} }

// Slug returns the oxidize.io/gpu-family label value.
func (f Family) Slug() string {
	switch f {
	case B200:
		return "b200"
	case A100:
		return "a100"
	case RTXPro6000:
		return "rtx-pro-6000"
	default:
		return "unknown"
	}
}

// FamilyFromSlug parses a family from its label value, case-insensitively.
// The second return value is false when the slug is unrecognized.
func FamilyFromSlug(s string) (Family, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "b200":
		return B200, true
	case "a100":
		return A100, true
	case "rtx-pro-6000", "rtx-pro6000", "rtxpro6000":
		return RTXPro6000, true
	default:
		return 0, false
	}
}

// Profile is the static hardware/scheduling profile for a GPU tier.
type Profile struct {
	Family            Family
	Product           string // NVML product name
	Generation        string // architecture shorthand
	MemoryMiB         uint32
	TDPWatts          uint32
	NVLink            bool
	MIGCapable        bool
	TimeSliceReplicas uint32
	NetworkClass      string
	WorkloadType      string
}

// ProfileFor returns the canonical profile for a family.
func ProfileFor(f Family) Profile {
	switch f {
	case B200:
		return Profile{B200, "NVIDIA-B200", "blackwell", 196608, 1000, true, false, 1, "infiniband", "training"}
	case A100:
		return Profile{A100, "NVIDIA-A100-SXM4-80GB", "ampere", 81920, 400, true, true, 2, "infiniband", "mixed"}
	case RTXPro6000:
		return Profile{RTXPro6000, "NVIDIA-RTX-Pro-6000", "ada", 98304, 300, false, false, 8, "ethernet", "workstation"}
	default:
		return Profile{}
	}
}

// AllProfiles returns the profiles for every family.
func AllProfiles() []Profile {
	out := make([]Profile, 0, 3)
	for _, f := range AllFamilies() {
		out = append(out, ProfileFor(f))
	}
	return out
}

// NodePoolSpec sizes a node pool of a given GPU family.
type NodePoolSpec struct {
	Family     Family
	NodeCount  uint32
	GPUPerNode uint32
}

func poolName(f Family) string {
	switch f {
	case B200:
		return "b200-training"
	case A100:
		return "a100-mixed"
	case RTXPro6000:
		return "rtx-pro6000"
	default:
		return "unknown"
	}
}

// NodePoolYAML renders the node-pool stanza for a single pool (spec §3.1).
func NodePoolYAML(spec NodePoolSpec) string {
	p := ProfileFor(spec.Family)
	var b strings.Builder
	b.WriteString("  " + poolName(spec.Family) + ":\n")
	b.WriteString("    count: " + strconv.FormatUint(uint64(spec.NodeCount), 10) + "\n")
	b.WriteString("    gpuPerNode: " + strconv.FormatUint(uint64(spec.GPUPerNode), 10) + "\n")
	b.WriteString("    labels:\n")
	b.WriteString("      oxidize.io/gpu-family: " + p.Family.Slug() + "\n")
	b.WriteString("      oxidize.io/gpu-arch: " + p.Generation + "\n")
	b.WriteString("      oxidize.io/workload-type: " + p.WorkloadType + "\n")
	b.WriteString("      oxidize.io/network-class: " + p.NetworkClass + "\n")
	b.WriteString("    taints:\n")
	b.WriteString("      - key: oxidize.io/gpu\n")
	b.WriteString("        value: " + p.Family.Slug() + "\n")
	b.WriteString("        effect: NoSchedule\n")
	return b.String()
}

// NodePoolsYAML renders the full nodePools document.
func NodePoolsYAML(specs []NodePoolSpec) string {
	var b strings.Builder
	b.WriteString("nodePools:\n")
	for _, s := range specs {
		b.WriteString(NodePoolYAML(s))
	}
	return b.String()
}

// Label is a key/value node label.
type Label struct{ Key, Value string }

// NodeLabels renders the GFD/scheduling labels a node must carry (spec §3.2).
func NodeLabels(f Family, gpuCount uint32) []Label {
	p := ProfileFor(f)
	migCapable := "false"
	if p.MIGCapable {
		migCapable = "true"
	}
	return []Label{
		{"nvidia.com/gpu.present", "true"},
		{"nvidia.com/gpu.product", p.Product},
		{"nvidia.com/gpu.count", strconv.FormatUint(uint64(gpuCount), 10)},
		{"nvidia.com/gpu.memory", strconv.FormatUint(uint64(p.MemoryMiB), 10)},
		{"nvidia.com/mig.capable", migCapable},
		{"oxidize.io/gpu-family", p.Family.Slug()},
		{"oxidize.io/gpu-generation", p.Generation},
		{"oxidize.io/network-class", p.NetworkClass},
	}
}

// DevicePluginConfigYAML renders the device-plugin time-slicing ConfigMap (§4.3).
func DevicePluginConfigYAML(families []Family) string {
	var b strings.Builder
	b.WriteString("apiVersion: v1\n")
	b.WriteString("kind: ConfigMap\n")
	b.WriteString("metadata:\n")
	b.WriteString("  name: nvidia-device-plugin-config\n")
	b.WriteString("  namespace: kube-system\n")
	b.WriteString("data:\n")
	b.WriteString("  config.yaml: |\n")
	b.WriteString("    version: v1\n")
	b.WriteString("    sharing:\n")
	b.WriteString("      timeSlicing:\n")
	b.WriteString("        renameByDefault: false\n")
	b.WriteString("        failRequestsGreaterThanOne: true\n")
	b.WriteString("        resources:\n")
	b.WriteString("          - name: nvidia.com/gpu\n")
	b.WriteString("            replicas: 1\n")

	var overrides []Family
	for _, f := range families {
		if ProfileFor(f).TimeSliceReplicas > 1 {
			overrides = append(overrides, f)
		}
	}
	if len(overrides) > 0 {
		b.WriteString("    nodes:\n")
		for _, f := range overrides {
			p := ProfileFor(f)
			b.WriteString("      - match:\n")
			b.WriteString("          - key: oxidize.io/gpu-family\n")
			b.WriteString("            operator: In\n")
			b.WriteString("            values:\n")
			b.WriteString("              - " + p.Family.Slug() + "\n")
			b.WriteString("        sharing:\n")
			b.WriteString("          timeSlicing:\n")
			b.WriteString("            renameByDefault: true\n")
			b.WriteString("            resources:\n")
			b.WriteString("              - name: nvidia.com/gpu\n")
			b.WriteString("                replicas: " + strconv.FormatUint(uint64(p.TimeSliceReplicas), 10) + "\n")
		}
	}
	return b.String()
}

// MIGProfile is a recommended MIG geometry (spec §4.4).
type MIGProfile struct {
	Name       string
	MemoryGB   uint32
	ComputeSMs uint32
	BestFor    string
}

// MIGProfiles returns the recommended A100 MIG geometries.
func MIGProfiles() []MIGProfile {
	return []MIGProfile{
		{"1g.10gb", 10, 14, "light inference, micro-services"},
		{"2g.20gb", 20, 28, "medium inference, small training"},
		{"3g.40gb", 40, 42, "large model inference, fine-tuning"},
		{"4g.40gb", 40, 56, "heavy inference, data processing"},
		{"7g.80gb", 80, 108, "large training (disable MIG)"},
	}
}

// MIGConfigYAML renders the MIG strategy ConfigMap, or ("", false) when the
// family is not MIG-capable.
func MIGConfigYAML(f Family) (string, bool) {
	p := ProfileFor(f)
	if !p.MIGCapable {
		return "", false
	}
	var b strings.Builder
	b.WriteString("apiVersion: v1\n")
	b.WriteString("kind: ConfigMap\n")
	b.WriteString("metadata:\n")
	b.WriteString("  name: nvidia-mig-config\n")
	b.WriteString("  namespace: kube-system\n")
	b.WriteString("data:\n")
	b.WriteString("  config.yaml: |\n")
	b.WriteString("    version: v1\n")
	b.WriteString("    flags:\n")
	b.WriteString("      migStrategy: mixed\n")
	b.WriteString("    nodes:\n")
	b.WriteString("      - match:\n")
	b.WriteString("          - key: nvidia.com/gpu.product\n")
	b.WriteString("            operator: In\n")
	b.WriteString("            values:\n")
	b.WriteString("              - " + p.Product + "\n")
	b.WriteString("        mig:\n")
	b.WriteString("          strategy: mixed\n")
	return b.String(), true
}

// GPUTolerationsYAML renders the standard pod tolerations (spec §3.3).
func GPUTolerationsYAML() string {
	return "tolerations:\n" +
		"  - key: \"oxidize.io/gpu\"\n" +
		"    operator: \"Exists\"\n" +
		"    effect: \"NoSchedule\"\n" +
		"  - key: \"nvidia.com/gpu\"\n" +
		"    operator: \"Exists\"\n" +
		"    effect: \"NoSchedule\"\n"
}

// HelmValuesYAML renders the GPU-Operator Helm values for a family (spec §5.2).
func HelmValuesYAML(f Family) string {
	p := ProfileFor(f)
	driver := "535.161.08"
	open := "false"
	if p.Generation == "blackwell" {
		driver = "550.54.15"
		open = "true"
	}
	mig := "false"
	if p.MIGCapable {
		mig = "true"
	}
	var b strings.Builder
	b.WriteString("# GPU-Operator values for the " + p.Family.Slug() + " (" + p.Generation + ") node pool\n")
	b.WriteString("driver:\n")
	b.WriteString("  enabled: true\n")
	b.WriteString("  version: \"" + driver + "\"\n")
	b.WriteString("  useOpenKernelModules: " + open + "\n")
	b.WriteString("toolkit:\n")
	b.WriteString("  enabled: true\n")
	b.WriteString("devicePlugin:\n")
	b.WriteString("  enabled: true\n")
	b.WriteString("  config:\n")
	b.WriteString("    name: nvidia-device-plugin-config\n")
	b.WriteString("dcgmExporter:\n")
	b.WriteString("  enabled: true\n")
	b.WriteString("  serviceMonitor:\n")
	b.WriteString("    enabled: true\n")
	b.WriteString("migManager:\n")
	b.WriteString("  enabled: " + mig + "\n")
	return b.String()
}

// PrometheusRulesYAML renders the DCGM PrometheusRule for GPU alerts (§8.1).
func PrometheusRulesYAML() string {
	return "apiVersion: monitoring.coreos.com/v1\n" +
		"kind: PrometheusRule\n" +
		"metadata:\n" +
		"  name: oxidize-gpu-alerts\n" +
		"spec:\n" +
		"  groups:\n" +
		"    - name: gpu-health\n" +
		"      rules:\n" +
		"        - alert: GPUHighTemperature\n" +
		"          expr: dcgm_gpu_temp > 85\n" +
		"          for: 5m\n" +
		"          labels:\n" +
		"            severity: critical\n" +
		"        - alert: GPUMemoryNearExhaustion\n" +
		"          expr: dcgm_fb_used / (dcgm_fb_free + dcgm_fb_used) > 0.95\n" +
		"          for: 10m\n" +
		"          labels:\n" +
		"            severity: warning\n" +
		"        - alert: NVLinkError\n" +
		"          expr: dcgm_nvlink_replay_error_count_total > 0\n" +
		"          for: 1m\n" +
		"          labels:\n" +
		"            severity: critical\n"
}

// DetectedGPU is a physical GPU discovered via nvidia-smi.
type DetectedGPU struct {
	Index          uint32
	Name           string
	MemoryTotalMiB uint32
	MIGEnabled     bool
	Family         Family // valid only when FamilyKnown is true
	FamilyKnown    bool
}

// ClassifyProduct classifies an NVML/nvidia-smi product name into a Family.
// The second return value is false when the name is unrecognized.
func ClassifyProduct(name string) (Family, bool) {
	n := strings.ToLower(name)
	switch {
	case strings.Contains(n, "b200"):
		return B200, true
	case strings.Contains(n, "a100"):
		return A100, true
	case (strings.Contains(n, "rtx") || strings.Contains(n, "pro")) && strings.Contains(n, "6000"):
		return RTXPro6000, true
	default:
		return 0, false
	}
}

// ParseNvidiaSMICSV parses the CSV output of
// nvidia-smi --query-gpu=index,name,memory.total,mig.mode.current
// --format=csv,noheader,nounits. Unparseable lines are skipped.
func ParseNvidiaSMICSV(output string) []DetectedGPU {
	var gpus []DetectedGPU
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Split(line, ",")
		for i := range fields {
			fields[i] = strings.TrimSpace(fields[i])
		}
		if len(fields) < 3 {
			continue
		}
		index, err := strconv.ParseUint(fields[0], 10, 32)
		if err != nil {
			continue
		}
		mem, err := strconv.ParseUint(fields[2], 10, 32)
		if err != nil {
			continue
		}
		migEnabled := len(fields) > 3 && strings.EqualFold(fields[3], "enabled")
		fam, known := ClassifyProduct(fields[1])
		gpus = append(gpus, DetectedGPU{
			Index:          uint32(index),
			Name:           fields[1],
			MemoryTotalMiB: uint32(mem),
			MIGEnabled:     migEnabled,
			Family:         fam,
			FamilyKnown:    known,
		})
	}
	return gpus
}

// DetectGPUs probes the local node for NVIDIA GPUs via nvidia-smi. It returns
// an empty slice when nvidia-smi is unavailable or fails.
func DetectGPUs() []DetectedGPU {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,name,memory.total,mig.mode.current",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return []DetectedGPU{}
	}
	return ParseNvidiaSMICSV(string(out))
}

// FamilyCount pairs a family with a count.
type FamilyCount struct {
	Family Family
	Count  uint32
}

// Summarize counts detected GPUs by family, in spec order.
func Summarize(gpus []DetectedGPU) []FamilyCount {
	counts := map[Family]uint32{}
	for _, g := range gpus {
		if g.FamilyKnown {
			counts[g.Family]++
		}
	}
	var out []FamilyCount
	for _, f := range AllFamilies() {
		if n := counts[f]; n > 0 {
			out = append(out, FamilyCount{f, n})
		}
	}
	// out is already in AllFamilies spec order; no extra sort needed.
	return out
}
