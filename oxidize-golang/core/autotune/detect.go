// Package autotune mirrors oxidize_core::autotune — hardware detection and
// rule-based inference tuning plans.
package autotune

import (
	"os"
	"runtime"
	"strconv"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/gpucluster"
	"github.com/Zapdev-labs/oxidize/golang/core/simd"
)

// OsKind identifies the host operating system.
type OsKind int

const (
	OsLinux OsKind = iota
	OsMacos
	OsWindows
	OsOther
)

func (o OsKind) String() string {
	switch o {
	case OsLinux:
		return "Linux"
	case OsMacos:
		return "Macos"
	case OsWindows:
		return "Windows"
	default:
		return "Other"
	}
}

// CpuVendor is a best-effort CPU vendor classification.
type CpuVendor int

const (
	CpuVendorUnknown CpuVendor = iota
	CpuVendorIntel
	CpuVendorAmd
	CpuVendorArm
)

func (v CpuVendor) String() string {
	switch v {
	case CpuVendorIntel:
		return "Intel"
	case CpuVendorAmd:
		return "Amd"
	case CpuVendorArm:
		return "Arm"
	default:
		return "Unknown"
	}
}

// HardwareInventory is a snapshot of host hardware from cheap probes.
type HardwareInventory struct {
	OS                  OsKind
	CPUVendor           CpuVendor
	SIMD                simd.Backend
	PhysicalCores       int
	LogicalCores        int
	NumaNodes           int
	MinNodeRAMBytes     uint64
	TotalRAMBytes       uint64
	HasGPU              bool
	GPUFamily           *gpucluster.Family
	GPUVRAMBytes        uint64
	HasMetal            bool
	HasCUDA             bool
	HasROCm             bool
	HasRDMA             bool
	IsWSL               bool
	ContainerMemLimit   *uint64
	Hugepages2MiBAvail  bool
}

// Summary returns a one-line hardware summary.
func (h HardwareInventory) Summary() string {
	gpu := "gpu=none"
	if h.HasGPU {
		fam := "unknown"
		if h.GPUFamily != nil {
			fam = h.GPUFamily.Slug()
		}
		gpu = "gpu=" + fam + " vram=" + strconv.FormatUint(h.GPUVRAMBytes/(1024*1024), 10) + " MiB"
	}
	return strings.Join([]string{
		"os=" + h.OS.String(),
		"cpu=" + h.CPUVendor.String(),
		"simd=" + h.SIMD.String(),
		"cores=" + strconv.Itoa(h.PhysicalCores) + " (" + strconv.Itoa(h.LogicalCores) + "t)",
		"numa=" + strconv.Itoa(h.NumaNodes),
		"ram=" + strconv.FormatUint(h.TotalRAMBytes/(1<<30), 10) + " GiB",
		gpu,
		"metal=" + strconv.FormatBool(h.HasMetal),
		"cuda=" + strconv.FormatBool(h.HasCUDA),
		"wsl=" + strconv.FormatBool(h.IsWSL),
	}, " ")
}

// Detect runs all hardware probes and returns an inventory.
func Detect() HardwareInventory {
	osKind := detectOS()
	physical := runtime.NumCPU()
	if physical < 1 {
		physical = 1
	}
	logical := physical
	minNodeRAM := uint64(4) << 30
	totalRAM := detectTotalRAMBytes()
	if totalRAM == 0 {
		totalRAM = minNodeRAM
	}

	gpus := gpucluster.DetectGPUs()
	hasGPU := len(gpus) > 0
	var vram uint64
	var fam *gpucluster.Family
	for _, g := range gpus {
		vram += uint64(g.MemoryTotalMiB) * 1024 * 1024
		if g.FamilyKnown && fam == nil {
			f := g.Family
			fam = &f
		}
	}

	inv := HardwareInventory{
		OS:                 osKind,
		CPUVendor:          detectCPUVendor(),
		SIMD:               simd.Preferred(),
		PhysicalCores:      physical,
		LogicalCores:       logical,
		NumaNodes:          detectNumaNodes(),
		MinNodeRAMBytes:    minNodeRAM,
		TotalRAMBytes:      totalRAM,
		HasGPU:             hasGPU,
		GPUFamily:          fam,
		GPUVRAMBytes:       vram,
		HasMetal:           runtime.GOOS == "darwin",
		HasCUDA:            hasGPU,
		HasROCm:            false,
		HasRDMA:            false,
		IsWSL:              detectWSL(),
		ContainerMemLimit:  detectCgroupMemLimit(),
		Hugepages2MiBAvail: detectHugepages2MiB(),
	}
	return inv
}

func detectOS() OsKind {
	switch runtime.GOOS {
	case "linux":
		return OsLinux
	case "darwin":
		return OsMacos
	case "windows":
		return OsWindows
	default:
		return OsOther
	}
}

func detectTotalRAMBytes() uint64 {
	if runtime.GOOS != "linux" {
		return 0
	}
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "MemTotal:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		kb, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			continue
		}
		return kb * 1024
	}
	return 0
}

func detectCPUVendor() CpuVendor {
	if runtime.GOARCH == "arm" || runtime.GOARCH == "arm64" {
		return CpuVendorArm
	}
	if runtime.GOOS != "linux" {
		return CpuVendorUnknown
	}
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return CpuVendorUnknown
	}
	lower := strings.ToLower(string(data))
	switch {
	case strings.Contains(lower, "authenticamd"):
		return CpuVendorAmd
	case strings.Contains(lower, "genuineintel"):
		return CpuVendorIntel
	default:
		return CpuVendorUnknown
	}
}

func detectNumaNodes() int {
	if runtime.GOOS != "linux" {
		return 1
	}
	entries, err := os.ReadDir("/sys/devices/system/node")
	if err != nil {
		return 1
	}
	n := 0
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "node") {
			n++
		}
	}
	if n < 1 {
		return 1
	}
	return n
}

func detectWSL() bool {
	if runtime.GOOS != "linux" {
		return false
	}
	for _, path := range []string{"/proc/sys/kernel/osrelease", "/proc/version"} {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		lower := strings.ToLower(string(data))
		if strings.Contains(lower, "microsoft") || strings.Contains(lower, "wsl") {
			return true
		}
	}
	return false
}

func detectCgroupMemLimit() *uint64 {
	if runtime.GOOS != "linux" {
		return nil
	}
	if limit := readCgroupV2Limit("/sys/fs/cgroup/memory.max"); limit != nil {
		return limit
	}
	return readCgroupV1Limit("/sys/fs/cgroup/memory/memory.limit_in_bytes")
}

func readCgroupV2Limit(path string) *uint64 {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "max" || trimmed == "" {
		return nil
	}
	n, err := strconv.ParseUint(trimmed, 10, 64)
	if err != nil || n == 0 || n >= ^uint64(0) {
		return nil
	}
	return &n
}

func readCgroupV1Limit(path string) *uint64 {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	n, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil || n == 0 || n >= (1<<60) {
		return nil
	}
	return &n
}

func detectHugepages2MiB() bool {
	if runtime.GOOS != "linux" {
		return false
	}
	data, err := os.ReadFile("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages")
	if err != nil {
		return false
	}
	n, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	return err == nil && n > 0
}

// IsSkylakeSP reports whether the host looks like Intel Skylake-SP (AVX-512 regression gate).
func IsSkylakeSP() bool {
	if runtime.GOOS != "linux" {
		return false
	}
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return false
	}
	lower := strings.ToLower(string(data))
	return strings.Contains(lower, "skylake") && strings.Contains(lower, "xeon")
}
