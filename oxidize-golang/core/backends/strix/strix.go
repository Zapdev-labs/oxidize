// Package strixbackend mirrors oxidize_core::backends::strix. The Strix
// profile selects between CPU, Vulkan, and Hybrid modes for AMD Strix
// hardware.
package strixbackend

// Mode mirrors StrixMode.
type Mode uint8

const (
	ModeCPU Mode = iota
	ModeVulkan
	ModeHybrid
)

// String returns the canonical mode name.
func (m Mode) String() string {
	switch m {
	case ModeCPU:
		return "cpu"
	case ModeVulkan:
		return "vulkan"
	case ModeHybrid:
		return "hybrid"
	default:
		return "unknown"
	}
}

// Profile mirrors StrixProfile.
type Profile struct {
	Mode         Mode
	LazyLoading  bool
	RDNA35Tuning bool
}

// DefaultProfile returns the default Strix profile.
func DefaultProfile() Profile {
	return Profile{Mode: ModeHybrid, LazyLoading: true, RDNA35Tuning: true}
}

// DetectMode returns the recommended mode for the current host. In this
// Go port we default to CPU since we have no RDNA detection.
func DetectMode() Mode { return ModeCPU }

// ShouldLazyLoadLayer mirrors should_lazy_load_layer.
func ShouldLazyLoadLayer(layerIndex, residentLayers int) bool {
	return layerIndex >= residentLayers
}

// Rdna35WorkgroupSize mirrors rdna35_workgroup_size. Larger hidden sizes
// prefer 256-wide; medium 128; small 64.
func Rdna35WorkgroupSize(hiddenSize int) uint32 {
	switch {
	case hiddenSize >= 8192:
		return 256
	case hiddenSize >= 2048:
		return 128
	default:
		return 64
	}
}
