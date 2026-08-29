// Package vulkanbackend mirrors oxidize_core::backends::vulkan. It provides
package vulkanbackend

import (
	"fmt"
	"sync"
)

// BuildInfo mirrors VulkanBuildInfo.
type BuildInfo struct {
	DetectedAtBuild bool
	LoaderPath      string
}

// Info returns the build-time detection result for the Vulkan backend.
func Info() BuildInfo { return BuildInfo{DetectedAtBuild: false} }

// DeviceClass mirrors VulkanDeviceClass.
type DeviceClass uint8

const (
	DeviceIntelArc DeviceClass = iota
	DeviceIntelIntegrated
	DeviceNvidia
	DeviceAmd
	DeviceOther
)

// String returns the canonical device class name.
func (d DeviceClass) String() string {
	switch d {
	case DeviceIntelArc:
		return "intel-arc"
	case DeviceIntelIntegrated:
		return "intel-integrated"
	case DeviceNvidia:
		return "nvidia"
	case DeviceAmd:
		return "amd"
	default:
		return "other"
	}
}

// DeviceInfo mirrors VulkanDeviceInfo.
type DeviceInfo struct {
	VendorID         uint32
	DeviceID         uint32
	DeviceName       string
	DeviceClass      DeviceClass
	ComputeQueueFamily uint32
}

// ClassifyDevice mirrors classify_vulkan_device.
func ClassifyDevice(vendorID, deviceID uint32, name string) DeviceClass {
	if vendorID == 0x8086 { // Intel
		if isLikelyIntelArc(deviceID) {
			return DeviceIntelArc
		}
		return DeviceIntelIntegrated
	}
	if vendorID == 0x10DE {
		return DeviceNvidia
	}
	if vendorID == 0x1002 || vendorID == 0x1022 {
		return DeviceAmd
	}
	return DeviceOther
}

func isLikelyIntelArc(deviceID uint32) bool {
	switch {
	case deviceID >= 0x4900 && deviceID <= 0x4FFF:
		return true
	case deviceID >= 0x5600 && deviceID <= 0x57FF:
		return true
	case deviceID >= 0x7D40 && deviceID <= 0x7D7F:
		return true
	}
	return false
}

// Shader mirrors VulkanShader.
type Shader uint8

const (
	ShaderQ4Q8Gemv Shader = iota
	ShaderFusedAttention
	ShaderLayerDispatch
	ShaderF32Gemm
	ShaderQ4KGemv
)

// String returns the canonical shader name.
func (s Shader) String() string {
	switch s {
	case ShaderQ4Q8Gemv:
		return "q4_q8_gemv"
	case ShaderFusedAttention:
		return "fused_attention"
	case ShaderLayerDispatch:
		return "layer_dispatch"
	case ShaderF32Gemm:
		return "f32_gemm"
	case ShaderQ4KGemv:
		return "q4_k_gemv"
	default:
		return "unknown"
	}
}

// LayerDispatch mirrors VulkanLayerDispatch.
type LayerDispatch struct {
	LayerIndex int
	Shader     Shader
	Workgroups [3]uint32
}

// Q4Q8GemvShader, Q4KGemvShader, F32GemmShader, FusedAttentionShader mirror
const (
	Q4Q8GemvShader      = "// SPIR-V placeholder for Q4_Q8 GEMV\n"
	Q4KGemvShader       = "// SPIR-V placeholder for Q4_K GEMV\n"
	F32GemmShader       = "// SPIR-V placeholder for F32 GEMM\n"
	FusedAttentionShader= "// SPIR-V placeholder for fused attention\n"
)

// CompileShaderSource mirrors compile_shader_source.
func CompileShaderSource(s Shader) (string, error) {
	switch s {
	case ShaderQ4Q8Gemv:
		return Q4Q8GemvShader, nil
	case ShaderQ4KGemv:
		return Q4KGemvShader, nil
	case ShaderF32Gemm:
		return F32GemmShader, nil
	case ShaderFusedAttention:
		return FusedAttentionShader, nil
	default:
		return "", &KernelError{Message: "unknown shader"}
	}
}

// PlanLayerDispatch mirrors plan_layer_dispatch. It returns a slice of
// LayerDispatch plans for the requested layer count and hidden size.
func PlanLayerDispatch(layerCount, hiddenSize int) []LayerDispatch {
	if layerCount <= 0 || hiddenSize <= 0 {
		return nil
	}
	plans := make([]LayerDispatch, 0, layerCount)
	for i := 0; i < layerCount; i++ {
		workgroups := [3]uint32{
			uint32(hiddenSize / 64),
			1,
			1,
		}
		if workgroups[0] == 0 {
			workgroups[0] = 1
		}
		plans = append(plans, LayerDispatch{
			LayerIndex: i,
			Shader:     ShaderFusedAttention,
			Workgroups: workgroups,
		})
	}
	return plans
}

// ShouldUseVulkanGemv mirrors should_use_vulkan_gemv.
func ShouldUseVulkanGemv(rows, cols int) bool {
	return rows*cols >= 4096
}

// ShouldUseVulkanGemm mirrors should_use_vulkan_gemm.
func ShouldUseVulkanGemm(rows, shared, cols int) bool {
	return rows*shared*cols >= 32768
}

// ValidateGemvDims mirrors validate_gemv_dims.
func ValidateGemvDims(rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return &KernelError{Message: "invalid dims"}
	}
	return nil
}

// ValidateGemmDims mirrors validate_gemm_dims.
func ValidateGemmDims(rows, shared, cols int) error {
	if rows <= 0 || shared <= 0 || cols <= 0 {
		return &KernelError{Message: "invalid dims"}
	}
	return nil
}

// KernelError mirrors VulkanKernelError.
type KernelError struct{ Message string }

func (e *KernelError) Error() string { return "vulkan kernel: " + e.Message }

// ensure unused imports are linked
var _ = fmt.Sprintf
var _ = sync.Mutex{}
