// Package webgpubackend mirrors oxidize_core::backends::webgpu. The WebGPU
// backend is a stub in this Go port that exposes BuildInfo and selector
// helpers.
package webgpubackend

// BuildInfo mirrors WebGpuBuildInfo.
type BuildInfo struct {
	DetectedAtBuild bool
}

// Info returns the build-time detection result for the WebGPU backend.
func Info() BuildInfo { return BuildInfo{DetectedAtBuild: false} }

// KernelError mirrors WebGpuKernelError.
type KernelError struct{ Message string }

func (e *KernelError) Error() string { return "webgpu kernel: " + e.Message }

// ShouldUseWebGpuGemv mirrors should_use_webgpu_gemv.
func ShouldUseWebGpuGemv(rows, cols int) bool { return rows*cols >= 4096 }

// ShouldUseWebGpuGemm mirrors should_use_webgpu_gemm.
func ShouldUseWebGpuGemm(rows, shared, cols int) bool { return rows*shared*cols >= 32768 }

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
