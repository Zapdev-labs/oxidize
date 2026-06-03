// Package cudabackend mirrors oxidize_core::backends::cuda. The CUDA backend
// is a stub in this build (no CUDA runtime is linked in Go); the package
// still exposes the BuildInfo, MemoryDevice, and validation helpers so that
// callers can probe for CUDA support at runtime.
package cudabackend

import "fmt"

// BuildInfo mirrors CudaBuildInfo.
type BuildInfo struct {
	DetectedAtBuild bool
	CudaPath        string
}

// Info returns the build-time detection result for the CUDA backend.
func Info() BuildInfo { return BuildInfo{DetectedAtBuild: false, CudaPath: ""} }

// MemoryDevice mirrors MemoryDevice.
type MemoryDevice uint8

const (
	MemoryCpu MemoryDevice = iota
	MemoryCuda
)

// String returns the canonical name of the memory device.
func (m MemoryDevice) String() string {
	switch m {
	case MemoryCpu:
		return "cpu"
	case MemoryCuda:
		return "cuda"
	default:
		return fmt.Sprintf("memory(%d)", uint8(m))
	}
}

// MemoryError mirrors MemoryError.
type MemoryError struct{ Message string }

func (e *MemoryError) Error() string { return "cuda memory: " + e.Message }

// Initialize is a stub. A real implementation would load the CUDA runtime.
func Initialize() error { return &MemoryError{Message: "cuda backend not linked in this build"} }

// GemvCudaError mirrors GemvCudaError.
type GemvCudaError struct{ Message string }

func (e *GemvCudaError) Error() string { return "cuda gemv: " + e.Message }

// GemmCudaError mirrors GemmCudaError.
type GemmCudaError struct{ Message string }

func (e *GemmCudaError) Error() string { return "cuda gemm: " + e.Message }

// GemvF32Cuda is a stub.
func GemvF32Cuda(_, _ []float32, _, _ int, _, _ []float32) error {
	return &GemvCudaError{Message: "cuda backend not linked"}
}

// GemmF32Cuda is a stub.
func GemmF32Cuda(_, _ []float32, _, _, _ int, _ []float32) error {
	return &GemmCudaError{Message: "cuda backend not linked"}
}

// GemvQuantizedCuda is a stub.
func GemvQuantizedCuda(_ []byte, _ int, _ []float32, _, _ int, _, _ []float32) error {
	return &GemvCudaError{Message: "cuda backend not linked"}
}

// ValidateGemvDims mirrors validate_gemv_dims.
func ValidateGemvDims(rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return &GemvCudaError{Message: fmt.Sprintf("invalid dims rows=%d cols=%d", rows, cols)}
	}
	return nil
}

// ValidateQ8_0GemvDims mirrors validate_q8_0_gemv_dims.
func ValidateQ8_0GemvDims(rows, cols int) error {
	if cols%32 != 0 {
		return &GemvCudaError{Message: "Q8_0 GEMV requires cols to be a multiple of 32"}
	}
	return ValidateGemvDims(rows, cols)
}

// ValidateGemmDims mirrors validate_gemm_dims.
func ValidateGemmDims(rows, shared, cols int) error {
	if rows <= 0 || shared <= 0 || cols <= 0 {
		return &GemmCudaError{Message: "invalid dims"}
	}
	return nil
}
