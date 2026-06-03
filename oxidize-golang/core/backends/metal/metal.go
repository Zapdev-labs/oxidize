// Package metalbackend mirrors oxidize_core::backends::metal. The Metal
// backend is macOS-only in Rust; in this Go port the package is purely a
// stub that exposes the BuildInfo and selector helpers. A real build could
// swap the stubs for a CGo-based implementation.
package metalbackend

import "fmt"

// BuildInfo mirrors MetalBuildInfo.
type BuildInfo struct {
	DetectedAtBuild bool
}

// Info returns the build-time detection result for the Metal backend.
func Info() BuildInfo { return BuildInfo{DetectedAtBuild: false} }

// Constants mirror the Rust module-level constants.
const (
	GemvKernelName     = "gemv_f32"
	GemvQ8_0KernelName = "gemv_q8_0"
	GemvMpsMinWorkItems = 4096
	GemmMpsMinWorkItems = 65536
)

// GemvMslSource is the (placeholder) MSL source used to drive MPS. In the
// Rust crate this is a multi-hundred-line string literal; we keep the
// declaration so callers can probe for its presence.
const GemvMslSource = "// Metal MSL source placeholder (Metal backend not linked in this Go build).\n"

// KernelError mirrors MetalKernelError.
type KernelError struct{ Message string }

func (e *KernelError) Error() string { return "metal kernel: " + e.Message }

// ShouldUseMpsGemv mirrors should_use_mps_gemv.
func ShouldUseMpsGemv(rows, cols int) bool {
	return rows*cols >= GemvMpsMinWorkItems
}

// ShouldUseMpsGemm mirrors should_use_mps_gemm.
func ShouldUseMpsGemm(rows, shared, cols int) bool {
	return rows*shared*cols >= GemmMpsMinWorkItems
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

// UnifiedMemoryError mirrors UnifiedMemoryError.
type UnifiedMemoryError struct{ Message string }

func (e *UnifiedMemoryError) Error() string { return "unified memory: " + e.Message }

// UnifiedMemoryStats mirrors UnifiedMemoryStats.
type UnifiedMemoryStats struct {
	TotalBytes uint64
	FreeBytes  uint64
}

// UnifiedBuffer mirrors UnifiedBuffer.
type UnifiedBuffer struct {
	SizeBytes int
	Ptr       uintptr
}

// UnifiedBufferManager mirrors UnifiedBufferManager.
type UnifiedBufferManager struct{}

func (m *UnifiedBufferManager) Allocate(size int) (UnifiedBuffer, error) {
	if size <= 0 {
		return UnifiedBuffer{}, fmt.Errorf("invalid size")
	}
	return UnifiedBuffer{SizeBytes: size}, nil
}
