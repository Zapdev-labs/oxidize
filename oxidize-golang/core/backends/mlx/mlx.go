// Package mlxbackend mirrors oxidize_core::backends::mlx. The MLX backend is
// macOS-only in the Rust crate; in this Go port it is a stub that exposes
// the BuildInfo and KernelError surface.
package mlxbackend

// BuildInfo mirrors MlxBuildInfo.
type BuildInfo struct {
	DetectedAtBuild bool
}

// Info returns the build-time detection result for the MLX backend.
func Info() BuildInfo { return BuildInfo{DetectedAtBuild: false} }

// KernelError mirrors MlxKernelError.
type KernelError struct{ Message string }

func (e *KernelError) Error() string { return "mlx kernel: " + e.Message }

// MlxTensor mirrors MlxTensor (opaque in this stub).
type MlxTensor struct{ ID int }

// MlxWeightStorage mirrors MlxWeightStorage.
type MlxWeightStorage struct{ ID int }

// MlxComputeBackend mirrors MlxComputeBackend (opaque in this stub).
type MlxComputeBackend struct{}
