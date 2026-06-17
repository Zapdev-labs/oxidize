//go:build !cuda

package cudabackend

// Initialize probes for an NVIDIA GPU via nvidia-smi.
func Initialize() error {
	if gpuPresent() {
		return nil
	}
	return &MemoryError{Message: "no NVIDIA GPU detected (nvidia-smi)"}
}

// Info returns build-time CUDA detection (native kernels require -tags=cuda).
func Info() BuildInfo { return BuildInfo{DetectedAtBuild: false, CudaPath: ""} }

// GemvF32Cuda falls back to host GEMV when CUDA is not linked.
func GemvF32Cuda(matrix, vector []float32, rows, cols int, out []float32) error {
	return &GemvCudaError{Message: "cuda native GEMV not linked; build with -tags=cuda"}
}
