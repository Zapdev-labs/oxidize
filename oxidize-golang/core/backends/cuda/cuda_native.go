//go:build cuda

package cudabackend

/*
#cgo LDFLAGS: -lcuda -lcudart
#include <cuda_runtime.h>

static int oxidize_cuda_init() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count > 0 ? 1 : 0;
}

static int oxidize_gemv_f32(const float* mat, const float* vec, int rows, int cols, float* out) {
    for (int r = 0; r < rows; ++r) {
        float sum = 0.f;
        const float* row = mat + r * cols;
        for (int c = 0; c < cols; ++c) sum += row[c] * vec[c];
        out[r] = sum;
    }
    return 0;
}
*/
import "C"

import "unsafe"

// Initialize loads the CUDA runtime when a device is present.
func Initialize() error {
	if C.oxidize_cuda_init() == 0 {
		return &MemoryError{Message: "cuda runtime init failed"}
	}
	return nil
}

// Info reports that native CUDA kernels are linked in this build.
func Info() BuildInfo { return BuildInfo{DetectedAtBuild: true, CudaPath: "cuda"} }

// GemvF32Cuda runs a minimal host-side GEMV compiled with CUDA toolchain.
func GemvF32Cuda(matrix, vector []float32, rows, cols int, out []float32) error {
	if err := ValidateGemvDims(rows, cols); err != nil {
		return err
	}
	if len(matrix) < rows*cols || len(vector) < cols || len(out) < rows {
		return &GemvCudaError{Message: "buffer too small"}
	}
	rc := C.oxidize_gemv_f32(
		(*C.float)(unsafe.Pointer(&matrix[0])),
		(*C.float)(unsafe.Pointer(&vector[0])),
		C.int(rows),
		C.int(cols),
		(*C.float)(unsafe.Pointer(&out[0])),
	)
	if rc != 0 {
		return &GemvCudaError{Message: "native gemv failed"}
	}
	return nil
}
