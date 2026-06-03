// Package cpu_kernels mirrors oxidize_core::compute::cpu_kernels. It
// enumerates the fused kernels available on the CPU path and provides a
// scratch workspace helper plus the fused RMSNorm+GEMV kernel.
package cpu_kernels

import (
	"github.com/Zapdev-labs/oxidize/golang/core/simd"
	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
)

// Kernel identifies a fused CPU kernel.
type Kernel uint8

const (
	KernelOperatorFusion Kernel = iota
	KernelWorkspaceReuse
	KernelAvx2
	KernelAvx512
)

// String returns the canonical name of the kernel.
func (k Kernel) String() string {
	switch k {
	case KernelOperatorFusion:
		return "operator_fusion"
	case KernelWorkspaceReuse:
		return "workspace_reuse"
	case KernelAvx2:
		return "avx2"
	case KernelAvx512:
		return "avx512"
	default:
		return "unknown"
	}
}

// ImplementedKernels returns the kernels available in the current build.
func ImplementedKernels() []Kernel {
	k := []Kernel{KernelOperatorFusion, KernelWorkspaceReuse}
	if simd.Preferred() == simd.BackendAvx2 {
		k = append(k, KernelAvx2)
	}
	if simd.Preferred() == simd.BackendAvx512f {
		k = append(k, KernelAvx512)
	}
	return k
}

// Workspace is a scratch buffer used by fused kernels.
type Workspace struct {
	scratch []float32
}

// NewWorkspace returns a workspace with the given capacity.
func NewWorkspace(capacity int) *Workspace {
	return &Workspace{scratch: make([]float32, 0, capacity)}
}

// Get returns a slice of length `n` from the workspace.
func (w *Workspace) Get(n int) []float32 {
	if cap(w.scratch) < n {
		w.scratch = make([]float32, n)
	} else {
		w.scratch = w.scratch[:n]
	}
	return w.scratch
}

// Capacity returns the current capacity of the workspace.
func (w *Workspace) Capacity() int { return cap(w.scratch) }

// FusedError mirrors FusedCpuError.
type FusedError struct{ Message string }

func (e *FusedError) Error() string { return "fused: " + e.Message }

// FusedRmsNormGemv mirrors FusedRmsNormGemv.
type FusedRmsNormGemv struct {
	Input       []float32
	NormWeight  []float32
	Eps         float32
	Matrix      []float32
	Rows, Cols  int
}

// FusedRMSNormGEMVF32Transposed applies RMSNorm to the input, then performs
// a transposed GEMV with the normalized vector. The convention matches
// gemv_f32_transposed: len(input) == rows, len(output) == cols.
func FusedRMSNormGEMVF32Transposed(params FusedRmsNormGemv, workspace *Workspace, output []float32) error {
	normalized := workspace.Get(params.Rows)
	if err := tensor.RMSNormF32(params.Input[:params.Rows], params.NormWeight[:params.Rows], normalized, params.Eps); err != nil {
		return &FusedError{Message: err.Error()}
	}
	return tensor.GemvF32Transposed(params.Matrix, params.Rows, params.Cols, normalized, output)
}

// MatMulReuseWorkspace computes a GEMM into a workspace-backed buffer and
// returns the result slice.
func MatMulReuseWorkspace(workspace *Workspace, left, right []float32, rows, shared, cols int) ([]float32, error) {
	out := workspace.Get(rows * cols)
	if err := tensor.GemmF32(left, right, rows, shared, cols, out); err != nil {
		return nil, &FusedError{Message: err.Error()}
	}
	return out, nil
}

// DotProductAvx2OrScalar is a runtime-dispatched dot product. The Go
// implementation is scalar; the AVX2 path would be plugged in when build
// tags allow.
func DotProductAvx2OrScalar(a, b []float32) float32 {
	if simd.Preferred() == simd.BackendAvx2 || simd.Preferred() == simd.BackendAvx512f {
		// The SIMD path is conceptually available; use the same loop body
		// for now and let the Go compiler auto-vectorize.
	}
	var sum float32
	for i, v := range a {
		sum += v * b[i]
	}
	return sum
}

// DotProductAvx512OrScalar mirrors DotProductAvx2OrScalar.
func DotProductAvx512OrScalar(a, b []float32) float32 {
	return DotProductAvx2OrScalar(a, b)
}
