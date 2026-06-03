package cpu_kernels

import (
	"math"
	"testing"
)

func TestImplementedKernels(t *testing.T) {
	k := ImplementedKernels()
	if len(k) == 0 {
		t.Fatal("kernels should not be empty")
	}
}

func TestWorkspace(t *testing.T) {
	w := NewWorkspace(64)
	buf := w.Get(32)
	if len(buf) != 32 {
		t.Fatalf("len = %d", len(buf))
	}
	if w.Capacity() < 32 {
		t.Fatalf("capacity = %d", w.Capacity())
	}
}

func TestFusedRMSNormGEMV(t *testing.T) {
	// Convention (mirrors oxidize_core::compute::cpu_kernels::fused_rms_norm_gemv_f32_transposed):
	//   - input/normalized length = rows
	//   - output length = cols
	//   - matrix is rows*cols row-major
	// output[c] = sum_r matrix[r][c] * normalized[r]
	rows := 4
	cols := 3
	input := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	// 4x3 matrix that picks out normalized[0] in the first column.
	matrix := []float32{
		1, 0, 0,
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
	}
	output := make([]float32, cols)
	w := NewWorkspace(rows)
	if err := FusedRMSNormGEMVF32Transposed(FusedRmsNormGemv{
		Input: input, NormWeight: weight, Eps: 1e-6,
		Matrix: matrix, Rows: rows, Cols: cols,
	}, w, output); err != nil {
		t.Fatalf("err: %v", err)
	}
	// normalized[0] = input[0] / sqrt(mean(input^2) + eps) = 1 / sqrt(7.5+1e-6)
	const mean = (1.0 + 4.0 + 9.0 + 16.0) / 4.0
	want := 1.0 / float32(math.Sqrt(mean))
	if math.Abs(float64(output[0]-want)) > 1e-3 {
		t.Fatalf("output[0] = %f, want %f", output[0], want)
	}
}

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	if got := DotProductAvx2OrScalar(a, b); got != 70 {
		t.Fatalf("dot = %f", got)
	}
}
