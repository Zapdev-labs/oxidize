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
	cols := 4
	rows := 3
	input := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	matrix := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	output := make([]float32, rows)
	w := NewWorkspace(cols)
	if err := FusedRMSNormGEMVF32Transposed(FusedRmsNormGemv{
		Input: input, NormWeight: weight, Eps: 1e-6,
		Matrix: matrix, Rows: rows, Cols: cols,
	}, w, output); err != nil {
		t.Fatalf("err: %v", err)
	}
	if math.Abs(float64(output[0])-1) > 1e-3 {
		t.Fatalf("output[0] = %f", output[0])
	}
}

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	if got := DotProductAvx2OrScalar(a, b); got != 70 {
		t.Fatalf("dot = %f", got)
	}
}
