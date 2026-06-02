package cpubackend

import (
	"math"
	"testing"
)

func TestCpuBackendAdd(t *testing.T) {
	b := New()
	a, _ := b.TensorFromF32([]float32{1, 2, 3})
	bb, _ := b.TensorFromF32([]float32{4, 5, 6})
	out, err := b.Add(a, bb)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	tc := out.(*CpuTensor)
	if tc.Data[0] != 5 || tc.Data[2] != 9 {
		t.Fatalf("add wrong: %v", tc.Data)
	}
}

func TestCpuBackendRmsNorm(t *testing.T) {
	b := New()
	in, _ := b.TensorFromF32([]float32{1, 2, 3, 4})
	w, _ := b.TensorFromF32([]float32{1, 1, 1, 1})
	out, err := b.RmsNorm(in, w, 1e-6)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	tc := out.(*CpuTensor)
	expected := 1.0 / math.Sqrt((1+4+9+16)/4.0)
	if math.Abs(float64(tc.Data[0])-expected) > 1e-3 {
		t.Fatalf("rms[0] = %f, want %f", tc.Data[0], expected)
	}
}

func TestCpuBackendSoftmax(t *testing.T) {
	b := New()
	x, _ := b.TensorFromF32([]float32{0, 0, 0})
	out, err := b.Softmax(x)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	tc := out.(*CpuTensor)
	for _, v := range tc.Data {
		if math.Abs(float64(v)-1.0/3.0) > 1e-6 {
			t.Fatalf("softmax = %v", tc.Data)
		}
	}
}
