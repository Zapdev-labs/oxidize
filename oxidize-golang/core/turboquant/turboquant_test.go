package turboquant

import (
	"math"
	"testing"
)

func TestQuantizeDequantize(t *testing.T) {
	data := &Data{}
	src := []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8}
	data.QuantizeF32(src, 1, 8, Int8)
	if len(data.Blocks) == 0 {
		t.Fatal("no blocks")
	}
	out := make([]float32, 8)
	data.DequantizeF32(out)
	for i, v := range src {
		if math.Abs(float64(out[i]-v)) > 0.05 {
			t.Fatalf("mismatch at %d: %f vs %f", i, v, out[i])
		}
	}
}

func TestQuantize4Bit(t *testing.T) {
	data := &Data{}
	src := []float32{0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 0.0}
	data.QuantizeF32(src, 1, 8, Int4)
	out := make([]float32, 8)
	data.DequantizeF32(out)
}

func TestGEMV(t *testing.T) {
	data := &Data{}
	src := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
	data.QuantizeF32(src, 4, 4, Int8)
	vec := []float32{1, 0, 0, 0}
	out := make([]float32, 4)
	data.GEMV(vec, out)
	if out[0] == 0 {
		t.Fatal("out[0] should be non-zero")
	}
}
