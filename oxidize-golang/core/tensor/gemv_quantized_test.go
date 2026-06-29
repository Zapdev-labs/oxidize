package tensor

import (
	"math"
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
)

func TestGemvQuantizedDispatchQ4K(t *testing.T) {
	const blocks = 2
	cols := blocks * quantization.QK_K
	rows := 9 // exercises both the x4 unroll and the scalar tail

	rowBytes := blocks * quantization.BLOCK_Q4_K_SIZE
	qbytes := make([]byte, rows*rowBytes)
	wDeq := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		w := make([]float32, cols)
		for i := range w {
			w[i] = float32(math.Sin(float64(i+r*11)*0.09)) * 1.2
		}
		if err := quantization.QuantizeScalar(quantization.TypeQ4_K_M, w, qbytes[r*rowBytes:(r+1)*rowBytes], nil); err != nil {
			t.Fatal(err)
		}
		if err := quantization.DequantQ4_K(qbytes[r*rowBytes:(r+1)*rowBytes], wDeq[r*cols:(r+1)*cols]); err != nil {
			t.Fatal(err)
		}
	}

	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = float32(math.Cos(float64(i) * 0.04))
	}

	out := make([]float32, rows)
	if err := GemvQuantizedDispatch(qbytes, quantization.TypeQ4_K_M, rows, cols, vec, out); err != nil {
		t.Fatal(err)
	}

	// Reference: dequant weights (f32) dot dequant(quant(vec)) — the dispatch
	// quantizes vec to Q8_K, so compare against that same approximation.
	q8 := make([]byte, blocks*quantization.BLOCK_Q8_K_SIZE)
	if err := quantization.QuantizeVectorQ8KInto(vec, blocks, q8); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < rows; r++ {
		want := quantization.Q4KQ8KRowDot(qbytes[r*rowBytes:(r+1)*rowBytes], blocks, q8)
		if math.Abs(float64(out[r]-want)) > 1e-3 {
			t.Fatalf("row %d: dispatch=%v want=%v", r, out[r], want)
		}
	}
}
