package flash_attention

import (
	"math"
	"testing"
)

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	if got := DotProductF32(a, b); got != 70 {
		t.Fatalf("dot = %f", got)
	}
}

func TestFlashAttentionDecode(t *testing.T) {
	headDim := 4
	seqLen := 3
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	query := []float32{1, 0, 0, 0}
	keys := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	values := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	output := make([]float32, headDim)
	if err := FlashAttentionDecodeF32(query, keys, values, output, seqLen, headDim, scale); err != nil {
		t.Fatalf("err: %v", err)
	}
	if output[0] == 0 {
		t.Fatal("output[0] should be non-zero")
	}
}

func TestFlashAttentionPrefill(t *testing.T) {
	headDim := 4
	seqLen := 2
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	queries := []float32{1, 0, 0, 0, 0, 1, 0, 0}
	keys := []float32{1, 0, 0, 0, 0, 1, 0, 0}
	values := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	output := make([]float32, seqLen*headDim)
	if err := FlashAttentionPrefillF32(queries, keys, values, output, seqLen, headDim, scale); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestFlashAttentionDecodeHeads(t *testing.T) {
	headDim := 2
	seqLen := 2
	headCount := 2
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	queries := []float32{1, 0, 0, 1}
	keys := []float32{1, 0, 0, 1, 0, 1, 1, 0}
	values := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	output := make([]float32, headCount*headDim)
	if err := FlashAttentionDecodeHeadsF32(queries, keys, values, output, headCount, seqLen, headDim, scale); err != nil {
		t.Fatalf("err: %v", err)
	}
}
