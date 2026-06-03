package flash_attention

import "testing"

func TestFlashAttentionDecodeGQAWindowLimitsKeys(t *testing.T) {
	headDim := 2
	kvLen := 4
	seqLen := 4
	window := 2
	q := []float32{1, 0}
	k := []float32{
		10, 0, 0, 10,
		0, 10, 10, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
	}
	v := []float32{
		1, 0, 0, 1,
		0, 1, 1, 0,
		2, 0, 0, 2,
		0, 2, 2, 0,
	}
	full := make([]float32, headDim)
	windowed := make([]float32, headDim)
	if err := FlashAttentionDecodeGQA(q, k, v, full, seqLen, headDim, kvLen, 0); err != nil {
		t.Fatal(err)
	}
	if err := FlashAttentionDecodeGQAWindow(q, k, v, windowed, seqLen, headDim, kvLen, 0, window); err != nil {
		t.Fatal(err)
	}
	if full[0] == windowed[0] && full[1] == windowed[1] {
		t.Fatalf("windowed attention should differ from full: full=%v window=%v", full, windowed)
	}
}
