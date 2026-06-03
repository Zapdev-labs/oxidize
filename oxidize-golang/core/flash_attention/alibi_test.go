package flash_attention

import "testing"

func TestFlashAttentionDecodeHeadsGQAAlibi(t *testing.T) {
	headDim := 2
	numHeads := 2
	numKV := 2
	kvLen := numKV * headDim
	seqLen := 2
	q := []float32{1, 0, 0, 1}
	k := []float32{1, 0, 0, 1, 0, 1, 1, 0}
	v := []float32{1, 0, 0, 1, 0, 1, 1, 0}
	out := make([]float32, numHeads*headDim)
	slopes := []float32{0.5, 0.25}
	if err := FlashAttentionDecodeHeadsGQAAlibi(q, k, v, out, seqLen, headDim, kvLen, numHeads, numKV, slopes, 1, 0); err != nil {
		t.Fatal(err)
	}
	plain := make([]float32, len(out))
	if err := FlashAttentionDecodeHeadsGQA(q, k, v, plain, seqLen, headDim, kvLen, numHeads, numKV); err != nil {
		t.Fatal(err)
	}
	if out[0] == plain[0] && out[1] == plain[1] && out[2] == plain[2] && out[3] == plain[3] {
		t.Fatalf("alibi should change output: alibi=%v plain=%v", out, plain)
	}
}
