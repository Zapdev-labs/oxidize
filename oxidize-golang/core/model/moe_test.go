package model

import (
	"testing"
)

func TestMoEFFNForwardSynthetic(t *testing.T) {
	h, inter, experts := 4, 8, 2
	moe := MoELayer{
		NumExperts: experts,
		GateInp:    NewF32WeightFromSlice([]float32{1, 0, 0, 1, 0, 1, 1, 0}, experts, h),
		GateExps:   NewF32WeightFromSlice(flatIdentity(experts, inter, h), experts*inter, h),
		UpExps:     NewF32WeightFromSlice(flatIdentity(experts, inter, h), experts*inter, h),
		DownExps:   NewF32WeightFromSlice(flatIdentity(experts, h, inter), experts*h, inter),
	}
	normed := []float32{1, 0, 0, 1}
	out := make([]float32, h)
	if err := moeFFNForward(moe, h, inter, 1, normed, out); err != nil {
		t.Fatal(err)
	}
	if out[0] == 0 && out[1] == 0 && out[2] == 0 && out[3] == 0 {
		t.Fatalf("expected non-zero moe output, got %v", out)
	}
}

func flatIdentity(experts, rows, cols int) []float32 {
	total := experts * rows * cols
	data := make([]float32, total)
	for e := 0; e < experts; e++ {
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				if r == c%rows {
					data[e*rows*cols+r*cols+c] = 1
				}
			}
		}
	}
	return data
}
