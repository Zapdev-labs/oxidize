package ggufcore

import "testing"

func TestArchitectureEmpty(t *testing.T) {
	if got := Architecture(File{}); got != "" {
		t.Fatalf("architecture = %q", got)
	}
}

func TestQuantizationOfAbsent(t *testing.T) {
	qt, ok := QuantizationOf(File{})
	if ok || qt != 0 {
		t.Fatalf("quantization = %v ok=%v", qt, ok)
	}
}

func TestQuantizedByteSizeF32(t *testing.T) {
	size := quantizedByteSize(TensorInfo{Dimensions: []uint64{2, 3}, GGMLType: 0})
	if size != 24 {
		t.Fatalf("f32 size = %d, want 24", size)
	}
}

func TestQuantizedByteSizeQ4_0(t *testing.T) {
	size := quantizedByteSize(TensorInfo{Dimensions: []uint64{32}, GGMLType: 2})
	if size != quantization.BLOCK_Q4_0_SIZE {
		t.Fatalf("q4_0 size = %d, want %d", size, quantization.BLOCK_Q4_0_SIZE)
	}
}
