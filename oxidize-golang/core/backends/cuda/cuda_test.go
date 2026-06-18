package cudabackend

import "testing"

func TestBuildInfo(t *testing.T) {
	info := Info()
	if info.DetectedAtBuild && info.CudaPath == "" {
		t.Fatal("native cuda build should set CudaPath")
	}
}

func TestValidateGemvDims(t *testing.T) {
	if err := ValidateGemvDims(0, 0); err == nil {
		t.Fatal("expected error for zero dims")
	}
	if err := ValidateGemvDims(1, 1); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestValidateQ8_0GemvDims(t *testing.T) {
	if err := ValidateQ8_0GemvDims(1, 33); err == nil {
		t.Fatal("expected misalignment error")
	}
	if err := ValidateQ8_0GemvDims(1, 32); err != nil {
		t.Fatalf("err: %v", err)
	}
}
