package webgpubackend

import "testing"

func TestBuildInfo(t *testing.T) {
	if Info().DetectedAtBuild {
		t.Fatal("stub: webgpu should not be detected")
	}
}

func TestShouldUseWebGPU(t *testing.T) {
	if !ShouldUseWebGpuGemv(100, 100) {
		t.Fatal("should use webgpu for large gemv")
	}
	if ShouldUseWebGpuGemv(1, 1) {
		t.Fatal("should not use webgpu for small gemv")
	}
}
