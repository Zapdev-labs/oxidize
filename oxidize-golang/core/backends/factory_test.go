package backends

import (
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/core/backend"
)

func TestNewComputeBackendCPU(t *testing.T) {
	res, err := NewComputeBackend("cpu", true)
	if err != nil {
		t.Fatal(err)
	}
	if res.Effective != backend.BackendCpu || res.FellBack {
		t.Fatalf("unexpected result: %+v", res)
	}
	if res.Backend == nil || res.Backend.Name() != "cpu" {
		t.Fatalf("backend = %v", res.Backend)
	}
}

func TestNewComputeBackendCudaFallback(t *testing.T) {
	res, err := NewComputeBackend("cuda", true)
	if err != nil {
		t.Fatal(err)
	}
	if !res.FellBack || res.Effective != backend.BackendCpu {
		t.Fatalf("expected cuda->cpu fallback, got %+v", res)
	}
	if res.Warning == "" {
		t.Fatal("expected warning")
	}
}

func TestNewComputeBackendCudaNoFallback(t *testing.T) {
	_, err := NewComputeBackend("cuda", false)
	if err == nil {
		t.Fatal("expected error without fallback")
	}
}
