package backends

import (
	"testing"

	cudabackend "github.com/Zapdev-labs/oxidize/golang/core/backends/cuda"

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

func TestNewComputeBackendCuda(t *testing.T) {
	res, err := NewComputeBackend("cuda", true)
	if err != nil {
		t.Fatal(err)
	}
	if res.Requested != backend.BackendCuda {
		t.Fatalf("requested = %v", res.Requested)
	}
	if res.FellBack {
		if res.Effective != backend.BackendCpu {
			t.Fatalf("expected cpu fallback, got %+v", res)
		}
		if res.Warning == "" {
			t.Fatal("expected warning on fallback")
		}
		return
	}
	if res.Backend == nil || res.Backend.Name() != "cuda" {
		t.Fatalf("backend = %v", res.Backend)
	}
}

func TestNewComputeBackendCudaNoFallback(t *testing.T) {
	if err := cudabackend.Initialize(); err != nil {
		t.Skip("cuda unavailable in this environment")
	}
	res, err := NewComputeBackend("cuda", false)
	if err != nil {
		t.Fatal(err)
	}
	if res.Backend.Name() != "cuda" {
		t.Fatalf("backend = %s", res.Backend.Name())
	}
}
