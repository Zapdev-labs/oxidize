package workspace

import "testing"

func TestHealthIsReady(t *testing.T) {
	if got := Health(); got.Status != "ready" {
		t.Fatalf("status = %q", got.Status)
	}
}

func TestBenchmarkInputIsReady(t *testing.T) {
	if got := BenchmarkInput(); got.Status != "ready" {
		t.Fatalf("status = %q", got.Status)
	}
}

func TestWasmStatusMatches(t *testing.T) {
	if got := WasmStatus(); got != "ready" {
		t.Fatalf("status = %q", got)
	}
}
