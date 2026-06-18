package autotune

import (
	"encoding/json"
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/core/gpucluster"
	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
	"github.com/Zapdev-labs/oxidize/golang/core/simd"
)

func TestDetectRuns(t *testing.T) {
	inv := Detect()
	if inv.PhysicalCores < 1 {
		t.Fatalf("physical cores = %d", inv.PhysicalCores)
	}
	if inv.LogicalCores < inv.PhysicalCores {
		t.Fatalf("logical %d < physical %d", inv.LogicalCores, inv.PhysicalCores)
	}
	if inv.NumaNodes < 1 {
		t.Fatalf("numa nodes = %d", inv.NumaNodes)
	}
	s := inv.Summary()
	if s == "" || !contains(s, "cores=") {
		t.Fatalf("summary missing cores: %q", s)
	}
}

func TestKVBytesPerToken(t *testing.T) {
	m := FingerprintFromParts("llama", 32, 4096, 32, 8, 128, 11008, 32000, 8<<30, quantization.TypeQ4_K_M)
	got := KVBytesPerToken(m, 2)
	if got != 131072 {
		t.Fatalf("kv bytes = %d want 131072", got)
	}
}

func TestPerLayerWeightBytes(t *testing.T) {
	m := FingerprintFromParts("llama", 32, 4096, 32, 8, 128, 11008, 32000, 8<<30, quantization.TypeQ4_K_M)
	b := PerLayerWeightBytes(m)
	if b < 200*1024*1024 || b > 260*1024*1024 {
		t.Fatalf("per-layer bytes = %d out of expected range", b)
	}
}

func TestDesktopNoGPU4B(t *testing.T) {
	inv := invDesktop()
	m := modelQwen34B()
	p := Plan(&inv, &m)
	if p.NGPULayers != 0 {
		t.Fatalf("n_gpu_layers = %d want 0", p.NGPULayers)
	}
	if p.Pipeline != PipelineContinuous {
		t.Fatalf("pipeline = %v want Continuous", p.Pipeline)
	}
	if len(p.Rationale) < 5 {
		t.Fatalf("expected rationale entries, got %d", len(p.Rationale))
	}
}

func TestDesktopBigModelLayerWise(t *testing.T) {
	inv := invDesktop()
	inv.TotalRAMBytes = 40 << 30
	m := model70B()
	p := Plan(&inv, &m)
	if !p.LayerWise {
		t.Fatal("expected layer_wise on tight RAM 70B")
	}
	if !p.Mmap || p.Mlock {
		t.Fatal("expected mmap on, mlock off")
	}
}

func TestA10032BFullOffload(t *testing.T) {
	inv := invA100()
	m := modelQwen32B()
	p := Plan(&inv, &m)
	if p.NGPULayers != m.LayerCount {
		t.Fatalf("n_gpu_layers = %d want %d", p.NGPULayers, m.LayerCount)
	}
	if p.Mmap {
		t.Fatal("fully on GPU should disable mmap")
	}
	if p.Pipeline != PipelinePaged {
		t.Fatalf("pipeline = %v want Paged", p.Pipeline)
	}
}

func TestOverridesFromPlan(t *testing.T) {
	inv := invDesktop()
	m := modelQwen34B()
	p := Plan(&inv, &m)
	o := OverridesFromPlan(&p)
	if o.Threads == nil || o.CtxSize == nil || o.NGPULayers == nil {
		t.Fatal("expected override fields")
	}
}

func TestPlanSummaryNonempty(t *testing.T) {
	inv := invDesktop()
	m := modelQwen34B()
	p := Plan(&inv, &m)
	s := p.Summary()
	if !contains(s, "threads") || !contains(s, "Rationale") {
		t.Fatalf("summary missing fields: %q", s)
	}
}

func TestPlanJSONRoundtrip(t *testing.T) {
	inv := invDesktop()
	m := modelQwen34B()
	p := Plan(&inv, &m)
	data, err := json.Marshal(ToPlanJSON(&p))
	if err != nil {
		t.Fatal(err)
	}
	if len(data) < 20 {
		t.Fatalf("json too short: %s", data)
	}
}

func invDesktop() HardwareInventory {
	return HardwareInventory{
		OS:              OsLinux,
		CPUVendor:       CpuVendorAmd,
		SIMD:            simd.BackendAvx2,
		PhysicalCores:   16,
		LogicalCores:    32,
		NumaNodes:       2,
		MinNodeRAMBytes: 32 << 30,
		TotalRAMBytes:   64 << 30,
	}
}

func invA100() HardwareInventory {
	inv := invDesktop()
	inv.PhysicalCores = 32
	inv.LogicalCores = 128
	inv.TotalRAMBytes = 256 << 30
	fam := gpucluster.A100
	inv.HasGPU = true
	inv.GPUFamily = &fam
	inv.GPUVRAMBytes = 80 << 30
	inv.HasCUDA = true
	return inv
}

func modelQwen34B() ModelFingerprint {
	return FingerprintFromParts("qwen2", 36, 2560, 20, 8, 128, 6912, 151936, 2_500_000_000, quantization.TypeQ4_K_M)
}

func modelQwen32B() ModelFingerprint {
	return FingerprintFromParts("qwen2", 64, 5120, 40, 8, 128, 13824, 151936, 20_000_000_000, quantization.TypeQ4_K_M)
}

func model70B() ModelFingerprint {
	return FingerprintFromParts("llama", 80, 8192, 64, 8, 128, 28672, 32000, 40_000_000_000, quantization.TypeQ4_K_M)
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(sub) == 0 || indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
