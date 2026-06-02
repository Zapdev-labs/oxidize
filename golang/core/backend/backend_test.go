package backend

import (
	"runtime"
	"testing"
)

func TestParseBackendVariants(t *testing.T) {
	cases := []struct {
		in   string
		want Backend
	}{
		{"cpu", BackendCpu},
		{"metal", BackendMetal},
		{"cuda", BackendCuda},
		{"mlx", BackendMlx},
		{"vulkan", BackendVulkan},
		{"intel-arc", BackendIntelArc},
		{"arc", BackendIntelArc},
		{"unknown", 0},
	}
	for _, c := range cases {
		got, err := ParseBackend(c.in)
		if c.in == "unknown" {
			if err == nil {
				t.Fatalf("expected error for %q", c.in)
			}
			continue
		}
		if err != nil {
			t.Fatalf("parse %q: %v", c.in, err)
		}
		if got != c.want {
			t.Fatalf("parse %q = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestBackendString(t *testing.T) {
	values := []Backend{BackendCpu, BackendMetal, BackendCuda, BackendMlx, BackendVulkan, BackendIntelArc}
	for _, b := range values {
		s := b.String()
		if _, err := ParseBackend(s); err != nil {
			t.Fatalf("roundtrip %v -> %q failed: %v", b, s, err)
		}
	}
}

func TestEffective(t *testing.T) {
	if runtime.GOOS != "darwin" {
		got, msg, _ := BackendMlx.Effective()
		if got != BackendCpu {
			t.Fatalf("mlx on linux should fall back to cpu, got %v", got)
		}
		if msg == "" {
			t.Fatal("expected warning when MLX is requested on Linux")
		}
	}
	if got, _, _ := BackendCpu.Effective(); got != BackendCpu {
		t.Fatalf("cpu must always be effective, got %v", got)
	}
}

func TestDTypeSizeInBytes(t *testing.T) {
	cases := map[DType]int{
		DTypeF32: 4, DTypeF16: 2, DTypeI8: 1, DTypeI16: 2, DTypeI32: 4, DTypeI64: 8,
	}
	for d, want := range cases {
		if got := d.SizeInBytes(); got != want {
			t.Fatalf("%v size = %d, want %d", d, got, want)
		}
	}
}
