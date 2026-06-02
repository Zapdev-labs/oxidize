package quantization

import "testing"

func TestQuantizedSize(t *testing.T) {
	cases := []struct {
		t    Type
		n    int
		want int
	}{
		{TypeF32, 1024, 4096},
		{TypeF16, 1024, 2048},
		{TypeQ4_0, 32, 18},
		{TypeQ8_0, 32, 34},
		{TypeQ2_K, 256, 84},
		{TypeQ6_K, 256, 210},
		{TypeNVFP4, 64, 34},
	}
	for _, c := range cases {
		got, err := QuantizedSize(c.t, c.n)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if got != c.want {
			t.Fatalf("%s/%d = %d, want %d", c.t, c.n, got, c.want)
		}
	}
}

func TestQuantizeDequantizeRoundtrip(t *testing.T) {
	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i) * 0.01
	}
	for _, qt := range []Type{TypeF32, TypeF16, TypeQ4_0, TypeQ8_0, TypeQ2_K, TypeQ6_K} {
		size, err := QuantizedSize(qt, len(src))
		if err != nil {
			t.Fatalf("size: %v", err)
		}
		buf := make([]byte, size)
		if err := QuantizeScalar(qt, src, buf, nil); err != nil {
			t.Fatalf("quantize %s: %v", qt, err)
		}
		out := make([]float32, len(src))
		// Dequantize routines take a raw byte buffer.
		if err := DQuantize(qt, buf, out); err != nil {
			t.Fatalf("dequant %s: %v", qt, err)
		}
	}
}

func TestParseType(t *testing.T) {
	for _, name := range []string{"F32", "F16", "Q4_0", "Q6_K", "Q8_K", "NVFP4"} {
		got, err := ParseType(name)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		if got.String() != name {
			t.Fatalf("roundtrip %s -> %s", name, got)
		}
	}
	if _, err := ParseType("nope"); err == nil {
		t.Fatal("expected error for unknown type")
	}
}

func TestFromLLamaFType(t *testing.T) {
	if FromLLamaFType(2) != TypeQ4_0 {
		t.Fatalf("ftype 2 should be Q4_0")
	}
	if FromLLamaFType(99) != TypeUnknown {
		t.Fatalf("ftype 99 should be Unknown")
	}
}

func TestIMatrixImportance(t *testing.T) {
	im := NewIMatrix([]float32{0.5, 2.0, 0})
	if im.ImportanceAt(0) != 0.5 {
		t.Fatalf("importance 0 = %f", im.ImportanceAt(0))
	}
	if im.ImportanceAt(1) != 2.0 {
		t.Fatalf("importance 1 = %f", im.ImportanceAt(1))
	}
	if im.ImportanceAt(2) != 1.0 {
		t.Fatalf("importance 2 = %f (should clamp to 1)", im.ImportanceAt(2))
	}
	if im.ImportanceAt(10) != 1.0 {
		t.Fatal("out-of-range should return 1")
	}
}
