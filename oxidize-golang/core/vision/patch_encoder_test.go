package vision

import "testing"

func TestPatchEncoderFromBytes(t *testing.T) {
	cfg := ClipBase()
	enc := NewPatchEncoder(cfg)
	raw := make([]byte, 3*cfg.ImageSize*cfg.ImageSize)
	for i := range raw {
		raw[i] = byte(i % 256)
	}
	vec, err := enc.Encode(raw)
	if err != nil {
		t.Fatal(err)
	}
	cols, rows := cfg.Patch()
	want := cols * rows * cfg.HiddenSize
	if len(vec) != want {
		t.Fatalf("len = %d want %d", len(vec), want)
	}
	if vec[0] == 0 && vec[len(vec)-1] == 0 {
		t.Fatal("expected non-zero patch embeddings")
	}
}
