package vision

import "testing"

func TestStubEncoder(t *testing.T) {
	cfg := DefaultConfig()
	enc := NewStubEncoder(cfg)
	dims := enc.Dims()
	if len(dims) != 3 {
		t.Fatalf("expected 3 dims, got %v", dims)
	}
	out, err := enc.Encode([]float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("expected non-empty encoding")
	}
}

func TestStubPreprocessor(t *testing.T) {
	p := NewStubPreprocessor(DefaultConfig())
	if _, err := p.Process([]byte("img"), ModalityImage); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Process([]byte("x"), ModalityAudio); err == nil {
		t.Fatal("expected unsupported modality")
	}
}

func TestEncodeMultimodal(t *testing.T) {
	enc := NewStubEncoder(DefaultConfig())
	prompt := MultimodalPrompt{Text: "hi", Images: []Vec{{0.1, 0.2, 0.3}}}
	out, err := Encode(prompt, enc)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("expected non-empty embedding")
	}
}
