package model

import "testing"

func TestApplyTokenEmbeddingDimsPreservesQwen3Metadata(t *testing.T) {
	cfg := InferenceConfig{HiddenSize: 2560, VocabSize: 151936}
	applyTokenEmbeddingDims(&cfg, []uint64{2560, 151936})
	if cfg.HiddenSize != 2560 {
		t.Fatalf("hidden_size=%d want 2560", cfg.HiddenSize)
	}
	if cfg.VocabSize != 151936 {
		t.Fatalf("vocab_size=%d want 151936", cfg.VocabSize)
	}
}

func TestApplyTokenEmbeddingDimsFillsMissingFromGGUFOrder(t *testing.T) {
	cfg := InferenceConfig{}
	applyTokenEmbeddingDims(&cfg, []uint64{2560, 151936})
	if cfg.HiddenSize != 2560 {
		t.Fatalf("hidden_size=%d want 2560", cfg.HiddenSize)
	}
	if cfg.VocabSize != 151936 {
		t.Fatalf("vocab_size=%d want 151936", cfg.VocabSize)
	}
}

func TestGemmaDecoderConfigLayerPattern(t *testing.T) {
	cfg := LlamaDecoderConfig{
		SlidingWindow:        1024,
		SlidingWindowPattern: 6,
		RopeTheta:            1000000,
		RopeThetaSWA:         10000,
	}
	for l := 0; l < 5; l++ {
		if cfg.LayerIsGlobal(l) {
			t.Fatalf("layer %d should be local", l)
		}
		if cfg.LayerRopeTheta(l) != 10000 {
			t.Fatalf("layer %d rope=%v want 10000", l, cfg.LayerRopeTheta(l))
		}
		if cfg.LayerSlidingWindow(l) != 1024 {
			t.Fatalf("layer %d window=%d want 1024", l, cfg.LayerSlidingWindow(l))
		}
	}
	if !cfg.LayerIsGlobal(5) {
		t.Fatalf("layer 5 should be global")
	}
	if cfg.LayerRopeTheta(5) != 1000000 {
		t.Fatalf("layer 5 rope=%v want 1000000", cfg.LayerRopeTheta(5))
	}
	if cfg.LayerSlidingWindow(5) != 0 {
		t.Fatalf("layer 5 window=%d want 0 (global)", cfg.LayerSlidingWindow(5))
	}
}

func TestMistralUniformWindowStillApplies(t *testing.T) {
	// No interleaving pattern: every layer uses the window (Mistral-style).
	cfg := LlamaDecoderConfig{SlidingWindow: 4096}
	for l := 0; l < 4; l++ {
		if cfg.LayerIsGlobal(l) {
			t.Fatalf("layer %d should be local (uniform window)", l)
		}
		if cfg.LayerSlidingWindow(l) != 4096 {
			t.Fatalf("layer %d window=%d want 4096", l, cfg.LayerSlidingWindow(l))
		}
	}
}
