package model

import (
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/internal/gguf"
)

func TestDFlashConfigFromGGUF(t *testing.T) {
	file := gguf.File{
		Metadata: map[string]gguf.MetadataValue{
			"general.architecture": {Type: gguf.MetadataString, String: "dflash-draft"},
			"dflash.hidden_size":   {Type: gguf.MetadataUint32, Uint64: 4096},
			"dflash.block_size":    {Type: gguf.MetadataUint32, Uint64: 8},
			"dflash.target_layer_ids": {
				Type: gguf.MetadataArray,
				Array: []gguf.MetadataValue{
					{Type: gguf.MetadataUint32, Uint64: 2},
					{Type: gguf.MetadataUint32, Uint64: 4},
				},
			},
		},
	}
	cfg := DFlashConfigFromGGUF(file)
	if cfg.HiddenSize != 4096 {
		t.Fatalf("hidden_size = %d, want 4096", cfg.HiddenSize)
	}
	if cfg.BlockSize != 8 {
		t.Fatalf("block_size = %d, want 8", cfg.BlockSize)
	}
	if len(cfg.TargetLayerIDs) != 2 || cfg.NumTargetLayers != 2 {
		t.Fatalf("target layers = %v (%d), want 2 ids", cfg.TargetLayerIDs, cfg.NumTargetLayers)
	}
}

func TestDFlashConfigDefaults(t *testing.T) {
	cfg := DefaultDFlashConfig()
	if cfg.HiddenSize != 2048 || cfg.NumHiddenLayers != 8 || cfg.BlockSize != 16 {
		t.Fatalf("unexpected defaults: %+v", cfg)
	}
	if cfg.MaskTokenID != 248070 {
		t.Fatalf("mask_token_id = %d", cfg.MaskTokenID)
	}
}

func TestDFlashDraftModelCreation(t *testing.T) {
	cfg := DefaultDFlashConfig()
	m := NewDFlashDraftModel(cfg)
	if len(m.Stack.Layers) != 0 {
		t.Fatalf("expected no layers, got %d", len(m.Stack.Layers))
	}
	if m.Config.VocabSize != 248320 {
		t.Fatalf("vocab_size = %d", m.Config.VocabSize)
	}
}

func TestDFlashBorrowedIOEmbeddingOutputWidths(t *testing.T) {
	m := NewDFlashDraftModel(DFlashConfig{
		HiddenSize:        4,
		NumHiddenLayers:   0,
		NumTargetLayers:   0,
		BlockSize:         1,
		TargetLayerIDs:    nil,
		MaskTokenID:       0,
		VocabSize:         2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		IntermediateSize:  4,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
	})
	m.Stack.TokEmbeddings = NewF32WeightFromSlice([]float32{1, 2, 3, 4}, 2, 2)
	m.Stack.Output = NewF32WeightFromSlice([]float32{1, 0, 0, 1}, 2, 2)
	m.Stack.Norm = []float32{1, 1, 1, 1}

	hidden := make([]float32, 4)
	if err := m.Stack.fillTokenEmbedding(0, hidden); err != nil {
		t.Fatal(err)
	}
	if hidden[0] != 1 || hidden[1] != 2 {
		t.Fatalf("embedding prefix = %v", hidden)
	}
	logits, err := m.Logits(hidden)
	if err != nil {
		t.Fatal(err)
	}
	if logits[0] != 1 || logits[1] != 2 {
		t.Fatalf("logits = %v", logits)
	}
}

func TestDFlashTargetHiddenDoesNotResidualAdd(t *testing.T) {
	cfg := DFlashConfig{
		HiddenSize:        2,
		NumHiddenLayers:   0,
		NumTargetLayers:   1,
		BlockSize:         1,
		TargetLayerIDs:    []int{0},
		MaskTokenID:       0,
		VocabSize:         1,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		IntermediateSize:  2,
		RMSNormEps:        1e-6,
		RopeTheta:         10000,
	}
	m := NewDFlashDraftModel(cfg)
	m.Stack.TokEmbeddings = NewF32WeightFromSlice([]float32{1, 1}, 1, 2)
	m.FC = NewF32WeightFromSlice([]float32{10, 0, 0, 0}, 2, 2)
	m.HiddenNorm = []float32{1, 1}
	m.Stack.Norm = []float32{1, 1}

	withCtx, err := m.ForwardToken(0, []float32{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	m.ResetCache()
	withoutCtx, err := m.ForwardToken(0, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range withCtx {
		if withCtx[i] != withoutCtx[i] {
			t.Fatalf("target hidden changed noise path: with=%v without=%v", withCtx, withoutCtx)
		}
	}
}

func TestDFlashForwardModelInterface(t *testing.T) {
	cfg := DFlashConfig{
		HiddenSize:        2,
		NumHiddenLayers:   0,
		VocabSize:         2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		RMSNormEps:        1e-6,
		RopeTheta:         10000,
	}
	m := NewDFlashDraftModel(cfg)
	m.Stack.TokEmbeddings = NewF32WeightFromSlice([]float32{1, 0, 0, 1}, 2, 2)
	m.Stack.Output = NewF32WeightFromSlice([]float32{1, 0, 0, 1}, 2, 2)
	m.Stack.Norm = []float32{1, 1}

	logits, err := m.Forward([]Token{0}, NewSession())
	if err != nil {
		t.Fatal(err)
	}
	if len(logits) != 2 {
		t.Fatalf("logits len = %d", len(logits))
	}
}
