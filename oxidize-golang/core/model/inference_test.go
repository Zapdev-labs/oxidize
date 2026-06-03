package model

import (
	"testing"
)

func TestInferenceForwardWithSyntheticWeights(t *testing.T) {
	cfg := InferenceConfig{
		Architecture:      ArchLlamaModel,
		LayerCount:        0,
		HiddenSize:        2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		VocabSize:         2,
		ContextSize:       16,
		RMSNormEps:        1e-6,
		RopeTheta:         10000,
	}
	stack := NewLlamaDecoderStack(LlamaDecoderConfigFromInference(cfg))
	stack.TokEmbeddings = NewF32WeightFromSlice([]float32{1, 0, 0, 1}, 2, 2)
	stack.Output = NewF32WeightFromSlice([]float32{1, 0, 0, 1}, 2, 2)
	stack.Norm = []float32{1, 1}

	m := NewInferenceModel(cfg, WeightStorage{}, stack)
	logits, err := m.Forward([]Token{0}, NewSession())
	if err != nil {
		t.Fatal(err)
	}
	if len(logits) != 2 {
		t.Fatalf("logits len = %d", len(logits))
	}
	if logits[0] == 0 && logits[1] == 0 {
		t.Fatalf("expected non-zero logits, got %v", logits)
	}
}
