package model

import (
	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
)

// DFlashConfig mirrors oxidize-core DFlashConfig (HuggingFace / GGUF metadata).
type DFlashConfig struct {
	HiddenSize          int
	NumHiddenLayers     int
	NumTargetLayers     int
	BlockSize           int
	TargetLayerIDs      []int
	MaskTokenID         uint32
	VocabSize           int
	NumAttentionHeads   int
	NumKeyValueHeads    int
	IntermediateSize    int
	RMSNormEps          float32
	RopeTheta           float32
}

// DefaultDFlashConfig returns Qwen3.6-35B-A3B-DFlash defaults.
func DefaultDFlashConfig() DFlashConfig {
	return DFlashConfig{
		HiddenSize:        2048,
		NumHiddenLayers:   8,
		NumTargetLayers:   40,
		BlockSize:         16,
		TargetLayerIDs:    []int{1, 10, 19, 28, 37},
		MaskTokenID:       248070,
		VocabSize:         248320,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  8,
		IntermediateSize:  8192,
		RMSNormEps:        1e-5,
		RopeTheta:         10000.0,
	}
}

// Qwen36_35B_A3B_DFlash is an alias for the default DFlash preset.
func Qwen36_35B_A3B_DFlash() DFlashConfig { return DefaultDFlashConfig() }

// HeadDim returns hidden_size / num_attention_heads.
func (c DFlashConfig) HeadDim() int {
	if c.NumAttentionHeads == 0 {
		return 0
	}
	return c.HiddenSize / c.NumAttentionHeads
}

// KVHeadDim returns the K/V head dimension.
func (c DFlashConfig) KVHeadDim() int { return c.HeadDim() }

// TargetHiddenWidth returns hidden_size * num_target_layers.
func (c DFlashConfig) TargetHiddenWidth() int { return c.HiddenSize * c.NumTargetLayers }

// DFlashConfigFromGGUF builds a DFlashConfig from parsed GGUF metadata and tensor names.
func DFlashConfigFromGGUF(file ggufcore.File) DFlashConfig {
	cfg := DefaultDFlashConfig()
	arch := ggufcore.Architecture(file)
	if arch == "" {
		arch = "dflash-draft"
	}
	archKey := func(suffix string) string { return arch + "." + suffix }
	archU32 := func(suffix string) (uint32, bool) {
		for _, key := range []string{archKey(suffix), "dflash." + suffix, "dflash-draft." + suffix} {
			if v, ok := file.Metadata[key]; ok {
				if n, ok := v.AsUint64(); ok {
					return uint32(n), true
				}
			}
		}
		return 0, false
	}
	archF32 := func(suffix string) (float32, bool) {
		for _, key := range []string{archKey(suffix), "dflash." + suffix, "dflash-draft." + suffix} {
			if v, ok := file.Metadata[key]; ok {
				if f, ok := v.AsFloat32(); ok {
					return f, true
				}
			}
		}
		return 0, false
	}
	if n, ok := archU32("hidden_size"); ok {
		cfg.HiddenSize = int(n)
	} else if n, ok := archU32("embedding_length"); ok {
		cfg.HiddenSize = int(n)
	}
	if n, ok := archU32("num_hidden_layers"); ok {
		cfg.NumHiddenLayers = int(n)
	} else if n, ok := archU32("block_count"); ok {
		cfg.NumHiddenLayers = int(n)
	}
	if n, ok := archU32("block_size"); ok {
		cfg.BlockSize = int(n)
	}
	if n, ok := archU32("mask_token_id"); ok {
		cfg.MaskTokenID = n
	}
	if n, ok := archU32("vocab_size"); ok {
		cfg.VocabSize = int(n)
	} else if n, ok := archU32("n_target_features"); ok {
		cfg.VocabSize = int(n)
	}
	if n, ok := archU32("num_attention_heads"); ok {
		cfg.NumAttentionHeads = int(n)
	} else if n, ok := archU32("attention.head_count"); ok {
		cfg.NumAttentionHeads = int(n)
	}
	if n, ok := archU32("num_key_value_heads"); ok {
		cfg.NumKeyValueHeads = int(n)
	} else if n, ok := archU32("attention.head_count_kv"); ok {
		cfg.NumKeyValueHeads = int(n)
	}
	if n, ok := archU32("intermediate_size"); ok {
		cfg.IntermediateSize = int(n)
	} else if n, ok := archU32("feed_forward_length"); ok {
		cfg.IntermediateSize = int(n)
	}
	if f, ok := archF32("rms_norm_eps"); ok {
		cfg.RMSNormEps = f
	} else if f, ok := archF32("attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = f
	}
	if f, ok := archF32("rope_theta"); ok {
		cfg.RopeTheta = f
	} else if f, ok := archF32("rope.freq_base"); ok {
		cfg.RopeTheta = f
	}

	targetIDs := parseTargetLayerIDs(file, archKey("target_layer_ids"))
	if len(targetIDs) == 0 {
		targetIDs = parseTargetLayerIDs(file, "dflash.target_layer_ids")
	}
	if len(targetIDs) == 0 {
		targetIDs = parseTargetLayerIDs(file, "dflash-draft.target_layer_ids")
	}
	numTarget := len(targetIDs)
	if numTarget == 0 {
		numTarget = numTargetLayersFromFC(file, cfg.HiddenSize)
	}
	if numTarget == 0 {
		if n, ok := archU32("num_target_layers"); ok {
			numTarget = int(n)
		} else {
			numTarget = cfg.NumHiddenLayers
		}
	}
	cfg.NumTargetLayers = numTarget
	if len(targetIDs) > 0 {
		cfg.TargetLayerIDs = append([]int(nil), targetIDs...)
	} else {
		cfg.TargetLayerIDs = make([]int, numTarget)
		for i := range cfg.TargetLayerIDs {
			cfg.TargetLayerIDs[i] = i
		}
	}
	return cfg
}

func parseTargetLayerIDs(file ggufcore.File, key string) []int {
	v, ok := file.Metadata[key]
	if !ok || len(v.Array) == 0 {
		return nil
	}
	out := make([]int, 0, len(v.Array))
	for _, elem := range v.Array {
		if n, ok := elem.AsUint64(); ok {
			out = append(out, int(n))
		}
	}
	return out
}

func numTargetLayersFromFC(file ggufcore.File, hidden int) int {
	if hidden <= 0 {
		return 0
	}
	for _, info := range ggufcore.MappedTensorInfos(file) {
		switch info.Name {
		case "fc.weight", "dflash_fc.weight", "model.fc.weight":
			if len(info.Dimensions) != 2 {
				continue
			}
			inW := int(info.Dimensions[0])
			outW := int(info.Dimensions[1])
			if outW == hidden && inW%hidden == 0 {
				return inW / hidden
			}
		}
	}
	return 0
}

// DFlashFeature identifies implemented DFlash capabilities.
type DFlashFeature string

const (
	DFlashFeatureLoading            DFlashFeature = "loading"
	DFlashFeatureTargetHiddenFusion DFlashFeature = "target-hidden-fusion"
	DFlashFeatureKvCache            DFlashFeature = "kv-cache"
	DFlashFeatureSpeculativeAlgo    DFlashFeature = "speculative-algorithm"
)

// ImplementedDFlashFeatures lists features exposed by the Go port.
func ImplementedDFlashFeatures() []DFlashFeature {
	return []DFlashFeature{
		DFlashFeatureLoading,
		DFlashFeatureTargetHiddenFusion,
		DFlashFeatureKvCache,
		DFlashFeatureSpeculativeAlgo,
	}
}

// DFlashDraftModel is the DFlash draft transformer (oxidize-core dflash.rs).
type DFlashDraftModel struct {
	Config            DFlashConfig
	Stack             *LlamaDecoderStack
	FC                F32Weight
	FCBias            []float32
	HiddenNorm        []float32
	TargetHiddenCache [][]float32
}

// NewDFlashDraftModel constructs an empty draft model with the given config.
func NewDFlashDraftModel(config DFlashConfig) *DFlashDraftModel {
	return &DFlashDraftModel{
		Config: config,
		Stack:  NewLlamaDecoderStack(LlamaDecoderConfigFromDFlash(config)),
	}
}

// Forward implements Model using batched prefill / single-token decode.
func (m *DFlashDraftModel) Forward(tokens []Token, session *Session) (Logits, error) {
	if len(tokens) == 0 {
		return nil, EmptyInputError
	}
	hidden, err := m.ForwardBatch(tokens)
	if err != nil {
		return nil, NewErrorf("%v", err)
	}
	logits, err := m.Logits(hidden)
	if err != nil {
		return nil, NewErrorf("%v", err)
	}
	session.RecordTokens(len(tokens))
	return logits, nil
}

// VocabSize returns configured vocabulary size.
func (m *DFlashDraftModel) VocabSize() int { return m.Config.VocabSize }

// ContextSize returns the default draft context window.
func (m *DFlashDraftModel) ContextSize() int { return 4096 }

// LayerCount returns the number of decoder layers.
func (m *DFlashDraftModel) LayerCount() int { return m.Config.NumHiddenLayers }
