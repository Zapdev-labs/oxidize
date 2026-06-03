package model

import (
	"fmt"
	"strconv"
	"sync"

	"github.com/Zapdev-labs/oxidize/golang/core/backend"
	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
	"github.com/Zapdev-labs/oxidize/golang/core/kv_cache"
)

// KVCacheDType mirrors the CLI's KvCacheDType enum.
type KVCacheDType string

const (
	KVCacheF32 KVCacheDType = "f32"
	KVCacheF16 KVCacheDType = "f16"
	KVCacheQ8  KVCacheDType = "q8"
	KVCacheQ4  KVCacheDType = "q4"
)

// MapDType converts a CLI KvCacheDType into the backend DType.
func (k KVCacheDType) MapDType() backend.DType {
	switch k {
	case KVCacheF16:
		return backend.DTypeF16
	case KVCacheQ8, KVCacheQ4:
		return backend.DTypeI8
	default:
		return backend.DTypeF32
	}
}

// InferenceConfig mirrors the Rust `InferenceConfig` struct.
type InferenceConfig struct {
	VocabSize           int
	ContextSize         int
	LayerCount          int
	HiddenSize          int
	IntermediateSize    int
	NumAttentionHeads   int
	NumKeyValueHeads    int
	KeyValueHeadDim     int
	KVCacheDType        backend.DType
	KVQuantization      kv_cache.Quantization
	RMSNormEps          float32
	RopeTheta           float32
	Architecture        Architecture
	SlidingWindow       int
	NumExperts          int
	NumExpertsPerToken  int
	AlibiNumHeads       int
}

// DefaultInferenceConfig returns sensible defaults.
func DefaultInferenceConfig() InferenceConfig {
	return InferenceConfig{
		VocabSize:          32000,
		ContextSize:        2048,
		LayerCount:         32,
		HiddenSize:         4096,
		IntermediateSize:   11008,
		NumAttentionHeads:  32,
		NumKeyValueHeads:   32,
		KeyValueHeadDim:    128,
		KVCacheDType:       backend.DTypeF32,
		KVQuantization:     kv_cache.QuantAsymmetric,
		RMSNormEps:         1e-5,
		RopeTheta:          10000.0,
		Architecture:       DefaultArchitecture,
		NumExpertsPerToken: 1,
	}
}

// HeadDim returns the head dimension (hidden_size / num_attention_heads).
func (c InferenceConfig) HeadDim() int {
	if c.NumAttentionHeads == 0 {
		return 0
	}
	return c.HiddenSize / c.NumAttentionHeads
}

// KVHeadDim returns the head dimension of the K/V heads.
func (c InferenceConfig) KVHeadDim() int {
	if c.KeyValueHeadDim != 0 {
		return c.KeyValueHeadDim
	}
	return c.HeadDim()
}

// FromGGUF builds an InferenceConfig from a parsed GGUF file's metadata.
func (c InferenceConfig) FromGGUF(file ggufcore.File) InferenceConfig {
	out := DefaultInferenceConfig()
	if v, ok := file.Metadata["llama.context_length"]; ok {
		if n, ok := v.AsUint64(); ok {
			out.ContextSize = int(n)
		}
	}
	if v, ok := file.Metadata["llama.embedding_length"]; ok {
		if n, ok := v.AsUint64(); ok {
			out.HiddenSize = int(n)
		}
	}
	if v, ok := file.Metadata["llama.attention.head_count"]; ok {
		if n, ok := v.AsUint64(); ok {
			out.NumAttentionHeads = int(n)
		}
	}
	if v, ok := file.Metadata["llama.attention.head_count_kv"]; ok {
		if n, ok := v.AsUint64(); ok {
			out.NumKeyValueHeads = int(n)
		}
	}
	if v, ok := file.Metadata["llama.feed_forward_length"]; ok {
		if n, ok := v.AsUint64(); ok {
			out.IntermediateSize = int(n)
		}
	}
	if v, ok := file.Metadata["llama.block_count"]; ok {
		if n, ok := v.AsUint64(); ok {
			out.LayerCount = int(n)
		}
	}
	if v, ok := file.Metadata["llama.attention.layer_norm_rms_epsilon"]; ok {
		if n, ok := v.AsUint64(); ok {
			out.RMSNormEps = float32(n)
		}
	}
	if v, ok := file.Metadata["tokenizer.ggml.bos_token_id"]; ok {
		if n, ok := v.AsUint64(); ok {
			out.VocabSize = int(n)
		}
	}
	// Try to recover vocab size from the embedding tensor if metadata is missing.
	for _, info := range file.TensorInfos {
		if info.Name == "token_embd.weight" && len(info.Dimensions) >= 2 {
			out.VocabSize = int(info.Dimensions[len(info.Dimensions)-2])
			out.HiddenSize = int(info.Dimensions[len(info.Dimensions)-1])
		}
	}
	if v, ok := file.Metadata["general.architecture"]; ok {
		switch v.String {
		case "llama":
			out.Architecture = ArchLlamaModel
		case "mistral":
			out.Architecture = ArchMistralModel
		case "mixtral":
			out.Architecture = ArchMixtralModel
		case "qwen2", "qwen":
			out.Architecture = ArchQwenModel
		case "gemma":
			out.Architecture = ArchGemmaModel
		case "phi2", "phi":
			out.Architecture = ArchPhiModel
		case "falcon":
			out.Architecture = ArchFalconModel
		case "gpt2":
			out.Architecture = ArchGpt2Model
		case "gptj":
			out.Architecture = ArchGptJModel
		case "gptneox":
			out.Architecture = ArchGptNeoXModel
		case "deepseek2":
			out.Architecture = ArchDeepSeekModel
		case "minimax":
			out.Architecture = ArchMiniMaxModel
		}
	}
	return out
}

// Workspace is a scratch buffer used by InferenceModel during forward.
type Workspace struct {
	mu      sync.Mutex
	scratch []float32
}

// NewWorkspace returns a workspace with the given initial capacity.
func NewWorkspace(capacity int) *Workspace { return &Workspace{scratch: make([]float32, 0, capacity)} }

// Get returns a slice of length n.
func (w *Workspace) Get(n int) []float32 {
	w.mu.Lock()
	defer w.mu.Unlock()
	if cap(w.scratch) < n {
		w.scratch = make([]float32, n)
	} else {
		w.scratch = w.scratch[:n]
	}
	return w.scratch
}

// Capacity returns the workspace capacity.
func (w *Workspace) Capacity() int { return cap(w.scratch) }

// WeightStorage is the placeholder for inference weight storage. In the Rust
// crate this is a tagged enum; in the Go port we hold the parsed GGUF
// bytes plus the architecture.
type WeightStorage struct {
	File *ggufcore.MappedFile
}

// InferenceModel is the canonical inference implementation.
type InferenceModel struct {
	Config    InferenceConfig
	Workspace *Workspace
	Storage   WeightStorage
	KVCache   *kv_cache.Cache
	Stack     *LlamaDecoderStack
}

// NewInferenceModel constructs an InferenceModel with optional loaded weights.
func NewInferenceModel(config InferenceConfig, storage WeightStorage, stack *LlamaDecoderStack) *InferenceModel {
	cfg := kv_cache.Config{
		LayerCount:   config.LayerCount,
		ContextSize:  config.ContextSize,
		HeadCount:    config.NumKeyValueHeads,
		HeadDim:      config.KVHeadDim(),
		DType:        "f32",
		Quantization: config.KVQuantization,
		Eviction:     kv_cache.EvictSlidingWindow,
	}
	return &InferenceModel{
		Config:    config,
		Workspace: NewWorkspace(config.HiddenSize * 4),
		Storage:   storage,
		KVCache:   kv_cache.NewCache(cfg),
		Stack:     stack,
	}
}

// Forward runs prefill or single-token decode and returns logits for the last token.
func (m *InferenceModel) Forward(tokens []Token, session *Session) (Logits, error) {
	if len(tokens) == 0 {
		return nil, EmptyInputError
	}
	requested := session.ConsumedTokens() + len(tokens)
	if m.Config.ContextSize > 0 && requested > m.Config.ContextSize {
		return nil, &ContextExceededError{
			ContextSize:          m.Config.ContextSize,
			RequestedTotalTokens: requested,
		}
	}
	if m.Stack == nil || !m.Stack.Loaded() {
		return make(Logits, m.Config.VocabSize), nil
	}

	startPos := session.ConsumedTokens()
	if m.Stack.PositionOffset != startPos {
		m.Stack.ResetCache()
		m.Stack.PositionOffset = startPos
	}

	var hidden []float32
	var err error
	if len(tokens) > 1 {
		batch := make([]uint32, len(tokens))
		for i, t := range tokens {
			batch[i] = uint32(t)
		}
		hidden, err = m.Stack.ForwardBatch(batch)
	} else {
		hidden, err = m.Stack.ForwardToken(uint32(tokens[0]))
	}
	if err != nil {
		return nil, NewErrorf("%v", err)
	}
	logits, err := m.Stack.Logits(hidden)
	if err != nil {
		return nil, NewErrorf("%v", err)
	}
	session.RecordTokens(len(tokens))
	return logits, nil
}

// VocabSize returns the configured vocabulary size.
func (m *InferenceModel) VocabSize() int { return m.Config.VocabSize }

// ContextSize returns the configured context size.
func (m *InferenceModel) ContextSize() int { return m.Config.ContextSize }

// LayerCount returns the configured layer count.
func (m *InferenceModel) LayerCount() int { return m.Config.LayerCount }

// RewindTo resets the decoder KV cache to a previous token count.
func (m *InferenceModel) RewindTo(_ *Session, n int) {
	if n < 0 {
		n = 0
	}
	if m.Stack != nil {
		m.Stack.ResetCache()
		m.Stack.PositionOffset = n
	}
	if m.KVCache != nil {
		m.KVCache = kv_cache.NewCache(m.KVCache.Config())
	}
}

// String returns a description of the model.
func (m *InferenceModel) String() string {
	return fmt.Sprintf("InferenceModel{arch=%s vocab=%d ctx=%d layers=%d hidden=%d}",
		m.Config.Architecture, m.Config.VocabSize, m.Config.ContextSize, m.Config.LayerCount, m.Config.HiddenSize)
}

// helpers for strconv (avoid unused import if BuildString disappears)
var _ = strconv.Itoa
