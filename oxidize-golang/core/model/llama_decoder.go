package model

// LlamaDecoderConfig holds tensor geometry for a standard Llama-style decoder stack.
type LlamaDecoderConfig struct {
	HiddenSize          int
	LayerCount          int
	IntermediateSize    int
	NumAttentionHeads   int
	NumKeyValueHeads    int
	KeyValueHeadDim     int
	VocabSize           int
	RMSNormEps          float32
	RopeTheta           float32
	NumExperts          int
	NumExpertsPerToken  int
	SlidingWindow       int
	UseAlibi            bool
	ParallelAttnFFN     bool
}

// HeadDim returns hidden_size / num_attention_heads.
func (c LlamaDecoderConfig) HeadDim() int {
	if c.NumAttentionHeads == 0 {
		return 0
	}
	return c.HiddenSize / c.NumAttentionHeads
}

// KVHeadDim returns per-head K/V dimension.
func (c LlamaDecoderConfig) KVHeadDim() int {
	if c.KeyValueHeadDim > 0 {
		return c.KeyValueHeadDim
	}
	return c.HeadDim()
}

// LlamaDecoderConfigFromInference maps InferenceConfig into decoder geometry.
func LlamaDecoderConfigFromInference(cfg InferenceConfig) LlamaDecoderConfig {
	window := cfg.SlidingWindow
	if window == 0 && cfg.Architecture.UsesSlidingWindow() {
		window = 4096
	}
	return LlamaDecoderConfig{
		HiddenSize:         cfg.HiddenSize,
		LayerCount:         cfg.LayerCount,
		IntermediateSize:   cfg.IntermediateSize,
		NumAttentionHeads:    cfg.NumAttentionHeads,
		NumKeyValueHeads:   cfg.NumKeyValueHeads,
		KeyValueHeadDim:    cfg.KVHeadDim(),
		VocabSize:          cfg.VocabSize,
		RMSNormEps:         cfg.RMSNormEps,
		RopeTheta:          cfg.RopeTheta,
		NumExperts:         cfg.NumExperts,
		NumExpertsPerToken: cfg.NumExpertsPerToken,
		SlidingWindow:      window,
		UseAlibi:           cfg.Architecture.UsesAlibi(),
		ParallelAttnFFN:    cfg.Architecture.UsesParallelAttnFFN(),
	}
}

// LlamaDecoderConfigFromDFlash maps DFlashConfig into decoder geometry.
func LlamaDecoderConfigFromDFlash(cfg DFlashConfig) LlamaDecoderConfig {
	return LlamaDecoderConfig{
		HiddenSize:        cfg.HiddenSize,
		LayerCount:        cfg.NumHiddenLayers,
		IntermediateSize:  cfg.IntermediateSize,
		NumAttentionHeads: cfg.NumAttentionHeads,
		NumKeyValueHeads:  cfg.NumKeyValueHeads,
		VocabSize:         cfg.VocabSize,
		RMSNormEps:        cfg.RMSNormEps,
		RopeTheta:         cfg.RopeTheta,
	}
}

// DecoderAttentionLayer holds per-layer attention projections.
type DecoderAttentionLayer struct {
	QProj, KProj, VProj, OProj F32Weight
	QNormWeight, KNormWeight   []float32
}

// DecoderLayer is norm → attention → MLP (dense and/or MoE).
type DecoderLayer struct {
	InputLayernorm, PostAttentionLayernorm []float32
	Attention                              DecoderAttentionLayer
	MLPGate, MLPUp, MLPDown                F32Weight
	MoE                                    MoELayer
}

// DecoderKvLayerCache stores per-layer KV in flattened [seq][kv_heads*head_dim] layout.
type DecoderKvLayerCache struct {
	Keys, Values []float32
	SeqLen       int
}

// DFlashAttentionLayer aliases DecoderAttentionLayer for the DFlash draft path.
type DFlashAttentionLayer = DecoderAttentionLayer

// DFlashDecoderLayer aliases DecoderLayer for the DFlash draft path.
type DFlashDecoderLayer = DecoderLayer

// DFlashKvLayerCache aliases DecoderKvLayerCache for the DFlash draft path.
type DFlashKvLayerCache = DecoderKvLayerCache

// NewDFlashKvLayerCache returns an empty cache.
func NewDFlashKvLayerCache() DFlashKvLayerCache { return DecoderKvLayerCache{} }

func (c *DecoderKvLayerCache) clear() {
	c.Keys = c.Keys[:0]
	c.Values = c.Values[:0]
	c.SeqLen = 0
}

func (c *DecoderKvLayerCache) reserveTokens(additional, kvLen int) {
	add := additional * kvLen
	if cap(c.Keys)-len(c.Keys) < add {
		c.Keys = append(c.Keys, make([]float32, 0, len(c.Keys)+add)...)
	}
	if cap(c.Values)-len(c.Values) < add {
		c.Values = append(c.Values, make([]float32, 0, len(c.Values)+add)...)
	}
}

// LlamaDecoderStack is the shared Llama-style transformer body used by InferenceModel
// and DFlashDraftModel.
type LlamaDecoderStack struct {
	Config         LlamaDecoderConfig
	Layers         []DecoderLayer
	TokEmbeddings  F32Weight
	Output         F32Weight
	Norm           []float32
	KVCache        []DecoderKvLayerCache
	PositionOffset int
	AlibiSlopes    []float32
}

// NewLlamaDecoderStack returns an empty stack sized for config.LayerCount layers.
func NewLlamaDecoderStack(config LlamaDecoderConfig) *LlamaDecoderStack {
	s := &LlamaDecoderStack{
		Config:  config,
		KVCache: make([]DecoderKvLayerCache, config.LayerCount),
	}
	if config.UseAlibi && config.NumAttentionHeads > 0 {
		s.AlibiSlopes = AlibiSlopes(config.NumAttentionHeads)
	}
	return s
}

// Loaded reports whether weights required for forward are present.
func (s *LlamaDecoderStack) Loaded() bool {
	return s.TokEmbeddings.IsLoaded() && s.Output.IsLoaded()
}

// ResetCache clears KV cache and position.
func (s *LlamaDecoderStack) ResetCache() {
	if len(s.KVCache) != s.Config.LayerCount {
		s.KVCache = make([]DecoderKvLayerCache, s.Config.LayerCount)
	} else {
		for i := range s.KVCache {
			s.KVCache[i].clear()
		}
	}
	s.PositionOffset = 0
}

// ReserveCacheTokens pre-allocates KV cache capacity.
func (s *LlamaDecoderStack) ReserveCacheTokens(tokens int) {
	kvLen := s.Config.NumKeyValueHeads * s.Config.KVHeadDim()
	for i := range s.KVCache {
		s.KVCache[i].reserveTokens(tokens, kvLen)
	}
}

func (s *LlamaDecoderStack) fillTokenEmbedding(token uint32, output []float32) error {
	if !s.TokEmbeddings.IsLoaded() {
		return nil
	}
	vocab := s.TokEmbeddings.outputDim()
	if vocab < 1 {
		vocab = 1
	}
	idx := int(token)
	if idx >= vocab {
		idx = vocab - 1
	}
	embW := s.TokEmbeddings.inputDim()
	if embW == len(output) {
		return s.TokEmbeddings.Row(idx, output)
	}
	emb := make([]float32, embW)
	if err := s.TokEmbeddings.Row(idx, emb); err != nil {
		return err
	}
	n := len(output)
	if n > len(emb) {
		n = len(emb)
	}
	copy(output[:n], emb[:n])
	return nil
}
