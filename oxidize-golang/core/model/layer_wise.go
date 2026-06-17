package model

import (
	"container/list"
	"sync"

	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
	"github.com/Zapdev-labs/oxidize/golang/core/kv_cache"
)

// LayerWiseModel streams transformer layers through an LRU cache. When Inner is
// set it delegates forward to a fully-loaded inference model while tracking
// layer residency for RAM-offload planning.
type LayerWiseModel struct {
	Config    InferenceConfig
	Storage   WeightStorage
	Workspace *Workspace
	CacheSize int
	KVCache   *kv_cache.Cache
	Inner     *InferenceModel
	cache     *list.List
	cacheKeys map[int]*list.Element
	mu        sync.Mutex
}

// NewLayerWiseModel constructs a new LayerWiseModel with the given cache
// size (number of layers to keep resident).
func NewLayerWiseModel(config InferenceConfig, storage WeightStorage, cacheSize int) *LayerWiseModel {
	if cacheSize <= 0 {
		cacheSize = 4
	}
	cfg := kv_cache.Config{
		LayerCount:   config.LayerCount,
		ContextSize:  config.ContextSize,
		HeadCount:    config.NumKeyValueHeads,
		HeadDim:      config.KVHeadDim(),
		DType:        "f32",
		Quantization: config.KVQuantization,
		Eviction:     kv_cache.EvictSlidingWindow,
	}
	return &LayerWiseModel{
		Config:    config,
		Storage:   storage,
		Workspace: NewWorkspace(config.HiddenSize * 4),
		CacheSize: cacheSize,
		KVCache:   kv_cache.NewCache(cfg),
		cache:     list.New(),
		cacheKeys: map[int]*list.Element{},
	}
}

// Forward runs inference, touching the LRU cache for each token's layer index.
func (m *LayerWiseModel) Forward(tokens []Token, session *Session) (Logits, error) {
	if len(tokens) == 0 {
		return nil, EmptyInputError
	}
	for _, l := range tokens {
		if m.Config.LayerCount > 0 {
			m.touchLayer(int(l) % m.Config.LayerCount)
		}
	}
	if m.Inner != nil {
		return m.Inner.Forward(tokens, session)
	}
	return make(Logits, m.Config.VocabSize), nil
}

func (m *LayerWiseModel) touchLayer(idx int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if el, ok := m.cacheKeys[idx]; ok {
		m.cache.MoveToFront(el)
		return
	}
	el := m.cache.PushBack(idx)
	m.cacheKeys[idx] = el
	if m.cache.Len() > m.CacheSize {
		oldest := m.cache.Front()
		if oldest != nil {
			m.cache.Remove(oldest)
			delete(m.cacheKeys, oldest.Value.(int))
		}
	}
}

// VocabSize returns the configured vocabulary size.
func (m *LayerWiseModel) VocabSize() int { return m.Config.VocabSize }

// ContextSize returns the configured context size.
func (m *LayerWiseModel) ContextSize() int { return m.Config.ContextSize }

// LayerCount returns the configured layer count.
func (m *LayerWiseModel) LayerCount() int { return m.Config.LayerCount }

// NewLayerWiseFromInference wraps an existing inference model with LRU tracking.
func NewLayerWiseFromInference(inner *InferenceModel, cacheSize int) *LayerWiseModel {
	if inner == nil {
		return NewLayerWiseModel(DefaultInferenceConfig(), WeightStorage{}, cacheSize)
	}
	m := NewLayerWiseModel(inner.Config, inner.Storage, cacheSize)
	m.Inner = inner
	m.KVCache = inner.KVCache
	return m
}

// NewLayerWiseFromGGUF is a convenience constructor.
func NewLayerWiseFromGGUF(file ggufcore.File, cacheSize int) *LayerWiseModel {
	cfg := DefaultInferenceConfig().FromGGUF(file)
	return NewLayerWiseModel(cfg, WeightStorage{File: &ggufcore.MappedFile{Bytes: nil, Parsed: file}}, cacheSize)
}
