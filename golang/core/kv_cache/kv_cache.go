// Package kv_cache mirrors oxidize_core::compute::kv_cache. It implements
// the per-layer key/value cache used by autoregressive generation, with
// multiple quantization schemes and eviction strategies.
package kv_cache

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// Quantization identifies the per-token or per-block KV cache quantization
// scheme.
type Quantization uint8

const (
	// QuantAsymmetric uses per-token scale+min (default).
	QuantAsymmetric Quantization = iota
	// QuantTurboQuant uses 32-element TurboQuant blocks.
	QuantTurboQuant
)

// EvictionStrategy is the cache eviction policy.
type EvictionStrategy uint8

const (
	EvictSlidingWindow EvictionStrategy = iota
	EvictStopAtCapacity
)

// String returns the canonical name of the eviction strategy.
func (e EvictionStrategy) String() string {
	switch e {
	case EvictSlidingWindow:
		return "sliding_window"
	case EvictStopAtCapacity:
		return "stop_at_capacity"
	default:
		return fmt.Sprintf("eviction(%d)", uint8(e))
	}
}

// Config is the KV cache configuration.
type Config struct {
	LayerCount        int           `json:"layer_count"`
	ContextSize       int           `json:"context_size"`
	HeadCount         int           `json:"head_count"`
	HeadDim           int           `json:"head_dim"`
	DType             string        `json:"dtype"`
	Quantization      Quantization  `json:"quantization"`
	Eviction          EvictionStrategy `json:"eviction"`
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		LayerCount:   32,
		ContextSize:  2048,
		HeadCount:    32,
		HeadDim:      128,
		DType:        "f16",
		Quantization: QuantAsymmetric,
		Eviction:     EvictSlidingWindow,
	}
}

// TokenSize returns the number of f32 values stored per token.
func (c Config) TokenSize() int { return c.HeadCount * c.HeadDim * 2 }

// LayerSize returns the per-layer byte budget (assuming f32 storage).
func (c Config) LayerSize() int { return c.ContextSize * c.TokenSize() * 4 }

// ElementCount returns the total element count.
func (c Config) ElementCount() int { return c.ContextSize * c.LayerCount * c.TokenSize() }

// BlocksPerToken returns the number of 32-element TurboQuant blocks per
// token (for the TurboQuant path).
func (c Config) BlocksPerToken() int {
	return (c.TokenSize() + 31) / 32
}

// Error is returned from KV cache operations.
type Error struct{ Message string }

func (e *Error) Error() string { return "kv_cache: " + e.Message }

// PersistenceError wraps JSON I/O failures during save/load.
type PersistenceError struct{ Err error }

func (e *PersistenceError) Error() string { return "kv_cache persistence: " + e.Err.Error() }
func (e *PersistenceError) Unwrap() error { return e.Err }

// Cache is the per-layer key/value cache.
type Cache struct {
	mu       sync.RWMutex
	config   Config
	keys     [][]float32
	values   [][]float32
	occupied int
	// per-layer metadata: how many tokens are currently stored
	lengths  []int
}

// NewCache returns a fresh cache with the given configuration.
func NewCache(config Config) *Cache {
	keys := make([][]float32, config.LayerCount)
	vals := make([][]float32, config.LayerCount)
	for i := range keys {
		keys[i] = make([]float32, 0, config.ContextSize*config.TokenSize()/2)
		vals[i] = make([]float32, 0, config.ContextSize*config.TokenSize()/2)
	}
	return &Cache{
		config:  config,
		keys:    keys,
		values:  vals,
		lengths: make([]int, config.LayerCount),
	}
}

// Config returns the cache configuration.
func (c *Cache) Config() Config { return c.config }

// Append stores a key/value pair for a given layer at the next position.
func (c *Cache) Append(layer int, key, value []float32) error {
	if layer < 0 || layer >= c.config.LayerCount {
		return &Error{Message: fmt.Sprintf("layer %d out of bounds", layer)}
	}
	if len(key) < c.config.HeadCount*c.config.HeadDim {
		return &Error{Message: "key too small"}
	}
	if len(value) < c.config.HeadCount*c.config.HeadDim {
		return &Error{Message: "value too small"}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.lengths[layer] >= c.config.ContextSize {
		switch c.config.Eviction {
		case EvictSlidingWindow:
			c.evictOldest(layer)
		case EvictStopAtCapacity:
			return &Error{Message: "cache full"}
		}
	}
	dim := c.config.HeadCount * c.config.HeadDim
	c.keys[layer] = append(c.keys[layer], key[:dim]...)
	c.values[layer] = append(c.values[layer], value[:dim]...)
	c.lengths[layer]++
	if layer == 0 {
		c.occupied++
	}
	return nil
}

// Get returns a copy of the current key/value buffers for the given layer.
func (c *Cache) Get(layer int) (keys, values []float32, length int, err error) {
	if layer < 0 || layer >= c.config.LayerCount {
		return nil, nil, 0, &Error{Message: "layer out of bounds"}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	dim := c.config.HeadCount * c.config.HeadDim
	length = c.lengths[layer]
	keys = append([]float32(nil), c.keys[layer][:length*dim]...)
	values = append([]float32(nil), c.values[layer][:length*dim]...)
	return keys, values, length, nil
}

func (c *Cache) evictOldest(layer int) {
	dim := c.config.HeadCount * c.config.HeadDim
	c.keys[layer] = c.keys[layer][dim:]
	c.values[layer] = c.values[layer][dim:]
	c.lengths[layer]--
}

// OccupiedTokens returns the maximum length across all layers.
func (c *Cache) OccupiedTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	max := 0
	for _, n := range c.lengths {
		if n > max {
			max = n
		}
	}
	return max
}

// SaveToDisk serializes the cache to disk in a simple format: header (json)
// followed by raw key/value byte buffers.
func (c *Cache) SaveToDisk(path string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	snap := struct {
		Config Config `json:"config"`
		Lengths []int `json:"lengths"`
	}{Config: c.config, Lengths: c.lengths}
	header, err := json.Marshal(snap)
	if err != nil {
		return &PersistenceError{Err: err}
	}
	out, err := os.Create(path)
	if err != nil {
		return &PersistenceError{Err: err}
	}
	defer out.Close()
	if err := binary.Write(out, binary.LittleEndian, uint32(len(header))); err != nil {
		return &PersistenceError{Err: err}
	}
	if _, err := out.Write(header); err != nil {
		return &PersistenceError{Err: err}
	}
	// keys then values
	for i := 0; i < c.config.LayerCount; i++ {
		if _, err := out.Write(asBytesF32(c.keys[i])); err != nil {
			return &PersistenceError{Err: err}
		}
	}
	for i := 0; i < c.config.LayerCount; i++ {
		if _, err := out.Write(asBytesF32(c.values[i])); err != nil {
			return &PersistenceError{Err: err}
		}
	}
	return nil
}

// LoadFromDisk is the inverse of SaveToDisk.
func LoadFromDisk(path string) (*Cache, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, &PersistenceError{Err: err}
	}
	if len(raw) < 4 {
		return nil, &PersistenceError{Err: fmt.Errorf("file too small")}
	}
	headerLen := binary.LittleEndian.Uint32(raw[:4])
	headerBytes := raw[4 : 4+headerLen]
	var snap struct {
		Config Config `json:"config"`
		Lengths []int `json:"lengths"`
	}
	if err := json.Unmarshal(headerBytes, &snap); err != nil {
		return nil, &PersistenceError{Err: err}
	}
	cache := NewCache(snap.Config)
	pos := 4 + int(headerLen)
	for i := 0; i < snap.Config.LayerCount; i++ {
		size := snap.Lengths[i] * snap.Config.HeadCount * snap.Config.HeadDim * 4
		cache.keys[i] = append(cache.keys[i], bytesAsF32(raw[pos:pos+size])...)
		pos += size
	}
	for i := 0; i < snap.Config.LayerCount; i++ {
		size := snap.Lengths[i] * snap.Config.HeadCount * snap.Config.HeadDim * 4
		cache.values[i] = append(cache.values[i], bytesAsF32(raw[pos:pos+size])...)
		pos += size
	}
	cache.lengths = snap.Lengths
	return cache, nil
}

func asBytesF32(data []float32) []byte {
	out := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(out[i*4:], float32bits(v))
	}
	return out
}

func bytesAsF32(data []byte) []float32 {
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

func float32bits(f float32) uint32 {
	// Imported via binary.Float32bits equivalent
	return binary.LittleEndian.Uint32(asBytesF32([]float32{f})[:4])
}

func float32frombits(b uint32) float32 {
	var f float32
	binary.LittleEndian.PutUint32(asBytesF32([]float32{f})[:4], b)
	return f
}

// ContinuousBatchError mirrors ContinuousBatchError.
type ContinuousBatchError struct{ Message string }

func (e *ContinuousBatchError) Error() string { return "continuous batch: " + e.Message }

// ContinuousBatchCache mirrors ContinuousBatchKvCache.
type ContinuousBatchCache struct {
	mu      sync.Mutex
	config  Config
	seqKeys map[uint64][][]float32
	seqVals map[uint64][][]float32
	lengths map[uint64][]int
}

// NewContinuousBatchCache returns a fresh continuous-batch cache.
func NewContinuousBatchCache(config Config) *ContinuousBatchCache {
	return &ContinuousBatchCache{
		config:  config,
		seqKeys: map[uint64][][]float32{},
		seqVals: map[uint64][][]float32{},
		lengths: map[uint64][]int{},
	}
}

// CreateSequence initializes storage for a new sequence.
func (c *ContinuousBatchCache) CreateSequence(seqID uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.seqKeys[seqID]; ok {
		return &ContinuousBatchError{Message: "sequence already exists"}
	}
	c.seqKeys[seqID] = make([][]float32, c.config.LayerCount)
	c.seqVals[seqID] = make([][]float32, c.config.LayerCount)
	c.lengths[seqID] = make([]int, c.config.LayerCount)
	return nil
}

// Append appends a token's key/value to a sequence.
func (c *ContinuousBatchCache) Append(seqID uint64, layer int, key, value []float32) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	keys, ok := c.seqKeys[seqID]
	if !ok {
		return &ContinuousBatchError{Message: "sequence not found"}
	}
	dim := c.config.HeadCount * c.config.HeadDim
	if keys[layer] == nil {
		keys[layer] = make([]float32, 0, dim)
	}
	if c.lengths[seqID][layer] >= c.config.ContextSize {
		return &ContinuousBatchError{Message: "sequence capacity exceeded"}
	}
	keys[layer] = append(keys[layer], key[:dim]...)
	c.seqVals[seqID][layer] = append(c.seqVals[seqID][layer], value[:dim]...)
	c.lengths[seqID][layer]++
	return nil
}

// RemoveSequence deletes a sequence's storage.
func (c *ContinuousBatchCache) RemoveSequence(seqID uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.seqKeys[seqID]; !ok {
		return &ContinuousBatchError{Message: "sequence not found"}
	}
	delete(c.seqKeys, seqID)
	delete(c.seqVals, seqID)
	delete(c.lengths, seqID)
	return nil
}

// SequenceLength returns the current length of a sequence for a given layer.
func (c *ContinuousBatchCache) SequenceLength(seqID uint64, layer int) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	l, ok := c.lengths[seqID]
	if !ok || layer < 0 || layer >= len(l) {
		return 0
	}
	return l[layer]
}
