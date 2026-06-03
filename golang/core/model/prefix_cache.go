package model

import (
	"crypto/sha1"
	"encoding/hex"
	"sync"
)

// PrefixHash mirrors PrefixHash.
type PrefixHash [20]byte

// String returns the hex-encoded hash.
func (p PrefixHash) String() string { return hex.EncodeToString(p[:]) }

// ComputePrefixHash returns a stable hash for the given token prefix.
func ComputePrefixHash(prefix []Token) PrefixHash {
	sum := sha1.Sum(prefixBytes(prefix))
	var h PrefixHash
	copy(h[:], sum[:])
	return h
}

func prefixBytes(prefix []Token) []byte {
	out := make([]byte, 0, len(prefix)*4)
	for _, t := range prefix {
		out = append(out, byte(t), byte(t>>8), byte(t>>16), byte(t>>24))
	}
	return out
}

// CachedPrefix mirrors CachedPrefix.
type CachedPrefix struct {
	Hash         PrefixHash
	Tokens       []Token
	Length       int
	SharedLayers int
}

// PrefixCache mirrors PrefixCache.
type PrefixCache struct {
	mu      sync.RWMutex
	entries map[PrefixHash]*CachedPrefix
	limit   int
	hits    int
	misses  int
}

// NewPrefixCache constructs a cache with a maximum number of entries.
func NewPrefixCache(limit int) *PrefixCache {
	if limit <= 0 {
		limit = 32
	}
	return &PrefixCache{entries: map[PrefixHash]*CachedPrefix{}, limit: limit}
}

// Lookup returns the cached prefix and true if present. It mutates the
// hit/miss counters, so it must take the full write lock to avoid data
// races with concurrent callers.
func (c *PrefixCache) Lookup(prefix []Token) (*CachedPrefix, bool) {
	h := ComputePrefixHash(prefix)
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[h]
	if ok {
		c.hits++
		return entry, true
	}
	c.misses++
	return nil, false
}

// Insert adds a prefix to the cache, evicting the oldest if necessary.
func (c *PrefixCache) Insert(prefix []Token, sharedLayers int) *CachedPrefix {
	h := ComputePrefixHash(prefix)
	entry := &CachedPrefix{Hash: h, Tokens: append([]Token(nil), prefix...), Length: len(prefix), SharedLayers: sharedLayers}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.entries) >= c.limit {
		var oldest *CachedPrefix
		for _, e := range c.entries {
			if oldest == nil || e.Length < oldest.Length {
				oldest = e
			}
		}
		if oldest != nil {
			delete(c.entries, oldest.Hash)
		}
	}
	c.entries[h] = entry
	return entry
}

// Clear removes all entries.
func (c *PrefixCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = map[PrefixHash]*CachedPrefix{}
}

// Stats returns hits and misses.
func (c *PrefixCache) Stats() (int, int) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.hits, c.misses
}

// HitRate returns the hit ratio.
func (c *PrefixCache) HitRate() float32 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	total := c.hits + c.misses
	if total == 0 {
		return 0
	}
	return float32(c.hits) / float32(total)
}

// PrefixMatcher mirrors PrefixMatcher.
type PrefixMatcher struct {
	Cache *PrefixCache
}

// NewPrefixMatcher constructs a matcher.
func NewPrefixMatcher(cache *PrefixCache) *PrefixMatcher { return &PrefixMatcher{Cache: cache} }

// LongestMatch returns the longest cached prefix for the given tokens.
func (m *PrefixMatcher) LongestMatch(tokens []Token) *CachedPrefix {
	if m.Cache == nil {
		return nil
	}
	for n := len(tokens); n > 0; n-- {
		entry, ok := m.Cache.Lookup(tokens[:n])
		if ok {
			return entry
		}
	}
	return nil
}

// CanReuseState reports whether the prefix can be reused for a new request.
func (m *PrefixMatcher) CanReuseState(prefix []Token) bool {
	entry, ok := m.Cache.Lookup(prefix)
	return ok && entry.SharedLayers > 0
}

// PrefixCacheConfig mirrors PrefixCacheConfig.
type PrefixCacheConfig struct {
	MaxEntries      int
	MinPrefixLength int
	EnableLRU       bool
}

// DefaultPrefixCacheConfig returns sensible defaults.
func DefaultPrefixCacheConfig() PrefixCacheConfig {
	return PrefixCacheConfig{MaxEntries: 64, MinPrefixLength: 4, EnableLRU: true}
}

// BuildPrefixCache constructs a cache from the config.
func BuildPrefixCache(cfg PrefixCacheConfig) *PrefixCache {
	return NewPrefixCache(cfg.MaxEntries)
}
