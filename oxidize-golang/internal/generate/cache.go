package generate

import (
	"fmt"
	"sync"

	"github.com/Zapdev-labs/oxidize/golang/core/model"
)

// CachedModel holds a loaded model and tokenizer path for reuse across requests.
type CachedModel struct {
	Path     string
	Result   LoaderResult
	Warning  string
	Backend  string
	FellBack bool
}

// ModelCache stores GGUF models keyed by path + loader settings.
type ModelCache struct {
	mu    sync.RWMutex
	items map[string]*CachedModel
}

// DefaultModelCache is the process-wide server cache.
var DefaultModelCache = &ModelCache{items: make(map[string]*CachedModel)}

func (c *ModelCache) cacheKey(path string, cfg LoaderConfig) string {
	return fmt.Sprintf("%s|%s|%d|%d|%d|%s", path, cfg.Backend, cfg.ContextSize, cfg.NGPULayers, cfg.GPUs, cfg.Parallelism)
}

// Get returns a cached entry if present.
func (c *ModelCache) Get(path string, cfg LoaderConfig) (*CachedModel, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	item, ok := c.items[c.cacheKey(path, cfg)]
	return item, ok
}

// Load returns a cached model or loads and stores it.
func (c *ModelCache) Load(path string, cfg LoaderConfig) (CachedModel, error) {
	key := c.cacheKey(path, cfg)
	c.mu.RLock()
	if item, ok := c.items[key]; ok {
		c.mu.RUnlock()
		return *item, nil
	}
	c.mu.RUnlock()

	result, err := LoadModelFromPath(path, cfg)
	if err != nil {
		return CachedModel{}, err
	}
	entry := &CachedModel{
		Path:     path,
		Result:   result,
		Warning:  result.Warning,
		Backend:  result.Backend,
		FellBack: result.FellBack,
	}
	c.mu.Lock()
	c.items[key] = entry
	c.mu.Unlock()
	return *entry, nil
}

// InferenceFromCache loads and returns *InferenceModel when possible.
func InferenceFromCache(path string, cfg LoaderConfig) (*model.InferenceModel, CachedModel, error) {
	entry, err := DefaultModelCache.Load(path, cfg)
	if err != nil {
		return nil, CachedModel{}, err
	}
	inference, ok := entry.Result.Model.(*model.InferenceModel)
	if !ok {
		return nil, entry, fmt.Errorf("generate: expected InferenceModel, got %T", entry.Result.Model)
	}
	return inference, entry, nil
}
