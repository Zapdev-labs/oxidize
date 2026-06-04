package model

import (
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/backends"
	"github.com/Zapdev-labs/oxidize/golang/core/kv_cache"
)

// ApplyLoaderConfig applies CLI loader options to a loaded inference model.
func ApplyLoaderConfig(m *InferenceModel, cfg LoaderConfig) {
	if m == nil {
		return
	}
	if cfg.ContextSize > 0 {
		m.Config.ContextSize = cfg.ContextSize
		if m.KVCache != nil {
			kvCfg := m.KVCache.Config()
			kvCfg.ContextSize = cfg.ContextSize
			m.KVCache = kv_cache.NewCache(kvCfg)
		}
	}
	_ = cfg.Backend
	_ = cfg.Threads
	_ = cfg.NGPULayers
	_ = cfg.GPUs
	_ = cfg.Parallelism
}

// ResolveLoaderBackend picks an effective backend name for logging/planning.
func ResolveLoaderBackend(name string, allowFallback bool) (effective string, warning string, err error) {
	res, err := backends.NewComputeBackend(name, allowFallback)
	if err != nil {
		return "", "", err
	}
	return res.Effective.String(), res.Warning, nil
}

// IsDFlashArchitecture reports whether a GGUF architecture string denotes DFlash.
func IsDFlashArchitecture(arch string) bool {
	arch = strings.ToLower(strings.TrimSpace(arch))
	return strings.Contains(arch, "dflash")
}
