package cudabackend

// F32Weight is a single f32 weight matrix passed to PreloadLayer.
type F32Weight struct {
	Data []float32
	Rows int
	Cols int
}

// SetLayerConfig configures the layer cache budget. Mirrors
// gpu_state.rs:set_layer_config.
func SetLayerConfig(config CudaLayerConfig) error {
	return withGPU(func(s *GpuState) error {
		s.layerConfig = config
		s.enforceBudget()
		return nil
	})
}

// PreloadLayer marks a layer as needed and uploads its f32 weights if not
// already resident, evicting LRU layers when over budget. Mirrors
// gpu_state.rs:preload_layer.
func PreloadLayer(layer LayerID, weights []F32Weight) error {
	return withGPU(func(s *GpuState) error {
		if _, ok := s.layerMap[layer]; ok {
			s.touchLayer(layer)
			return nil
		}
		entry := &layerEntry{}
		for _, w := range weights {
			key := f32CacheKey(w.Data)
			if _, ok := s.residentF32[key]; !ok {
				buf := append([]float32(nil), w.Data...)
				entry.bytes += uint64(len(buf) * sizeOfF32)
				s.residentF32[key] = buf
			}
			entry.f32Keys = append(entry.f32Keys, key)
		}
		s.residentBytes += entry.bytes
		s.layerMap[layer] = entry
		s.touchLayer(layer)
		return nil
	})
}

// EvictLayer explicitly evicts a layer from VRAM. Mirrors
// gpu_state.rs:evict_layer.
func EvictLayer(layer LayerID) error {
	return withGPU(func(s *GpuState) error {
		s.removeFromLRU(layer)
		s.evictLayerInternal(layer)
		return nil
	})
}

// ResidentVramBytes reports the bytes of weight data currently resident on the
// GPU. Mirrors gpu_state.rs:resident_vram_bytes.
func ResidentVramBytes() uint64 {
	var n uint64
	_ = withGPU(func(s *GpuState) error {
		n = s.residentBytes
		return nil
	})
	return n
}

// ClearResidentCache clears all resident weight caches (f16, f32, quant, and
// layer entries). Mirrors gpu_state.rs:clear_resident_cache.
func ClearResidentCache() error {
	return withGPU(func(s *GpuState) error {
		s.residentF16 = make(map[weightCacheKey][]uint16)
		s.residentF32 = make(map[weightCacheKey][]float32)
		s.residentQuant = make(map[weightCacheKey][]byte)
		s.layerMap = make(map[LayerID]*layerEntry)
		s.layerLRU = nil
		s.orphanF16Keys = nil
		s.orphanQuantKeys = nil
		s.residentBytes = 0
		return nil
	})
}
