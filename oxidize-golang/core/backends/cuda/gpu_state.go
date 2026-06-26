package cudabackend

import (
	"hash/fnv"
	"math"
	"sync"
)

// Package-level GPU state. The Rust port keys this off thread_local! because
// each thread owns a CUDA context; Go goroutines are not pinned to OS threads,
// so the port keeps a single process-wide state guarded by a mutex. This avoids
// duplicating weight caches per goroutine while preserving the same LRU/budget
// semantics.
var (
	gpuStateOnce sync.Once
	gpuState     *GpuState
)

func globalGpuState() *GpuState {
	gpuStateOnce.Do(func() { gpuState = newGpuState() })
	return gpuState
}

// withGPU runs f against the process-wide GPU state with the state lock held,
// mirroring gpu_state.rs:with_gpu.
func withGPU(f func(*GpuState) error) error {
	st := globalGpuState()
	st.mu.Lock()
	defer st.mu.Unlock()
	return f(st)
}

const (
	sizeOfF32 = 4
	sizeOfU16 = 2
)

// f32CacheKey derives a stable cache key from an f32 weight matrix using its
// length and an FNV-1a hash of its bit pattern.
func f32CacheKey(data []float32) weightCacheKey {
	h := fnv.New64a()
	var b [4]byte
	for _, v := range data {
		bits := math.Float32bits(v)
		b[0] = byte(bits)
		b[1] = byte(bits >> 8)
		b[2] = byte(bits >> 16)
		b[3] = byte(bits >> 24)
		_, _ = h.Write(b[:])
	}
	return weightCacheKey{hash: h.Sum64(), len: len(data)}
}

// byteCacheKey derives a stable cache key from raw quantized bytes.
func byteCacheKey(data []byte) weightCacheKey {
	h := fnv.New64a()
	_, _ = h.Write(data)
	return weightCacheKey{hash: h.Sum64(), len: len(data)}
}

// getF32Buffer returns a reusable f32 buffer from the pool, or allocates one.
// Mirrors GpuState::get_f32_buffer.
func (s *GpuState) getF32Buffer(n int) []float32 {
	if pool := s.f32Pool[n]; len(pool) > 0 {
		buf := pool[len(pool)-1]
		s.f32Pool[n] = pool[:len(pool)-1]
		for i := range buf {
			buf[i] = 0
		}
		return buf
	}
	return make([]float32, n)
}

// returnF32Buffer returns a buffer to the pool. Mirrors return_f32_buffer.
func (s *GpuState) returnF32Buffer(buf []float32) {
	s.f32Pool[len(buf)] = append(s.f32Pool[len(buf)], buf)
}

// getQ8KBuffer returns a reusable byte buffer from the pool, or allocates one.
func (s *GpuState) getQ8KBuffer(n int) []byte {
	if pool := s.q8kPool[n]; len(pool) > 0 {
		buf := pool[len(pool)-1]
		s.q8kPool[n] = pool[:len(pool)-1]
		return buf
	}
	return make([]byte, n)
}

// returnQ8KBuffer returns a byte buffer to the pool.
func (s *GpuState) returnQ8KBuffer(buf []byte) {
	s.q8kPool[len(buf)] = append(s.q8kPool[len(buf)], buf)
}

// touchLayer marks a layer most-recently-used and enforces the budget.
// Mirrors GpuState::touch_layer.
func (s *GpuState) touchLayer(layer LayerID) {
	if s.layerConfig.MaxResidentLayers == 0 && s.layerConfig.MaxVramBytes == 0 {
		return // unlimited
	}
	s.removeFromLRU(layer)
	s.layerLRU = append(s.layerLRU, layer)
	s.enforceBudget()
}

func (s *GpuState) removeFromLRU(layer LayerID) {
	for i, id := range s.layerLRU {
		if id == layer {
			s.layerLRU = append(s.layerLRU[:i], s.layerLRU[i+1:]...)
			return
		}
	}
}

// enforceBudget evicts LRU layers and orphan entries until within budget.
// Mirrors GpuState::enforce_budget / enforce_budget_protecting(None).
func (s *GpuState) enforceBudget() { s.enforceBudgetProtecting(nil) }

func (s *GpuState) enforceBudgetProtecting(protect *weightCacheKey) {
	maxLayers := s.layerConfig.MaxResidentLayers
	maxBytes := s.layerConfig.MaxVramBytes

	for {
		overLayer := maxLayers > 0 && len(s.layerLRU) > maxLayers
		overBytes := maxBytes > 0 && s.residentBytes > maxBytes
		if !overLayer && !overBytes {
			break
		}
		if len(s.layerLRU) == 0 {
			break
		}
		evict := s.layerLRU[0]
		s.layerLRU = s.layerLRU[1:]
		s.evictLayerInternal(evict)
	}

	for maxBytes > 0 && s.residentBytes > maxBytes {
		if len(s.orphanF16Keys) > 0 {
			key := s.orphanF16Keys[0]
			s.orphanF16Keys = s.orphanF16Keys[1:]
			if buf, ok := s.residentF16[key]; ok {
				s.residentBytes -= uint64(len(buf) * sizeOfU16)
				delete(s.residentF16, key)
			}
			continue
		}
		if len(s.orphanQuantKeys) > 0 {
			key := s.orphanQuantKeys[0]
			if protect != nil && key == *protect {
				// Don't evict the entry the caller still needs.
				break
			}
			s.orphanQuantKeys = s.orphanQuantKeys[1:]
			if buf, ok := s.residentQuant[key]; ok {
				s.residentBytes -= uint64(len(buf))
				delete(s.residentQuant, key)
			}
			continue
		}
		break
	}
}

// ensureVramHeadroom evicts until residentBytes+additional fits the budget,
// before inserting new weights. Mirrors GpuState::ensure_vram_headroom.
func (s *GpuState) ensureVramHeadroom(additional uint64) {
	maxBytes := s.layerConfig.MaxVramBytes
	if maxBytes == 0 {
		return
	}
	for s.residentBytes+additional > maxBytes {
		if len(s.layerLRU) > 0 {
			evict := s.layerLRU[0]
			s.layerLRU = s.layerLRU[1:]
			s.evictLayerInternal(evict)
			continue
		}
		if len(s.orphanF16Keys) > 0 {
			key := s.orphanF16Keys[0]
			s.orphanF16Keys = s.orphanF16Keys[1:]
			if buf, ok := s.residentF16[key]; ok {
				s.residentBytes -= uint64(len(buf) * sizeOfU16)
				delete(s.residentF16, key)
			}
			continue
		}
		if len(s.orphanQuantKeys) > 0 {
			key := s.orphanQuantKeys[0]
			s.orphanQuantKeys = s.orphanQuantKeys[1:]
			if buf, ok := s.residentQuant[key]; ok {
				s.residentBytes -= uint64(len(buf))
				delete(s.residentQuant, key)
			}
			continue
		}
		break
	}
}

func (s *GpuState) touchOrphanQuant(key weightCacheKey) {
	for i, k := range s.orphanQuantKeys {
		if k == key {
			s.orphanQuantKeys = append(s.orphanQuantKeys[:i], s.orphanQuantKeys[i+1:]...)
			break
		}
	}
	s.orphanQuantKeys = append(s.orphanQuantKeys, key)
}

// ensureResidentQuant uploads quantized weights once and reuses the buffer on
// later tokens. Mirrors GpuState::ensure_resident_quant.
func (s *GpuState) ensureResidentQuant(key weightCacheKey, host []byte) {
	if _, ok := s.residentQuant[key]; !ok {
		s.ensureVramHeadroom(uint64(len(host)))
		buf := append([]byte(nil), host...)
		s.residentBytes += uint64(len(buf))
		s.residentQuant[key] = buf
		s.orphanQuantKeys = append(s.orphanQuantKeys, key)
		s.enforceBudgetProtecting(&key)
	} else {
		s.touchOrphanQuant(key)
	}
}

// evictLayerInternal drops a layer's weights when no other layer references
// them. Mirrors GpuState::evict_layer_internal.
func (s *GpuState) evictLayerInternal(layer LayerID) {
	entry, ok := s.layerMap[layer]
	if !ok {
		return
	}
	delete(s.layerMap, layer)
	for _, key := range entry.f32Keys {
		if s.keyReferencedF32(key) {
			continue
		}
		if buf, ok := s.residentF32[key]; ok {
			s.residentBytes -= uint64(len(buf) * sizeOfF32)
			delete(s.residentF32, key)
		}
	}
	for _, key := range entry.f16Keys {
		if s.keyReferencedF16(key) {
			continue
		}
		if buf, ok := s.residentF16[key]; ok {
			s.residentBytes -= uint64(len(buf) * sizeOfU16)
			delete(s.residentF16, key)
		}
	}
}

func (s *GpuState) keyReferencedF32(key weightCacheKey) bool {
	for _, e := range s.layerMap {
		for _, k := range e.f32Keys {
			if k == key {
				return true
			}
		}
	}
	return false
}

func (s *GpuState) keyReferencedF16(key weightCacheKey) bool {
	for _, e := range s.layerMap {
		for _, k := range e.f16Keys {
			if k == key {
				return true
			}
		}
	}
	return false
}
