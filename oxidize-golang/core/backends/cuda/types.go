package cudabackend

import (
	"sync"
)

// GgmlType mirrors the numeric GGML tensor type ids used to dispatch quantized
// GEMV kernels. Values match the upstream GGML enum so the same integer flows
// through the Rust, Go, and Python ports unchanged.
type GgmlType int

const (
	GgmlTypeF32  GgmlType = 0
	GgmlTypeF16  GgmlType = 1
	GgmlTypeQ4_0 GgmlType = 2
	GgmlTypeQ4_1 GgmlType = 3
	GgmlTypeQ8_0 GgmlType = 8
	GgmlTypeQ2_K GgmlType = 10
	GgmlTypeQ4_K GgmlType = 12
	GgmlTypeQ6_K GgmlType = 14
)

// dequantKernel describes the on-the-fly dequant kernel for a quantized type:
// the kernel name (informational, mirrors Rust), the raw block size in bytes,
// and how many decoded f32 values a block produces.
type dequantKernel struct {
	name       string
	blockBytes int
	valsPerBlk int
}

// dequantKernelFor mirrors gemv_quantized.rs:dequant_kernel_for. It returns the
// kernel descriptor for a type, or ok=false when no GPU dequant path exists and
// the caller must fall back to the CPU quantized path.
func dequantKernelFor(t GgmlType) (dequantKernel, bool) {
	switch t {
	case GgmlTypeQ8_0:
		return dequantKernel{"dequant_q8_0_kernel", 34, 32}, true
	case GgmlTypeQ4_K:
		return dequantKernel{"dequant_q4_k_kernel", 144, 256}, true
	case GgmlTypeQ6_K:
		return dequantKernel{"dequant_q6_k_kernel", 210, 256}, true
	case GgmlTypeQ2_K:
		return dequantKernel{"dequant_q2_k_kernel", 84, 256}, true
	case GgmlTypeQ4_0:
		return dequantKernel{"dequant_q4_0_kernel", 18, 32}, true
	default:
		return dequantKernel{}, false
	}
}

// SupportsQuantizedGpu mirrors gemv_quantized.rs:supports_quantized_gpu.
// Callers should fall back to the CPU quantized path when this returns false.
func SupportsQuantizedGpu(t GgmlType) bool {
	_, ok := dequantKernelFor(t)
	return ok
}

// LayerID tags a group of weight matrices as belonging to the same model layer.
// The inference engine calls PreloadLayer before a forward pass and EvictLayer
// when the layer is no longer needed.
type LayerID = uint32

// CudaLayerConfig configures layer-by-layer VRAM management (AirLLM-style).
// Mirrors types.rs:CudaLayerConfig.
type CudaLayerConfig struct {
	// MaxResidentLayers is the maximum number of layers to keep resident in
	// VRAM at once. 0 = unlimited (default, loads all layers).
	MaxResidentLayers int
	// MaxVramBytes is the maximum VRAM bytes to use for weight caching.
	// 0 = unlimited (default).
	MaxVramBytes uint64
}

// layerEntry records the resident weight keys owned by a layer and the bytes
// they consume. Mirrors types.rs:LayerEntry.
type layerEntry struct {
	f32Keys []weightCacheKey
	f16Keys []weightCacheKey
	bytes   uint64
}

// weightCacheKey identifies a resident weight matrix. The Rust port keys by
// (pointer, len, content_hash); in pure-Go we key by a stable content hash plus
// length so identical host buffers map to the same resident entry.
type weightCacheKey struct {
	hash uint64
	len  int
}

// GpuActivationBuffer holds GPU-resident activation buffers for a single decode
// step, allocated once and reused across tokens. Mirrors
// types.rs:GpuActivationBuffer. In the pure-Go build these are host-side slices.
type GpuActivationBuffer struct {
	Hidden           []float32
	Normed           []float32
	FfnGate          []float32
	FfnUp            []float32
	FfnDownIn        []float32
	HiddenSize       int
	IntermediateSize int
}

// GpuState mirrors types.rs:GpuState: a per-context bundle of resident weight
type GpuState struct {
	mu sync.Mutex

	residentF32   map[weightCacheKey][]float32
	residentF16   map[weightCacheKey][]uint16
	residentQuant map[weightCacheKey][]byte

	f32Pool map[int][][]float32
	q8kPool map[int][][]byte

	layerConfig CudaLayerConfig
	layerLRU    []LayerID
	layerMap    map[LayerID]*layerEntry

	residentBytes uint64

	orphanF16Keys   []weightCacheKey
	orphanQuantKeys []weightCacheKey

	activation *GpuActivationBuffer
}

func newGpuState() *GpuState {
	return &GpuState{
		residentF32:   make(map[weightCacheKey][]float32),
		residentF16:   make(map[weightCacheKey][]uint16),
		residentQuant: make(map[weightCacheKey][]byte),
		f32Pool:       make(map[int][][]float32),
		q8kPool:       make(map[int][][]byte),
		layerMap:      make(map[LayerID]*layerEntry),
	}
}
