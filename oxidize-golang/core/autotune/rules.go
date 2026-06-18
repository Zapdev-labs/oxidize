package autotune

import (
	"fmt"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/gpucluster"
	"github.com/Zapdev-labs/oxidize/golang/core/kv_cache"
	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
	"github.com/Zapdev-labs/oxidize/golang/core/simd"
	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
)

// PipelineMode is the batch / scheduling mode.
type PipelineMode int

const (
	PipelineSequential PipelineMode = iota
	PipelineContinuous
	PipelinePaged
	PipelineAsymmetric
)

func (p PipelineMode) String() string {
	switch p {
	case PipelineSequential:
		return "Sequential"
	case PipelineContinuous:
		return "Continuous"
	case PipelinePaged:
		return "Paged"
	case PipelineAsymmetric:
		return "Asymmetric"
	default:
		return "Unknown"
	}
}

// SpeculativeSpec recommends a speculative decoding strategy.
type SpeculativeSpec int

const (
	SpeculativeNone SpeculativeSpec = iota
	SpeculativeDFlash
	SpeculativeMTP
)

func (s SpeculativeSpec) String() string {
	switch s {
	case SpeculativeNone:
		return "None"
	case SpeculativeDFlash:
		return "DFlash"
	case SpeculativeMTP:
		return "Mtp"
	default:
		return "Unknown"
	}
}

// OxkIsa is the oxidize-kernels ISA selection.
type OxkIsa int

const (
	OxkScalar OxkIsa = iota
	OxkAvx2
	OxkAvx512
)

// OxkTile is the oxidize-kernels tile width.
type OxkTile int

const (
	OxkT1 OxkTile = iota
	OxkT4
	OxkT8
	OxkT16
)

// TuningPlan is a fully-resolved autotune recommendation.
type TuningPlan struct {
	Threads              int
	CtxSize              int
	KVCacheDType         tensor.DType
	KVQuantization       kv_cache.Quantization
	NGPULayers           int
	GPUSplit             []float32
	Mmap                 bool
	Mlock                bool
	MmapHugepages        bool
	MmapPrefetch         bool
	NumaReplicateDense   bool
	LayerWise            bool
	LayerCache           int
	Pipeline             PipelineMode
	Speculative          SpeculativeSpec
	DecodeTileTokens     int
	OxkISA               OxkIsa
	OxkTile              OxkTile
	ExpectedPromptTPS    float32
	ExpectedDecodeTPS    float32
	Rationale            []string
}

// Summary returns a human-readable plan summary.
func (p TuningPlan) Summary() string {
	var b strings.Builder
	fmt.Fprintf(&b, "threads           : %d\n", p.Threads)
	fmt.Fprintf(&b, "ctx_size          : %d\n", p.CtxSize)
	fmt.Fprintf(&b, "kv_cache_dtype    : %s (quantization: %v)\n", p.KVCacheDType, p.KVQuantization)
	fmt.Fprintf(&b, "n_gpu_layers      : %d\n", p.NGPULayers)
	if len(p.GPUSplit) > 0 {
		fmt.Fprintf(&b, "gpu_split         : %v\n", p.GPUSplit)
	}
	fmt.Fprintf(&b, "mmap=%t mlock=%t mmap_hugepages=%t mmap_prefetch=%t\n",
		p.Mmap, p.Mlock, p.MmapHugepages, p.MmapPrefetch)
	fmt.Fprintf(&b, "numa_replicate    : %t\n", p.NumaReplicateDense)
	fmt.Fprintf(&b, "layer_wise=%t layer_cache=%d\n", p.LayerWise, p.LayerCache)
	fmt.Fprintf(&b, "pipeline          : %s\n", p.Pipeline)
	fmt.Fprintf(&b, "speculative       : %s\n", p.Speculative)
	fmt.Fprintf(&b, "decode_tile_tokens: %d\n", p.DecodeTileTokens)
	fmt.Fprintf(&b, "oxk_isa/tile      : %v / %v\n", p.OxkISA, p.OxkTile)
	fmt.Fprintf(&b, "expected t/s      : prompt ≈ %.1f  decode ≈ %.1f\n",
		p.ExpectedPromptTPS, p.ExpectedDecodeTPS)
	if len(p.Rationale) > 0 {
		b.WriteString("\nRationale:\n")
		for _, r := range p.Rationale {
			fmt.Fprintf(&b, "  - %s\n", r)
		}
	}
	return b.String()
}

// Plan builds a tuning plan for the given hardware and model.
func Plan(inv *HardwareInventory, model *ModelFingerprint) TuningPlan {
	plan := TuningPlan{
		KVCacheDType:   tensor.DTypeF32,
		KVQuantization: kv_cache.QuantAsymmetric,
		Mmap:           true,
		Pipeline:       PipelineSequential,
		Speculative:    SpeculativeNone,
		OxkISA:         OxkScalar,
		OxkTile:        OxkT1,
	}
	tier0HardRules(inv, model, &plan)
	tier1ISA(inv, &plan)
	tier2GPUOffload(inv, model, &plan)
	tier3KVAndCtx(inv, model, &plan)
	tier4LayerCacheAndNUMA(inv, model, &plan)
	tier5Speculative(inv, model, &plan)
	tier6Threads(inv, &plan)
	tier7DecodeTile(&plan)
	tier8Pipeline(inv, model, &plan)
	estimateTPS(inv, model, &plan)
	return plan
}

func tier0HardRules(inv *HardwareInventory, model *ModelFingerprint, plan *TuningPlan) {
	ramBudget := effectiveRAMBytes(inv)
	if ramBudget < model.FileSizeBytes*12/10 {
		plan.Mmap = true
		plan.Mlock = false
		plan.LayerWise = true
		plan.LayerCache = max(inv.PhysicalCores/4, 1)
		plan.Rationale = append(plan.Rationale, fmt.Sprintf(
			"model (%.1f GiB) exceeds 1.2× effective RAM (%.1f GiB) → streaming layers, mmap=ON, mlock=OFF, layer_wise=ON, layer_cache=%d",
			float64(model.FileSizeBytes)/(1<<30),
			float64(ramBudget)/(1<<30),
			plan.LayerCache,
		))
	} else {
		plan.Rationale = append(plan.Rationale, fmt.Sprintf(
			"model (%.1f GiB) fits in effective RAM (%.1f GiB) → mmap=ON, mlock=OFF by default",
			float64(model.FileSizeBytes)/(1<<30),
			float64(ramBudget)/(1<<30),
		))
	}
	if model.IsMoE && inv.PhysicalCores <= 8 {
		plan.NumaReplicateDense = false
		plan.Rationale = append(plan.Rationale,
			"MoE on <= 8 cores → NUMA replication disabled (overhead exceeds benefit)")
	}
	if inv.OS == OsMacos && inv.HasMetal {
		plan.Rationale = append(plan.Rationale,
			"macOS + Metal build available → keep --backend cpu (Metal auto-promotion lives in runtime)")
	}
}

func tier1ISA(inv *HardwareInventory, plan *TuningPlan) {
	switch inv.SIMD {
	case simd.BackendAvx512f:
		if IsSkylakeSP() {
			plan.OxkISA = OxkAvx2
			plan.OxkTile = OxkT8
			plan.Rationale = append(plan.Rationale,
				"Skylake-SP detected → AVX-512 disabled; AVX2 x8")
		} else {
			plan.OxkISA = OxkAvx512
			plan.OxkTile = OxkT8
			plan.Rationale = append(plan.Rationale,
				"AVX-512F available + non-Skylake → AVX-512 x8")
		}
	case simd.BackendAvx2:
		plan.OxkISA = OxkAvx2
		if inv.PhysicalCores >= 16 {
			plan.OxkTile = OxkT8
			plan.Rationale = append(plan.Rationale, "AVX2 only → AVX2 x8")
		} else {
			plan.OxkTile = OxkT4
			plan.Rationale = append(plan.Rationale, "AVX2 only → AVX2 x4")
		}
	case simd.BackendNeon:
		plan.OxkISA = OxkScalar
		plan.OxkTile = OxkT1
		plan.Rationale = append(plan.Rationale, "ARM/Neon → scalar oxk (no Neon kernel yet)")
	default:
		plan.OxkISA = OxkScalar
		plan.OxkTile = OxkT1
		plan.Rationale = append(plan.Rationale, "No SIMD beyond SSE2 → scalar oxk")
	}
}

func tier2GPUOffload(inv *HardwareInventory, model *ModelFingerprint, plan *TuningPlan) {
	if !inv.HasGPU && !inv.HasROCm && !inv.HasCUDA {
		plan.NGPULayers = 0
		return
	}
	if !inv.HasGPU {
		plan.NGPULayers = 0
		if inv.HasROCm {
			plan.Rationale = append(plan.Rationale,
				"ROCm build detected but no GPU inventory — set --backend rocm and pass --n-gpu-layers manually")
		}
		return
	}
	perLayer := PerLayerWeightBytes(*model)
	if perLayer == 0 {
		plan.NGPULayers = 0
		return
	}
	usableVRAM := uint64(float64(inv.GPUVRAMBytes) * 0.85)
	n := int(usableVRAM / perLayer)
	if inv.GPUVRAMBytes < model.FileSizeBytes/4 {
		n = 0
		plan.Rationale = append(plan.Rationale, fmt.Sprintf(
			"GPU VRAM (%.1f GiB) < 25%% of model size (%.1f GiB) → n_gpu_layers=0",
			float64(inv.GPUVRAMBytes)/(1<<30),
			float64(model.FileSizeBytes)/(1<<30),
		))
	} else {
		if n > model.LayerCount {
			n = model.LayerCount
		}
		if n == model.LayerCount {
			plan.Mmap = false
			plan.Mlock = false
			plan.Rationale = append(plan.Rationale, fmt.Sprintf(
				"GPU can hold the full model (%d/%d layers) → mmap=OFF",
				n, model.LayerCount,
			))
		} else {
			plan.Rationale = append(plan.Rationale, fmt.Sprintf(
				"GPU offload: %d/%d layers at %.1f GiB usable VRAM",
				n, model.LayerCount, float64(usableVRAM)/(1<<30),
			))
		}
	}
	plan.NGPULayers = n
}

func tier3KVAndCtx(inv *HardwareInventory, model *ModelFingerprint, plan *TuningPlan) {
	vramGiB := inv.GPUVRAMBytes / (1 << 30)
	switch {
	case inv.HasGPU && vramGiB >= 16:
		plan.KVCacheDType = tensor.DTypeF16
		plan.KVQuantization = kv_cache.QuantAsymmetric
		plan.Rationale = append(plan.Rationale, ">= 16 GiB VRAM → kv=F16")
	case (inv.HasGPU && vramGiB >= 8) || model.LayerCount >= 80:
		plan.KVCacheDType = tensor.DTypeF16
		plan.KVQuantization = kv_cache.QuantAsymmetric
		plan.Rationale = append(plan.Rationale, "8-16 GiB VRAM or deep model → kv=F16 + asymmetric")
	case vramGiB < 8 || model.LayerCount >= 60 || inv.TotalRAMBytes < (32<<30):
		plan.KVCacheDType = tensor.DTypeF16
		plan.KVQuantization = kv_cache.QuantTurboQuant
		plan.Rationale = append(plan.Rationale, "low VRAM / RAM or very deep model → kv=F16 + TurboQuant")
	default:
		plan.KVCacheDType = tensor.DTypeF16
		plan.KVQuantization = kv_cache.QuantAsymmetric
	}

	ramBudget := effectiveRAMBytes(inv)
	overhead := uint64(8 << 30)
	var kvBudget uint64
	if ramBudget > model.FileSizeBytes+overhead {
		kvBudget = ramBudget - model.FileSizeBytes - overhead
	} else {
		kvBudget = 0
	}
	kvBytes := KVBytesPerToken(*model, 2)
	ctxCap := 4096
	if kvBytes > 0 {
		cap := int(kvBudget / kvBytes)
		if cap < ctxCap {
			ctxCap = cap
		}
		if ctxCap > 131072 {
			ctxCap = 131072
		}
	}
	defaultCtx := 4096
	if model.NumKVHeads <= 4 {
		defaultCtx = 8192
	}
	if defaultCtx > ctxCap {
		defaultCtx = ctxCap
	}
	if defaultCtx < 512 {
		defaultCtx = 512
	}
	plan.CtxSize = defaultCtx
	plan.Rationale = append(plan.Rationale, fmt.Sprintf(
		"ctx_size=%d (capped to fit %d bytes of KV)", plan.CtxSize, kvBudget,
	))
}

func tier4LayerCacheAndNUMA(inv *HardwareInventory, model *ModelFingerprint, plan *TuningPlan) {
	if plan.NGPULayers == model.LayerCount && model.LayerCount > 0 {
		plan.LayerCache = 0
		plan.NumaReplicateDense = false
		return
	}
	if plan.LayerCache == 0 {
		plan.LayerCache = clamp(inv.PhysicalCores, 2, 8)
		plan.Rationale = append(plan.Rationale, fmt.Sprintf(
			"layer_cache=%d (~1 layer per 2 cores, capped at 8)", plan.LayerCache,
		))
	}
	if inv.NumaNodes >= 2 && inv.PhysicalCores >= 16 && !model.IsMoE && plan.OxkISA != OxkScalar {
		plan.NumaReplicateDense = true
		plan.Rationale = append(plan.Rationale,
			"NUMA nodes>=2, cores>=16, dense model, SIMD available → NUMA-replicate dense weights")
	}
}

func tier5Speculative(inv *HardwareInventory, model *ModelFingerprint, plan *TuningPlan) {
	if !inv.HasGPU {
		return
	}
	if model.HasMTP {
		plan.Speculative = SpeculativeMTP
		plan.Rationale = append(plan.Rationale,
			"model has MTP tensors + GPU → suggest MTP speculative decoding")
		return
	}
	if isDFlashCompatible(model.Architecture) {
		plan.Speculative = SpeculativeDFlash
		plan.Rationale = append(plan.Rationale, fmt.Sprintf(
			"%s on GPU → suggest DFlash speculative decoding", model.Architecture,
		))
	}
}

func isDFlashCompatible(arch string) bool {
	switch arch {
	case "qwen2", "qwen3", "llama", "lfm2":
		return true
	default:
		return false
	}
}

func tier6Threads(inv *HardwareInventory, plan *TuningPlan) {
	if inv.HasGPU && plan.NGPULayers > 0 && plan.OxkISA != OxkScalar {
		plan.Threads = max(inv.PhysicalCores/8, 4)
		plan.Rationale = append(plan.Rationale,
			"GPU does most work → CPU threads kept low to avoid contention")
		return
	}
	if inv.ContainerMemLimit != nil {
		plan.Threads = clamp(inv.PhysicalCores, 2, 8)
		plan.Rationale = append(plan.Rationale,
			"container memory limit present → cap threads")
		return
	}
	plan.Threads = inv.PhysicalCores
	plan.Rationale = append(plan.Rationale, fmt.Sprintf(
		"CPU-only path → threads = physical_cores (%d)", inv.PhysicalCores,
	))
}

func tier7DecodeTile(plan *TuningPlan) {
	if plan.CtxSize > 8192 {
		plan.DecodeTileTokens = 1024
		plan.Rationale = append(plan.Rationale, "ctx > 8192 → split-K decode tile = 1024")
	} else if plan.CtxSize > 4096 && plan.OxkISA == OxkAvx2 {
		plan.DecodeTileTokens = 512
		plan.Rationale = append(plan.Rationale, "ctx > 4096 on AVX2 → split-K decode tile = 512")
	}
}

func tier8Pipeline(inv *HardwareInventory, model *ModelFingerprint, plan *TuningPlan) {
	if inv.HasGPU && plan.NGPULayers > 0 {
		plan.Pipeline = PipelinePaged
		plan.Rationale = append(plan.Rationale,
			"GPU + layers on GPU → paged attention (continuous batching)")
		return
	}
	if inv.PhysicalCores >= 8 && inv.TotalRAMBytes >= (64<<30) && !model.IsMoE {
		plan.Pipeline = PipelineContinuous
		plan.Rationale = append(plan.Rationale,
			">= 8 cores, >= 64 GiB, dense model → continuous batching")
		return
	}
	plan.Pipeline = PipelineSequential
	plan.Rationale = append(plan.Rationale, "low-resource or MoE → sequential (default)")
}

func estimateTPS(inv *HardwareInventory, model *ModelFingerprint, plan *TuningPlan) {
	perCore := perCoreDecodeTPS(*model)
	cpuTPS := float32(inv.PhysicalCores) * perCore
	memBW := float32(inv.TotalRAMBytes) * 0.7
	memTPS := float32(0)
	if model.FileSizeBytes > 0 {
		memTPS = memBW / float32(model.FileSizeBytes)
	}
	cpuBranch := cpuTPS
	if memTPS < cpuBranch {
		cpuBranch = memTPS
	}
	gpuTPS := float32(0)
	if inv.HasGPU {
		if inv.GPUFamily != nil {
			switch *inv.GPUFamily {
			case gpucluster.B200:
				gpuTPS = 200
			case gpucluster.A100:
				gpuTPS = 90
			case gpucluster.RTXPro6000:
				gpuTPS = 70
			default:
				gpuTPS = 30
			}
		} else {
			gpuTPS = 30
		}
	}
	if inv.HasGPU && plan.NGPULayers > 0 {
		plan.ExpectedDecodeTPS = gpuTPS
	} else {
		plan.ExpectedDecodeTPS = cpuBranch
	}
	plan.ExpectedPromptTPS = plan.ExpectedDecodeTPS * 6
}

func perCoreDecodeTPS(model ModelFingerprint) float32 {
	sizeClass := "large"
	if model.FileSizeBytes <= 8<<30 {
		sizeClass = "small"
	} else if model.FileSizeBytes <= 30<<30 {
		sizeClass = "medium"
	}
	switch model.Quant {
	case quantization.TypeQ4_K_M, quantization.TypeQ4_K_S:
		switch sizeClass {
		case "small":
			return 1.2
		case "medium":
			return 0.6
		default:
			return 0.25
		}
	case quantization.TypeQ2_K, quantization.TypeQ3_K_S:
		switch sizeClass {
		case "small":
			return 1.6
		case "medium":
			return 0.8
		default:
			return 0.35
		}
	case quantization.TypeQ8_0:
		return 0.8
	case quantization.TypeF16:
		return 0.4
	case quantization.TypeQ5_K_M, quantization.TypeQ5_K_S:
		switch sizeClass {
		case "small":
			return 0.9
		case "medium":
			return 0.45
		default:
			return 0.20
		}
	case quantization.TypeQ6_K:
		switch sizeClass {
		case "small":
			return 0.7
		case "medium":
			return 0.35
		default:
			return 0.18
		}
	default:
		return 0.5
	}
}

func effectiveRAMBytes(inv *HardwareInventory) uint64 {
	if inv.ContainerMemLimit != nil {
		if *inv.ContainerMemLimit < inv.TotalRAMBytes {
			return *inv.ContainerMemLimit
		}
	}
	return inv.TotalRAMBytes
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
