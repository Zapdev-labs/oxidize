package autotune

import "github.com/Zapdev-labs/oxidize/golang/core/kv_cache"

// PlanJSON is a JSON-friendly snapshot of a TuningPlan.
type PlanJSON struct {
	Threads           int      `json:"threads"`
	CtxSize           int      `json:"ctx_size"`
	KVCacheDType      string   `json:"kv_cache_dtype"`
	KVQuantization    string   `json:"kv_quantization"`
	NGPULayers        int      `json:"n_gpu_layers"`
	Mmap              bool     `json:"mmap"`
	Mlock             bool     `json:"mlock"`
	LayerWise         bool     `json:"layer_wise"`
	LayerCache        int      `json:"layer_cache"`
	Pipeline          string   `json:"pipeline"`
	Speculative       string   `json:"speculative"`
	DecodeTileTokens  int      `json:"decode_tile_tokens"`
	OxkISA            string   `json:"oxk_isa"`
	OxkTile           int      `json:"oxk_tile"`
	ExpectedPromptTPS float32  `json:"expected_prompt_tps"`
	ExpectedDecodeTPS float32  `json:"expected_decode_tps"`
	Rationale         []string `json:"rationale"`
}

// PlanJSON converts a plan to a JSON-serializable struct.
func ToPlanJSON(plan *TuningPlan) PlanJSON {
	return PlanJSON{
		Threads:           plan.Threads,
		CtxSize:           plan.CtxSize,
		KVCacheDType:      plan.KVCacheDType.String(),
		KVQuantization:    kvQuantString(plan.KVQuantization),
		NGPULayers:        plan.NGPULayers,
		Mmap:              plan.Mmap,
		Mlock:             plan.Mlock,
		LayerWise:         plan.LayerWise,
		LayerCache:        plan.LayerCache,
		Pipeline:          pipelineString(plan.Pipeline),
		Speculative:       plan.Speculative.String(),
		DecodeTileTokens:  plan.DecodeTileTokens,
		OxkISA:            oxkISAString(plan.OxkISA),
		OxkTile:           oxkTileInt(plan.OxkTile),
		ExpectedPromptTPS: plan.ExpectedPromptTPS,
		ExpectedDecodeTPS: plan.ExpectedDecodeTPS,
		Rationale:         append([]string(nil), plan.Rationale...),
	}
}

func kvQuantString(q kv_cache.Quantization) string {
	switch q {
	case kv_cache.QuantAsymmetric:
		return "asymmetric"
	case kv_cache.QuantTurboQuant:
		return "turboquant"
	default:
		return "unknown"
	}
}

func oxkISAString(isa OxkIsa) string {
	switch isa {
	case OxkAvx2:
		return "avx2"
	case OxkAvx512:
		return "avx512"
	default:
		return "scalar"
	}
}

func oxkTileInt(tile OxkTile) int {
	switch tile {
	case OxkT4:
		return 4
	case OxkT8:
		return 8
	case OxkT16:
		return 16
	default:
		return 1
	}
}
