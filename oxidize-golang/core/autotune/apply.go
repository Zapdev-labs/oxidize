package autotune

import "github.com/Zapdev-labs/oxidize/golang/core/kv_cache"

// PlanOverrides holds per-flag autotune recommendations for CLI/server apply.
type PlanOverrides struct {
	Threads        *int
	CtxSize        *int
	NGPULayers     *int
	LayerCache     *int
	LayerWise      *bool
	Mmap           *bool
	Mlock          *bool
	MmapHugepages  *bool
	MmapPrefetch   *bool
	RAMOffload     *bool
	CPUOptimized   *bool
	TurboQuant     *bool
	Pipeline       *string
	DecodeTile     *int
}

// OverridesFromPlan converts a tuning plan into flag overrides.
func OverridesFromPlan(plan *TuningPlan) PlanOverrides {
	pipeline := pipelineString(plan.Pipeline)
	turbo := plan.KVQuantization == kv_cache.QuantTurboQuant
	cpuOpt := false
	decodeTile := (*int)(nil)
	if plan.DecodeTileTokens > 0 {
		dt := plan.DecodeTileTokens
		decodeTile = &dt
	}
	return PlanOverrides{
		Threads:       &plan.Threads,
		CtxSize:       &plan.CtxSize,
		NGPULayers:    &plan.NGPULayers,
		LayerCache:    &plan.LayerCache,
		LayerWise:     &plan.LayerWise,
		Mmap:          &plan.Mmap,
		Mlock:         &plan.Mlock,
		MmapHugepages: &plan.MmapHugepages,
		MmapPrefetch:  &plan.MmapPrefetch,
		RAMOffload:    &plan.Mlock,
		CPUOptimized:  &cpuOpt,
		TurboQuant:    &turbo,
		Pipeline:      &pipeline,
		DecodeTile:    decodeTile,
	}
}

func pipelineString(mode PipelineMode) string {
	switch mode {
	case PipelineSequential:
		return "sequential"
	case PipelineContinuous:
		return "continuous"
	case PipelinePaged:
		return "paged"
	case PipelineAsymmetric:
		return "asymmetric"
	default:
		return "sequential"
	}
}
