package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/autotune"
	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
)

type flagVisits map[string]bool

func (v flagVisits) set(name string) { v[name] = true }
func (v flagVisits) wasSet(name string) bool { return v[name] }

// applyAutotune fingerprints the model, optionally prints the plan, and fills unset flags.
func applyAutotune(modelPath string, opts *genOptions, visits flagVisits, stderr io.Writer) error {
	if opts.NoAuto || !opts.Auto {
		return nil
	}
	mapped, err := ggufcore.LoadMapped(modelPath)
	if err != nil {
		return err
	}
	inv := autotune.Detect()
	fp := autotune.Fingerprint(mapped)
	plan := autotune.Plan(&inv, &fp)
	if shouldPrintPlan(opts.PrintPlan) {
		if opts.PrintPlan == "json" {
			data, err := json.MarshalIndent(autotune.ToPlanJSON(&plan), "", "  ")
			if err != nil {
				return err
			}
			_, _ = fmt.Fprintln(stderr, string(data))
		} else {
			_, _ = fmt.Fprintf(stderr, "\n[oxidize auto-tune plan]\n%s", plan.Summary())
		}
	}
	overrides := autotune.OverridesFromPlan(&plan)
	if !visits.wasSet("threads") && overrides.Threads != nil && *overrides.Threads > 0 {
		opts.Threads = *overrides.Threads
	}
	if !visits.wasSet("ctx-size") && overrides.CtxSize != nil && *overrides.CtxSize > 0 {
		opts.CtxSize = *overrides.CtxSize
	}
	if !visits.wasSet("n-gpu-layers") && overrides.NGPULayers != nil {
		opts.NGPULayers = *overrides.NGPULayers
	}
	if !visits.wasSet("layer-cache") && overrides.LayerCache != nil && *overrides.LayerCache > 0 {
		opts.LayerCache = *overrides.LayerCache
	}
	if !visits.wasSet("layer-wise") && overrides.LayerWise != nil && *overrides.LayerWise {
		opts.LayerWise = true
	}
	if !visits.wasSet("paged") && overrides.Pipeline != nil && *overrides.Pipeline == "paged" {
		opts.UsePaged = true
	}
	if !visits.wasSet("ram-offload") && overrides.RAMOffload != nil && *overrides.RAMOffload {
		opts.RAMOffload = true
	}
	if plan.Speculative == autotune.SpeculativeDFlash && !visits.wasSet("dflash-fusion") && opts.DraftModel == "" {
		opts.DFlashFusion = true
	}
	_, _ = fmt.Fprintf(stderr,
		"[oxidize auto-tune] applied: threads=%d ctx=%d n_gpu_layers=%d layer_wise=%t layer_cache=%d paged=%t (cores=%d ram=%d GiB gpu=%d MiB)\n",
		opts.Threads, opts.CtxSize, opts.NGPULayers, opts.LayerWise, opts.LayerCache, opts.UsePaged,
		inv.PhysicalCores, inv.TotalRAMBytes/(1<<30), inv.GPUVRAMBytes/(1024*1024),
	)
	return nil
}

func shouldPrintPlan(mode string) bool {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "json", "yes", "true", "1":
		return true
	case "no", "false", "0":
		return false
	case "auto":
		fi, err := os.Stderr.Stat()
		if err != nil {
			return true
		}
		return (fi.Mode() & os.ModeCharDevice) != 0
	default:
		return true
	}
}
