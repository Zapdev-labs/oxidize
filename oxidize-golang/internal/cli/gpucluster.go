package cli

import (
	"flag"
	"fmt"
	"io"

	"github.com/Zapdev-labs/oxidize/golang/core/gpucluster"
)

func gpuClusterCommand(args []string, stdout, stderr io.Writer) error {
	sub := "help"
	if len(args) > 0 {
		sub = args[0]
	}
	switch sub {
	case "profiles":
		for _, p := range gpucluster.AllProfiles() {
			_, _ = fmt.Fprintf(stdout,
				"%-14s product=%-26s arch=%-9s mem=%dMiB tdp=%dW nvlink=%v mig=%v timeslice=%d net=%s\n",
				p.Family.Slug(), p.Product, p.Generation, p.MemoryMiB, p.TDPWatts,
				p.NVLink, p.MIGCapable, p.TimeSliceReplicas, p.NetworkClass)
		}
		return nil
	case "detect":
		gpus := gpucluster.DetectGPUs()
		if len(gpus) == 0 {
			_, _ = fmt.Fprintln(stdout, "no NVIDIA GPUs detected (nvidia-smi unavailable or no devices)")
			return nil
		}
		for _, g := range gpus {
			fam := "unknown"
			if g.FamilyKnown {
				fam = g.Family.Slug()
			}
			_, _ = fmt.Fprintf(stdout, "GPU %d: %s (%dMiB) family=%s mig=%v\n",
				g.Index, g.Name, g.MemoryTotalMiB, fam, g.MIGEnabled)
		}
		_, _ = fmt.Fprintln(stdout, "--- summary ---")
		for _, c := range gpucluster.Summarize(gpus) {
			_, _ = fmt.Fprintf(stdout, "%s: %d\n", c.Family.Slug(), c.Count)
		}
		return nil
	case "generate":
		return gpuClusterGenerate(args[1:], stdout, stderr)
	default:
		_, _ = fmt.Fprintln(stdout,
			"usage: oxidize gpu-cluster <generate|detect|profiles>\n\n"+
				"generate [--family b200|a100|rtx-pro-6000] [--nodes N] [--gpus-per-node N]\n"+
				"detect   probe local NVIDIA GPUs via nvidia-smi\n"+
				"profiles list known GPU tier profiles")
		return nil
	}
}

func gpuClusterGenerate(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("gpu-cluster generate", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	familyStr := fs.String("family", "", "b200|a100|rtx-pro-6000 (default: full three-tier cluster)")
	nodes := fs.Uint("nodes", 0, "node count")
	gpusPerNode := fs.Uint("gpus-per-node", 0, "physical GPUs per node")
	if err := fs.Parse(args); err != nil {
		return err
	}

	var specs []gpucluster.NodePoolSpec
	if *familyStr != "" {
		fam, ok := gpucluster.FamilyFromSlug(*familyStr)
		if !ok {
			_, _ = fmt.Fprintln(stderr, "error: --family expects b200|a100|rtx-pro-6000")
			return fmt.Errorf("invalid family %q", *familyStr)
		}
		count := uint32(*nodes)
		if count == 0 {
			count = defaultNodeCount(fam)
		}
		gpn := uint32(*gpusPerNode)
		if gpn == 0 {
			gpn = defaultGPUsPerNode(fam)
		}
		specs = []gpucluster.NodePoolSpec{{Family: fam, NodeCount: count, GPUPerNode: gpn}}
	} else {
		specs = []gpucluster.NodePoolSpec{
			{Family: gpucluster.B200, NodeCount: 8, GPUPerNode: 8},
			{Family: gpucluster.A100, NodeCount: 16, GPUPerNode: 8},
			{Family: gpucluster.RTXPro6000, NodeCount: 4, GPUPerNode: 2},
		}
	}

	families := make([]gpucluster.Family, 0, len(specs))
	for _, s := range specs {
		families = append(families, s.Family)
	}

	_, _ = fmt.Fprint(stdout, gpucluster.NodePoolsYAML(specs))
	_, _ = fmt.Fprintln(stdout, "---")
	_, _ = fmt.Fprint(stdout, gpucluster.DevicePluginConfigYAML(families))
	for _, f := range families {
		if mig, ok := gpucluster.MIGConfigYAML(f); ok {
			_, _ = fmt.Fprintln(stdout, "---")
			_, _ = fmt.Fprint(stdout, mig)
		}
	}
	_, _ = fmt.Fprintln(stdout, "---")
	_, _ = fmt.Fprint(stdout, gpucluster.PrometheusRulesYAML())
	for _, f := range families {
		_, _ = fmt.Fprintln(stdout, "---")
		_, _ = fmt.Fprint(stdout, gpucluster.HelmValuesYAML(f))
	}
	return nil
}

func defaultNodeCount(f gpucluster.Family) uint32 {
	switch f {
	case gpucluster.B200:
		return 8
	case gpucluster.A100:
		return 16
	case gpucluster.RTXPro6000:
		return 4
	default:
		return 1
	}
}

func defaultGPUsPerNode(f gpucluster.Family) uint32 {
	switch f {
	case gpucluster.RTXPro6000:
		return 2
	default:
		return 8
	}
}
