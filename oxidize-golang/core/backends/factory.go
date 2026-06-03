package backends

import (
	"fmt"
	"runtime"

	"github.com/Zapdev-labs/oxidize/golang/core/backend"
	cpubackend "github.com/Zapdev-labs/oxidize/golang/core/backends/cpu"
	cudabackend "github.com/Zapdev-labs/oxidize/golang/core/backends/cuda"
	metalbackend "github.com/Zapdev-labs/oxidize/golang/core/backends/metal"
	vulkanbackend "github.com/Zapdev-labs/oxidize/golang/core/backends/vulkan"
)

// FactoryResult is returned by NewComputeBackend.
type FactoryResult struct {
	Backend   backend.ComputeBackend
	Requested backend.Backend
	Effective backend.Backend
	Warning   string
	FellBack  bool
}

// NewComputeBackend selects a compute backend by name. GPU backends that are not
// linked in this build fall back to the CPU implementation when allowFallback is true.
func NewComputeBackend(name string, allowFallback bool) (FactoryResult, error) {
	requested, err := backend.ParseBackend(name)
	if err != nil {
		if allowFallback {
			return FactoryResult{
				Backend:   cpubackend.New(),
				Requested: backend.BackendCpu,
				Effective: backend.BackendCpu,
				Warning:   fmt.Sprintf("unknown backend %q; using cpu", name),
				FellBack:  true,
			}, nil
		}
		return FactoryResult{}, err
	}
	effective, warn, _ := requested.Effective()
	avail, reason := backendAvailable(effective)
	if avail {
		return FactoryResult{
			Backend:   cpubackend.New(),
			Requested: requested,
			Effective: effective,
			Warning:   warn,
		}, nil
	}
	if !allowFallback {
		return FactoryResult{}, fmt.Errorf("backend %s unavailable: %s", effective, reason)
	}
	msg := reason
	if warn != "" {
		msg = warn + "; " + reason
	}
	return FactoryResult{
		Backend:   cpubackend.New(),
		Requested: requested,
		Effective: backend.BackendCpu,
		Warning:   msg,
		FellBack:  true,
	}, nil
}

func backendAvailable(b backend.Backend) (bool, string) {
	switch b {
	case backend.BackendCpu:
		return true, ""
	case backend.BackendMetal:
		if runtime.GOOS != "darwin" {
			return false, "metal requires macOS"
		}
		if !metalbackend.Info().DetectedAtBuild {
			return false, "metal backend not linked in this build"
		}
		return true, ""
	case backend.BackendCuda:
		if !cudabackend.Info().DetectedAtBuild {
			return false, "cuda backend not linked in this build"
		}
		if err := cudabackend.Initialize(); err != nil {
			return false, err.Error()
		}
		return true, ""
	case backend.BackendVulkan, backend.BackendIntelArc:
		if !vulkanbackend.Info().DetectedAtBuild {
			return false, "vulkan backend not linked in this build"
		}
		return true, ""
	case backend.BackendMlx:
		if runtime.GOOS != "darwin" {
			return false, "mlx requires macOS"
		}
		return false, "mlx compute backend not implemented in Go port"
	default:
		return false, "unsupported backend"
	}
}
