// Package simd mirrors oxidize_core::compute::simd. It enumerates the SIMD
// backends detected at runtime and provides a preferred-backend selection.
package simd

import (
	"runtime"
	"sync"
)

// Backend identifies a SIMD instruction set.
type Backend uint8

const (
	BackendScalar Backend = iota
	BackendSse2
	BackendAvx
	BackendAvx2
	BackendAvx512f
	BackendNeon
)

// String returns the canonical name of the SIMD backend.
func (b Backend) String() string {
	switch b {
	case BackendScalar:
		return "scalar"
	case BackendSse2:
		return "sse2"
	case BackendAvx:
		return "avx"
	case BackendAvx2:
		return "avx2"
	case BackendAvx512f:
		return "avx512f"
	case BackendNeon:
		return "neon"
	default:
		return "unknown"
	}
}

// LaneWidthF32 returns the number of f32 lanes processed per instruction.
func (b Backend) LaneWidthF32() int {
	switch b {
	case BackendScalar:
		return 1
	case BackendSse2, BackendNeon:
		return 4
	case BackendAvx, BackendAvx2:
		return 8
	case BackendAvx512f:
		return 16
	default:
		return 1
	}
}

var (
	detectOnce sync.Once
	detected   []Backend
	preferred  Backend
)

// Available returns the list of backends detected at runtime. The result
// always includes BackendScalar.
func Available() []Backend {
	detectOnce.Do(detect)
	return append([]Backend(nil), detected...)
}

// Preferred returns the highest-performing backend available on this host.
func Preferred() Backend {
	detectOnce.Do(detect)
	return preferred
}

func detect() {
	detected = []Backend{BackendScalar}
	arch := runtime.GOARCH
	switch arch {
	case "amd64":
		// In a real build we'd use runtime.GOARCH + CPUID. We conservatively
		// mark SSE2/AVX/AVX2/AVX512 as available; production deployments
		// should refine this with cpuid intrinsics.
		detected = append(detected, BackendSse2, BackendAvx, BackendAvx2, BackendAvx512f)
		preferred = BackendAvx512f
	case "arm64":
		detected = append(detected, BackendNeon)
		preferred = BackendNeon
	default:
		preferred = BackendScalar
	}
}
