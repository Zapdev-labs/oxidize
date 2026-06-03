// Package backend mirrors oxidize_core::backend (the Backend enum and
// ComputeBackend trait). It provides platform-aware backend selection and the
// abstraction that every compute backend (CPU, CUDA, Metal, MLX, Vulkan,
// WebGPU) implements.
package backend

import (
	"fmt"
	"runtime"
	"strings"
)

// Backend enumerates the supported compute backends.
type Backend uint8

const (
	BackendCpu Backend = iota
	BackendMetal
	BackendCuda
	BackendMlx
	BackendVulkan
	BackendIntelArc
)

// String returns the canonical name of the backend.
func (b Backend) String() string {
	switch b {
	case BackendCpu:
		return "cpu"
	case BackendMetal:
		return "metal"
	case BackendCuda:
		return "cuda"
	case BackendMlx:
		return "mlx"
	case BackendVulkan:
		return "vulkan"
	case BackendIntelArc:
		return "intel-arc"
	default:
		return fmt.Sprintf("backend(%d)", uint8(b))
	}
}

// ParseBackend parses a backend name, mirroring Backend::from_str.
func ParseBackend(name string) (Backend, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "cpu":
		return BackendCpu, nil
	case "metal":
		return BackendMetal, nil
	case "cuda":
		return BackendCuda, nil
	case "mlx":
		return BackendMlx, nil
	case "vulkan":
		return BackendVulkan, nil
	case "intel-arc", "arc":
		return BackendIntelArc, nil
	default:
		return 0, fmt.Errorf("unknown backend: %q", name)
	}
}

// Effective returns the runtime backend that should be used for the current
// platform, along with an optional warning message. The MLX backend is only
// available on macOS; on Linux/Windows it is downgraded to CPU.
func (b Backend) Effective() (Backend, string, bool) {
	if b == BackendMlx && runtime.GOOS != "darwin" {
		return BackendCpu, "MLX backend requested but unavailable on Linux; falling back to CPU", true
	}
	if b == BackendIntelArc && !vulkanDetected() {
		return BackendVulkan, "Intel Arc backend requested but Vulkan was not detected at build time; using Vulkan fallback path", true
	}
	return b, "", false
}

func vulkanDetected() bool { return false }

// ComputeBackend is the abstraction every backend implements. It mirrors the
// `ComputeBackend` Rust trait (see oxidize-core/src/backend.rs).
type ComputeBackend interface {
	Name() string
	TensorFromF32(data []float32) (TensorHandle, error)
	TensorFromF32_2D(data []float32, rows, cols int) (TensorHandle, error)
	TensorToF32(tensor TensorHandle, out []float32) (int, error)
	TensorShape(tensor TensorHandle) []int
	TensorDType(tensor TensorHandle) DType

	RmsNorm(input, weight TensorHandle, eps float32) (TensorHandle, error)
	ApplyRope(input TensorHandle, position, headDim int, theta float32) (TensorHandle, error)
	AttentionDecode(query, keyCache, valueCache TensorHandle, seqLen, headDim int, scale float32) (TensorHandle, error)
	Gemv(matrix WeightStorage, vector TensorHandle, rows, cols int) (TensorHandle, error)
	Gemm(a, b TensorHandle, rows, sharedDim, cols int) (TensorHandle, error)
	Add(a, b TensorHandle) (TensorHandle, error)
	Mul(a, b TensorHandle) (TensorHandle, error)
	Sigmoid(x TensorHandle) (TensorHandle, error)
	Softmax(x TensorHandle) (TensorHandle, error)
	Synchronize() error
}

// TensorHandle is the opaque per-backend tensor type. It is implemented by
// every backend (e.g. the CPU backend returns plain *cpu.Tensor).
type TensorHandle any

// WeightStorage is the opaque per-backend weight storage handle.
type WeightStorage any

// DType mirrors tensor::DType; redeclared here to avoid an import cycle in
// the backends that may want to interact with the interface.
type DType uint8

const (
	DTypeF32 DType = iota
	DTypeF16
	DTypeI8
	DTypeI16
	DTypeI32
	DTypeI64
)

func (d DType) String() string {
	switch d {
	case DTypeF32:
		return "f32"
	case DTypeF16:
		return "f16"
	case DTypeI8:
		return "i8"
	case DTypeI16:
		return "i16"
	case DTypeI32:
		return "i32"
	case DTypeI64:
		return "i64"
	default:
		return fmt.Sprintf("dtype(%d)", uint8(d))
	}
}

// SizeInBytes returns the number of bytes occupied by a single element.
func (d DType) SizeInBytes() int {
	switch d {
	case DTypeF32, DTypeI32:
		return 4
	case DTypeF16, DTypeI16:
		return 2
	case DTypeI8:
		return 1
	case DTypeI64:
		return 8
	default:
		return 0
	}
}
