// Package cpubackend implements the ComputeBackend interface on the CPU. It
// mirrors the per-backend struct found in oxidize-core/src/backends/cuda.rs
// (and the CPU path) and uses the tensor package for all kernels.
package cpubackend

import (
	"encoding/binary"
	"math"
	"sync"

	"github.com/Zapdev-labs/oxidize/golang/core/backend"
	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
)

// CpuTensor is the CPU tensor handle.
type CpuTensor struct {
	Data  []float32
	Shape []int
}

// CpuWeightStorage holds quantized weight bytes + dequant metadata.
type CpuWeightStorage struct {
	Bytes   []byte
	Dequant func([]byte, []float32) error
}

// Cpu is the CPU implementation of ComputeBackend.
type Cpu struct {
	mu sync.Mutex
}

// New constructs a CPU backend.
func New() *Cpu { return &Cpu{} }

// Name returns the backend name.
func (c *Cpu) Name() string { return "cpu" }

// TensorFromF32 creates a 1-D tensor.
func (c *Cpu) TensorFromF32(data []float32) (backend.TensorHandle, error) {
	out := append([]float32(nil), data...)
	return &CpuTensor{Data: out, Shape: []int{len(out)}}, nil
}

// TensorFromF32_2D creates a 2-D tensor.
func (c *Cpu) TensorFromF32_2D(data []float32, rows, cols int) (backend.TensorHandle, error) {
	if len(data) != rows*cols {
		return nil, &cpuError{Message: "data length != rows*cols"}
	}
	out := append([]float32(nil), data...)
	return &CpuTensor{Data: out, Shape: []int{rows, cols}}, nil
}

// TensorToF32 copies the tensor data to `out`. Returns the number copied.
func (c *Cpu) TensorToF32(t backend.TensorHandle, out []float32) (int, error) {
	tc, ok := t.(*CpuTensor)
	if !ok {
		return 0, &cpuError{Message: "non-cpu tensor"}
	}
	n := len(tc.Data)
	if len(out) < n {
		return 0, &cpuError{Message: "output too small"}
	}
	copy(out, tc.Data)
	return n, nil
}

// TensorShape returns the shape.
func (c *Cpu) TensorShape(t backend.TensorHandle) []int {
	tc, ok := t.(*CpuTensor)
	if !ok {
		return nil
	}
	return append([]int(nil), tc.Shape...)
}

// TensorDType returns the dtype (always F32 for the CPU backend).
func (c *Cpu) TensorDType(_ backend.TensorHandle) backend.DType { return backend.DTypeF32 }

// RmsNorm delegates to the tensor package.
func (c *Cpu) RmsNorm(input, weight backend.TensorHandle, eps float32) (backend.TensorHandle, error) {
	in, ok := input.(*CpuTensor)
	if !ok {
		return nil, &cpuError{Message: "input is not a cpu tensor"}
	}
	w, ok := weight.(*CpuTensor)
	if !ok {
		return nil, &cpuError{Message: "weight is not a cpu tensor"}
	}
	out := make([]float32, len(in.Data))
	if err := tensor.RMSNormF32(in.Data, w.Data, out, eps); err != nil {
		return nil, err
	}
	return &CpuTensor{Data: out, Shape: append([]int(nil), in.Shape...)}, nil
}

// ApplyRope delegates to the tensor package.
func (c *Cpu) ApplyRope(input backend.TensorHandle, position, headDim int, theta float32) (backend.TensorHandle, error) {
	tc, ok := input.(*CpuTensor)
	if !ok {
		return nil, &cpuError{Message: "input is not a cpu tensor"}
	}
	out := make([]float32, len(tc.Data))
	if err := tensor.ApplyRopeF32(tc.Data, out, position, headDim, theta); err != nil {
		return nil, err
	}
	return &CpuTensor{Data: out, Shape: append([]int(nil), tc.Shape...)}, nil
}

// AttentionDecode delegates to scaled dot-product attention.
func (c *Cpu) AttentionDecode(query, keyCache, valueCache backend.TensorHandle, seqLen, headDim int, scale float32) (backend.TensorHandle, error) {
	q, k, v := castTriple(query, keyCache, valueCache)
	out := make([]float32, headDim)
	if err := tensor.ScaledDotProductAttentionF32(q.Data, k.Data, v.Data, out, seqLen, headDim, scale); err != nil {
		return nil, err
	}
	return &CpuTensor{Data: out, Shape: []int{headDim}}, nil
}

// Gemv delegates to the tensor package.
func (c *Cpu) Gemv(matrix backend.WeightStorage, vector backend.TensorHandle, rows, cols int) (backend.TensorHandle, error) {
	ws, ok := matrix.(*CpuWeightStorage)
	if !ok {
		return nil, &cpuError{Message: "weight storage must be cpu"}
	}
	v, ok := vector.(*CpuTensor)
	if !ok {
		return nil, &cpuError{Message: "vector must be cpu tensor"}
	}
	weights := make([]float32, rows*cols)
	if ws.Dequant != nil {
		if err := ws.Dequant(ws.Bytes, weights); err != nil {
			return nil, err
		}
	} else {
		if len(ws.Bytes) < rows*cols*4 {
			return nil, &cpuError{Message: "weight storage too small"}
		}
		for i := 0; i < rows*cols; i++ {
			weights[i] = math.Float32frombits(binary.LittleEndian.Uint32(ws.Bytes[i*4:]))
		}
	}
	out := make([]float32, rows)
	if err := tensor.GemvF32(weights, rows, cols, v.Data, out); err != nil {
		return nil, err
	}
	return &CpuTensor{Data: out, Shape: []int{rows}}, nil
}

// Gemm delegates to the tensor package.
func (c *Cpu) Gemm(a, b backend.TensorHandle, rows, sharedDim, cols int) (backend.TensorHandle, error) {
	at, ok := a.(*CpuTensor)
	if !ok {
		return nil, &cpuError{Message: "a must be cpu tensor"}
	}
	bt, ok := b.(*CpuTensor)
	if !ok {
		return nil, &cpuError{Message: "b must be cpu tensor"}
	}
	out := make([]float32, rows*cols)
	if err := tensor.GemmF32(at.Data, bt.Data, rows, sharedDim, cols, out); err != nil {
		return nil, err
	}
	return &CpuTensor{Data: out, Shape: []int{rows, cols}}, nil
}

// Add performs element-wise addition.
func (c *Cpu) Add(a, b backend.TensorHandle) (backend.TensorHandle, error) {
	at, bt := castPair(a, b)
	out := make([]float32, len(at.Data))
	for i := range out {
		if i < len(bt.Data) {
			out[i] = at.Data[i] + bt.Data[i]
		} else {
			out[i] = at.Data[i]
		}
	}
	return &CpuTensor{Data: out, Shape: append([]int(nil), at.Shape...)}, nil
}

// Mul performs element-wise multiplication.
func (c *Cpu) Mul(a, b backend.TensorHandle) (backend.TensorHandle, error) {
	at, bt := castPair(a, b)
	out := make([]float32, len(at.Data))
	for i := range out {
		if i < len(bt.Data) {
			out[i] = at.Data[i] * bt.Data[i]
		} else {
			out[i] = 0
		}
	}
	return &CpuTensor{Data: out, Shape: append([]int(nil), at.Shape...)}, nil
}

// Sigmoid returns sigmoid(x) elementwise.
func (c *Cpu) Sigmoid(x backend.TensorHandle) (backend.TensorHandle, error) {
	tc, ok := x.(*CpuTensor)
	if !ok {
		return nil, &cpuError{Message: "x must be cpu tensor"}
	}
	out := make([]float32, len(tc.Data))
	if err := tensor.Sigmoid(tc.Data, out); err != nil {
		return nil, err
	}
	return &CpuTensor{Data: out, Shape: append([]int(nil), tc.Shape...)}, nil
}

// Softmax delegates to the tensor package.
func (c *Cpu) Softmax(x backend.TensorHandle) (backend.TensorHandle, error) {
	tc, ok := x.(*CpuTensor)
	if !ok {
		return nil, &cpuError{Message: "x must be cpu tensor"}
	}
	out := make([]float32, len(tc.Data))
	if err := tensor.SoftmaxF32(tc.Data, out, len(tc.Data)); err != nil {
		return nil, err
	}
	return &CpuTensor{Data: out, Shape: append([]int(nil), tc.Shape...)}, nil
}

// Synchronize is a no-op for the CPU backend.
func (c *Cpu) Synchronize() error { return nil }

func castPair(a, b backend.TensorHandle) (*CpuTensor, *CpuTensor) {
	at := a.(*CpuTensor)
	bt := b.(*CpuTensor)
	return at, bt
}

func castTriple(a, b, c backend.TensorHandle) (*CpuTensor, *CpuTensor, *CpuTensor) {
	return a.(*CpuTensor), b.(*CpuTensor), c.(*CpuTensor)
}

type cpuError struct{ Message string }

func (e *cpuError) Error() string { return "cpu backend: " + e.Message }
