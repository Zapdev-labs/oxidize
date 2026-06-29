package cudabackend

import (
	"github.com/Zapdev-labs/oxidize/golang/core/backend"
	cpubackend "github.com/Zapdev-labs/oxidize/golang/core/backends/cpu"
)

// Cuda implements ComputeBackend with CUDA GEMV when native code is linked,
// otherwise delegating tensor ops to the CPU backend while reporting name cuda.
type Cuda struct {
	cpu *cpubackend.Cpu
}

// New constructs a CUDA backend wrapper.
func New() *Cuda { return &Cuda{cpu: cpubackend.New()} }

// Name returns the backend identifier.
func (c *Cuda) Name() string { return "cuda" }

func (c *Cuda) TensorFromF32(data []float32) (backend.TensorHandle, error) {
	return c.cpu.TensorFromF32(data)
}

func (c *Cuda) TensorFromF32_2D(data []float32, rows, cols int) (backend.TensorHandle, error) {
	return c.cpu.TensorFromF32_2D(data, rows, cols)
}

func (c *Cuda) TensorToF32(tensor backend.TensorHandle, out []float32) (int, error) {
	return c.cpu.TensorToF32(tensor, out)
}

func (c *Cuda) TensorShape(tensor backend.TensorHandle) []int { return c.cpu.TensorShape(tensor) }

func (c *Cuda) TensorDType(tensor backend.TensorHandle) backend.DType {
	return c.cpu.TensorDType(tensor)
}

func (c *Cuda) RmsNorm(input, weight backend.TensorHandle, eps float32) (backend.TensorHandle, error) {
	return c.cpu.RmsNorm(input, weight, eps)
}

func (c *Cuda) ApplyRope(input backend.TensorHandle, position, headDim int, theta float32) (backend.TensorHandle, error) {
	return c.cpu.ApplyRope(input, position, headDim, theta)
}

func (c *Cuda) AttentionDecode(query, keyCache, valueCache backend.TensorHandle, seqLen, headDim int, scale float32) (backend.TensorHandle, error) {
	return c.cpu.AttentionDecode(query, keyCache, valueCache, seqLen, headDim, scale)
}

func (c *Cuda) Gemv(matrix backend.WeightStorage, vector backend.TensorHandle, rows, cols int) (backend.TensorHandle, error) {
	if ws, ok := matrix.(*cpubackend.CpuWeightStorage); ok {
		if vec, ok := vector.(*cpubackend.CpuTensor); ok {
			mat := make([]float32, rows*cols)
			out := make([]float32, rows)
			if ws.Dequant != nil {
				if err := ws.Dequant(ws.Bytes, mat); err == nil {
					if err := gemvF32Native(mat, vec.Data, rows, cols, out); err == nil {
						return c.cpu.TensorFromF32(out)
					}
				}
			}
		}
	}
	return c.cpu.Gemv(matrix, vector, rows, cols)
}

func (c *Cuda) Gemm(a, b backend.TensorHandle, rows, sharedDim, cols int) (backend.TensorHandle, error) {
	if at, ok := a.(*cpubackend.CpuTensor); ok {
		if bt, ok := b.(*cpubackend.CpuTensor); ok &&
			len(at.Data) >= rows*sharedDim && len(bt.Data) >= sharedDim*cols {
			out := make([]float32, rows*cols)
			if err := GemmF32Cuda(at.Data, bt.Data, rows, sharedDim, cols, out); err == nil {
				return c.cpu.TensorFromF32_2D(out, rows, cols)
			}
		}
	}
	return c.cpu.Gemm(a, b, rows, sharedDim, cols)
}

func (c *Cuda) Add(a, b backend.TensorHandle) (backend.TensorHandle, error) { return c.cpu.Add(a, b) }

func (c *Cuda) Mul(a, b backend.TensorHandle) (backend.TensorHandle, error) { return c.cpu.Mul(a, b) }

func (c *Cuda) Sigmoid(x backend.TensorHandle) (backend.TensorHandle, error) { return c.cpu.Sigmoid(x) }

func (c *Cuda) Softmax(x backend.TensorHandle) (backend.TensorHandle, error) { return c.cpu.Softmax(x) }

func (c *Cuda) Synchronize() error { return nil }

func gemvF32Native(matrix, vector []float32, rows, cols int, out []float32) error {
	if err := GemvF32Cuda(matrix, vector, rows, cols, out); err == nil {
		return nil
	}
	for r := 0; r < rows; r++ {
		var sum float32
		row := matrix[r*cols : (r+1)*cols]
		for c := 0; c < cols && c < len(vector); c++ {
			sum += row[c] * vector[c]
		}
		out[r] = sum
	}
	return nil
}
