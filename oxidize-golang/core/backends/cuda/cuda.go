package cudabackend

import (
	"fmt"

	quant "github.com/Zapdev-labs/oxidize/golang/core/quantization"
)

// BuildInfo mirrors CudaBuildInfo.
type BuildInfo struct {
	DetectedAtBuild bool
	CudaPath        string
}

// MemoryDevice mirrors MemoryDevice.
type MemoryDevice uint8

const (
	MemoryCpu MemoryDevice = iota
	MemoryCuda
)

// String returns the canonical name of the memory device.
func (m MemoryDevice) String() string {
	switch m {
	case MemoryCpu:
		return "cpu"
	case MemoryCuda:
		return "cuda"
	default:
		return fmt.Sprintf("memory(%d)", uint8(m))
	}
}

// MemoryError mirrors MemoryError.
type MemoryError struct{ Message string }

func (e *MemoryError) Error() string { return "cuda memory: " + e.Message }

// GemvCudaError mirrors GemvCudaError.
type GemvCudaError struct{ Message string }

func (e *GemvCudaError) Error() string { return "cuda gemv: " + e.Message }

// GemmCudaError mirrors GemmCudaError.
type GemmCudaError struct{ Message string }

func (e *GemmCudaError) Error() string { return "cuda gemm: " + e.Message }

// GemmF32Cuda computes output = left[rows x shared] * right[shared x cols],
// row-major, mirroring gemm.rs:gemm_f32_cuda. Without the cuda build tag this
// runs an optimized host-side GEMM; the cuda tag can override it with cuBLAS.
func GemmF32Cuda(left, right []float32, rows, shared, cols int, output []float32) error {
	if err := ValidateGemmDims(rows, shared, cols); err != nil {
		return err
	}
	if len(left) < rows*shared || len(right) < shared*cols || len(output) < rows*cols {
		return &GemmCudaError{Message: "buffer too small"}
	}
	for r := 0; r < rows; r++ {
		lrow := left[r*shared : r*shared+shared]
		orow := output[r*cols : r*cols+cols]
		for c := range orow {
			orow[c] = 0
		}
		for k := 0; k < shared; k++ {
			a := lrow[k]
			if a == 0 {
				continue
			}
			rrow := right[k*cols : k*cols+cols]
			for c := 0; c < cols; c++ {
				orow[c] += a * rrow[c]
			}
		}
	}
	return nil
}

// GemvQuantizedCuda dequantizes a quantized weight matrix of GGML type ggmlType
// (rows x cols) and computes output[r] = dot(matrix_row_r, vector). Mirrors
// gemv_quantized.rs's on-the-fly quantized GEMV dispatch. The optional scratch
// slice (sized rows*cols) is reused as the dequant target to avoid allocation.
//
// The signature is kept stable: (qbytes, ggmlType, vector, rows, cols, output,
// scratch). Pass nil scratch to allocate internally.
func GemvQuantizedCuda(qbytes []byte, ggmlType int, vector []float32, rows, cols int, output, scratch []float32) error {
	if err := ValidateGemvDims(rows, cols); err != nil {
		return err
	}
	if len(vector) < cols || len(output) < rows {
		return &GemvCudaError{Message: "buffer too small"}
	}
	t := GgmlType(ggmlType)
	if !SupportsQuantizedGpu(t) {
		return &GemvCudaError{Message: fmt.Sprintf("unsupported quant type %d for GPU gemv", ggmlType)}
	}
	dequant := make([]float32, rows*cols)
	if len(scratch) >= rows*cols {
		dequant = scratch[:rows*cols]
	}
	if err := dequantizeMatrix(qbytes, t, dequant); err != nil {
		return err
	}
	for r := 0; r < rows; r++ {
		row := dequant[r*cols : r*cols+cols]
		var sum float32
		for c := 0; c < cols; c++ {
			sum += row[c] * vector[c]
		}
		output[r] = sum
	}
	return nil
}

// ggmlToQuantType maps a GGML numeric type id to the quantization package Type.
func ggmlToQuantType(t GgmlType) (quant.Type, bool) {
	switch t {
	case GgmlTypeF32:
		return quant.TypeF32, true
	case GgmlTypeF16:
		return quant.TypeF16, true
	case GgmlTypeQ4_0:
		return quant.TypeQ4_0, true
	case GgmlTypeQ8_0:
		return quant.TypeQ8_0, true
	case GgmlTypeQ2_K:
		return quant.TypeQ2_K, true
	case GgmlTypeQ4_K:
		return quant.TypeQ4_K_M, true
	case GgmlTypeQ6_K:
		return quant.TypeQ6_K, true
	default:
		return quant.TypeUnknown, false
	}
}

// dequantizeMatrix decodes raw quantized bytes into f32 using the shared
// quantization kernels.
func dequantizeMatrix(qbytes []byte, t GgmlType, out []float32) error {
	qt, ok := ggmlToQuantType(t)
	if !ok {
		return &GemvCudaError{Message: fmt.Sprintf("no dequant for ggml type %d", t)}
	}
	if err := quant.DequantizeScalar(qt, qbytes, out); err != nil {
		return &GemvCudaError{Message: "dequant failed: " + err.Error()}
	}
	return nil
}

// gemvQuantizedInto is the internal GEMV used by the GPU-native forward pass.
// It caches the raw quantized weight resident (uploaded once) and performs the
// dequant-and-dot against vector, writing rows results into out.
func gemvQuantizedInto(s *GpuState, qbytes []byte, t GgmlType, vector []float32, rows, cols int, out []float32) error {
	if len(vector) < cols || len(out) < rows {
		return &GemvCudaError{Message: "buffer too small"}
	}
	key := byteCacheKey(qbytes)
	s.ensureResidentQuant(key, qbytes)
	resident := s.residentQuant[key]
	if resident == nil {
		resident = qbytes
	}
	dequant := make([]float32, rows*cols)
	if err := dequantizeMatrix(resident, t, dequant); err != nil {
		return err
	}
	for r := 0; r < rows; r++ {
		row := dequant[r*cols : r*cols+cols]
		var sum float32
		for c := 0; c < cols; c++ {
			sum += row[c] * vector[c]
		}
		out[r] = sum
	}
	return nil
}

// ValidateGemvDims mirrors validate_gemv_dims.
func ValidateGemvDims(rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return &GemvCudaError{Message: fmt.Sprintf("invalid dims rows=%d cols=%d", rows, cols)}
	}
	return nil
}

// ValidateQ8_0GemvDims mirrors validate_q8_0_gemv_dims.
func ValidateQ8_0GemvDims(rows, cols int) error {
	if cols%32 != 0 {
		return &GemvCudaError{Message: "Q8_0 GEMV requires cols to be a multiple of 32"}
	}
	return ValidateGemvDims(rows, cols)
}

// ValidateGemmDims mirrors validate_gemm_dims.
func ValidateGemmDims(rows, shared, cols int) error {
	if rows <= 0 || shared <= 0 || cols <= 0 {
		return &GemmCudaError{Message: "invalid dims"}
	}
	return nil
}
