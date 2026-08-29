package cudabackend

import (
	"fmt"
	"math"
)

// GpuInitActivationBuffers allocates (or reallocates) the activation buffers
func GpuInitActivationBuffers(hiddenSize, intermediateSize int) error {
	if hiddenSize <= 0 || intermediateSize <= 0 {
		return &MemoryError{Message: "activation buffer dims must be positive"}
	}
	return withGPU(func(s *GpuState) error {
		if ab := s.activation; ab != nil &&
			ab.HiddenSize == hiddenSize && ab.IntermediateSize == intermediateSize {
			return nil
		}
		s.activation = &GpuActivationBuffer{
			Hidden:           make([]float32, hiddenSize),
			Normed:           make([]float32, hiddenSize),
			FfnGate:          make([]float32, intermediateSize),
			FfnUp:            make([]float32, intermediateSize),
			FfnDownIn:        make([]float32, intermediateSize),
			HiddenSize:       hiddenSize,
			IntermediateSize: intermediateSize,
		}
		return nil
	})
}

// GpuUploadHidden copies a host hidden-state slice into activation.Hidden.
// Mirrors gpu_kernels.rs:gpu_upload_hidden.
func GpuUploadHidden(hidden []float32) error {
	return withGPU(func(s *GpuState) error {
		ab := s.activation
		if ab == nil {
			return &MemoryError{Message: "activation buffers not initialised"}
		}
		if len(hidden) != ab.HiddenSize {
			return &MemoryError{Message: fmt.Sprintf(
				"gpu_upload_hidden: slice len %d != hidden_size %d", len(hidden), ab.HiddenSize)}
		}
		copy(ab.Hidden, hidden)
		return nil
	})
}

// GpuDownloadHidden copies activation.Hidden back into out. Mirrors
// gpu_native_forward.rs:gpu_download_hidden.
func GpuDownloadHidden(out []float32) error {
	return withGPU(func(s *GpuState) error {
		ab := s.activation
		if ab == nil {
			return &MemoryError{Message: "activation buffers not initialised"}
		}
		if len(out) != ab.HiddenSize {
			return &MemoryError{Message: fmt.Sprintf(
				"gpu_download_hidden: out len %d != hidden_size %d", len(out), ab.HiddenSize)}
		}
		copy(out, ab.Hidden)
		return nil
	})
}

// GpuRmsNorm runs RMS-norm reading activation.Hidden and writing
// activation.Normed. weight is a per-element scale of length hidden_size and is
// cached in residentF32. Mirrors gpu_kernels.rs:gpu_rms_norm.
func GpuRmsNorm(weight []float32, eps float32) error {
	return withGPU(func(s *GpuState) error {
		ab := s.activation
		if ab == nil {
			return &MemoryError{Message: "activation buffers not initialised"}
		}
		if len(weight) != ab.HiddenSize {
			return &MemoryError{Message: fmt.Sprintf(
				"gpu_rms_norm: weight len %d != hidden_size %d", len(weight), ab.HiddenSize)}
		}
		// Cache the norm weight resident, mirroring the Rust pointer-identity pattern.
		key := f32CacheKey(weight)
		if _, ok := s.residentF32[key]; !ok {
			s.residentF32[key] = append([]float32(nil), weight...)
		}
		rmsNormInto(ab.Hidden, weight, eps, ab.Normed)
		return nil
	})
}

// rmsNormInto computes y = (x / rms(x)) * weight, matching the GGML/Llama
// RMS-norm used by the Rust kernel.
func rmsNormInto(x, weight []float32, eps float32, out []float32) {
	n := len(x)
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	scale := float32(1.0 / math.Sqrt(ss/float64(n)+float64(eps)))
	for i := 0; i < n; i++ {
		out[i] = x[i] * scale * weight[i]
	}
}

// GpuAttnRmsAndQkvQ4K runs the fused attention pre-projection on-device: it
func GpuAttnRmsAndQkvQ4K(
	attnNorm []float32, eps float32,
	wq []byte, qLen int,
	wk []byte, kvLen int,
	wv []byte,
	qType GgmlType, hiddenSize int,
	qOut, kOut, vOut []float32,
) error {
	if _, ok := dequantKernelFor(qType); !ok {
		return &GemvCudaError{Message: fmt.Sprintf("unsupported quant type %d for fused qkv", qType)}
	}
	if len(qOut) < qLen || len(kOut) < kvLen || len(vOut) < kvLen {
		return &GemvCudaError{Message: "qkv output buffers too small"}
	}
	return withGPU(func(s *GpuState) error {
		ab := s.activation
		if ab == nil {
			return &MemoryError{Message: "activation buffers not initialised"}
		}
		if len(attnNorm) != hiddenSize || ab.HiddenSize != hiddenSize {
			return &MemoryError{Message: fmt.Sprintf(
				"gpu_attn_rms_and_qkv: hidden_size mismatch (norm=%d hidden=%d arg=%d)",
				len(attnNorm), ab.HiddenSize, hiddenSize)}
		}
		// RMS-norm hidden -> normed.
		rmsNormInto(ab.Hidden, attnNorm, eps, ab.Normed)
		// Q/K/V projections against the normed hidden state.
		if err := gemvQuantizedInto(s, wq, qType, ab.Normed, qLen, hiddenSize, qOut); err != nil {
			return err
		}
		if err := gemvQuantizedInto(s, wk, qType, ab.Normed, kvLen, hiddenSize, kOut); err != nil {
			return err
		}
		return gemvQuantizedInto(s, wv, qType, ab.Normed, kvLen, hiddenSize, vOut)
	})
}
