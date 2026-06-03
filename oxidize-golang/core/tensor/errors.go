package tensor

import (
	"fmt"
)

// Error types mirroring oxidize_core::compute::tensor::*Error enums. Each
// error type also implements the `error` interface and JSON marshalling is
// intentionally avoided (errors are typically wrapped and printed).

// GemvError is returned from GEMV kernels.
type GemvError struct{ Message string }

func (e *GemvError) Error() string { return "gemv: " + e.Message }

// GemmError is returned from GEMM kernels.
type GemmError struct{ Message string }

func (e *GemmError) Error() string { return "gemm: " + e.Message }

// AttentionError is returned from scaled dot-product attention.
type AttentionError struct{ Message string }

func (e *AttentionError) Error() string { return "attention: " + e.Message }

// RopeError is returned from rotary position embedding kernels.
type RopeError struct{ Message string }

func (e *RopeError) Error() string { return "rope: " + e.Message }

// SwiGluError is returned from the SwiGLU activation kernel.
type SwiGluError struct{ Message string }

func (e *SwiGluError) Error() string { return "swiglu: " + e.Message }

// LinearActivationError is returned from fused linear+activation kernels.
type LinearActivationError struct{ Message string }

func (e *LinearActivationError) Error() string { return "linear: " + e.Message }

// RmsNormError is returned from RMS normalization kernels.
type RmsNormError struct{ Message string }

func (e *RmsNormError) Error() string { return "rms_norm: " + e.Message }

// LayerNormError is returned from layer normalization kernels.
type LayerNormError struct{ Message string }

func (e *LayerNormError) Error() string { return "layer_norm: " + e.Message }

// SoftmaxError is returned from softmax kernels.
type SoftmaxError struct{ Message string }

func (e *SoftmaxError) Error() string { return "softmax: " + e.Message }

// dimCheck returns true if `actual` matches `expected` in length and product.
func dimCheck(name string, expected, actual int) error {
	if actual != expected {
		return fmt.Errorf("%s: dim mismatch: expected %d, got %d", name, expected, actual)
	}
	return nil
}

// lengthCheck returns true if `actual` is at least `expected`.
func lengthCheck(name string, expected, actual int) error {
	if actual < expected {
		return fmt.Errorf("%s: length %d < required %d", name, actual, expected)
	}
	return nil
}
