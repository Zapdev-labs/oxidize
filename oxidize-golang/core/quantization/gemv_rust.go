// +build cgo,rust_ffi

package quantization

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR}/../../../target/release -loxidize_ffi -lpthread -ldl -lm
#include <stdint.h>

extern int oxidize_gemv_quantized(
	uint32_t quant_type,
	const uint8_t* qbytes, size_t qbytes_len,
	size_t rows, size_t cols,
	const float* vector,
	float* output
);
*/
import "C"
import "unsafe"

// rustQuantType maps our Go quantization.Type to the integer expected by
// oxidize_gemv_quantized in oxidize-ffi/src/lib.rs.
func rustQuantType(t Type) (C.uint32_t, bool) {
	switch t {
	case TypeF32:
		return 0, true
	case TypeF16:
		return 1, true
	case TypeQ4_0:
		return 2, true
	case TypeQ4_1:
		return 3, true
	case TypeQ8_0:
		return 6, true
	case TypeQ2_K:
		return 7, true
	case TypeQ3_K_S, TypeQ3_K_M, TypeQ3_K_L:
		return 8, true // all Q3_K variants
	case TypeQ4_K_S:
		return 11, true
	case TypeQ4_K_M:
		return 12, true
	case TypeQ5_K_S:
		return 13, true
	case TypeQ5_K_M:
		return 14, true
	case TypeQ6_K:
		return 15, true
	default:
		return 0, false
	}
}

// GemvRust calls the Rust AVX2+FMA optimized GEMV kernel via CGo.
// It handles all quantization types supported by oxidize-core's
// gemv_quantized_f32 (Q4_K, Q8_0, Q6_K, etc.).
// Returns false if the type is not supported and the caller should fall back.
func GemvRust(qbytes []byte, qtype Type, rows, cols int, vector, output []float32) (bool, error) {
	rt, ok := rustQuantType(qtype)
	if !ok {
		return false, nil
	}
	if rows <= 0 || cols <= 0 {
		return true, &Error{Message: "non-positive dimension for GemvRust"}
	}
	if len(qbytes) == 0 || len(vector) < cols || len(output) < rows {
		return true, &Error{Message: "buffer too small for GemvRust"}
	}
	rc := C.oxidize_gemv_quantized(
		rt,
		(*C.uint8_t)(unsafe.Pointer(&qbytes[0])),
		C.size_t(len(qbytes)),
		C.size_t(rows),
		C.size_t(cols),
		(*C.float)(unsafe.Pointer(&vector[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
	)
	if rc != 0 {
		return true, &Error{Message: "oxidize_gemv_quantized returned error"}
	}
	return true, nil
}
