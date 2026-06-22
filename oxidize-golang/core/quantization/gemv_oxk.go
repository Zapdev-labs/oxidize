//go:build cgo && oxk

package quantization

/*
#cgo CFLAGS: -I${SRCDIR}/../../../liboxk/include
#cgo LDFLAGS: -L${SRCDIR}/../../../dist/liboxk -L${SRCDIR}/../../../liboxk -loxk -lm -Wl,-rpath,${SRCDIR}/../../../dist/liboxk -Wl,-rpath,${SRCDIR}/../../../liboxk
#include <stdint.h>
#include "oxk.h"

static int oxk_gemv_quantized_bridge(
	uint32_t quant_type,
	const uint8_t* qbytes, size_t qbytes_len,
	size_t rows, size_t cols,
	const float* vector,
	float* output
) {
	return oxk_gemv_quantized(quant_type, qbytes, qbytes_len, rows, cols, vector, output);
}
*/
import "C"
import (
	"sync"
	"unsafe"
)

var oxkInitOnce sync.Once

func ensureOxkInit() {
	oxkInitOnce.Do(func() {
		C.oxk_init()
	})
}

// oxkQuantType maps Go quantization.Type to oxidize_gemv_quantized integer codes.
func oxkQuantType(t Type) (C.uint32_t, bool) {
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
		return 8, true
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

// GemvOxk calls the C++ OXK kernel library for fused quantized GEMV.
// Returns false if the type is unsupported (caller should fall back).
func GemvOxk(qbytes []byte, qtype Type, rows, cols int, vector, output []float32) (bool, error) {
	qt, ok := oxkQuantType(qtype)
	if !ok {
		return false, nil
	}
	if rows <= 0 || cols <= 0 {
		return true, &Error{Message: "non-positive dimension for GemvOxk"}
	}
	if len(qbytes) == 0 || len(vector) < cols || len(output) < rows {
		return true, &Error{Message: "buffer too small for GemvOxk"}
	}
	ensureOxkInit()
	rc := C.oxk_gemv_quantized_bridge(
		qt,
		(*C.uint8_t)(unsafe.Pointer(&qbytes[0])),
		C.size_t(len(qbytes)),
		C.size_t(rows),
		C.size_t(cols),
		(*C.float)(unsafe.Pointer(&vector[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
	)
	if rc != 0 {
		return false, nil
	}
	return true, nil
}

// OxkHasAVX2 reports whether AVX2 kernels are available.
func OxkHasAVX2() bool {
	ensureOxkInit()
	return C.oxk_has_avx2() != 0
}

// OxkDotF32 computes a dot product using OXK SIMD kernels when available.
func OxkDotF32(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	ensureOxkInit()
	return float32(C.oxk_dot_f32(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.size_t(n),
	))
}
