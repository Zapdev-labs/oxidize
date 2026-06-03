// Package quantization mirrors oxidize_core::compute::quantization. It
// enumerates the supported GGML quantization types and provides per-format
// dequantize kernels, size lookups, scalar quantize kernels, importance
// matrix support, and mixed-precision planning.
package quantization

import (
	"fmt"
	"math"
)

// Type identifies a GGUF / GGML quantization scheme. It mirrors
// GgufQuantizationType from oxidize-core/src/format/gguf.rs.
type Type uint32

const (
	TypeF32 Type = iota
	TypeF16
	TypeQ4_0
	TypeQ4_1
	TypeQ5_0
	TypeQ5_1
	TypeQ8_0
	TypeQ2_K
	TypeQ3_K_S
	TypeQ3_K_M
	TypeQ3_K_L
	TypeQ4_K_S
	TypeQ4_K_M
	TypeQ5_K_S
	TypeQ5_K_M
	TypeQ6_K
	TypeIQ2_XXS
	TypeIQ2_XS
	TypeIQ3_XXS
	TypeIQ1_S
	TypeIQ4_NL
	TypeIQ3_S
	TypeIQ2_S
	TypeIQ4_XS
	TypeIQ1_M
	TypeNVFP4
	TypeQ8_K
	TypeUnknown // sentinel
)

// Block sizes re-exported as constants.
const (
	QK8_0          = 32
	QK4_0          = 32
	QK4_1          = 32
	QK5_0          = 32
	QK5_1          = 32
	QK_K           = 256
	QK_NVFP4       = 64
	QK_NVFP4_SUB   = 16
	BLOCK_Q4_0_SIZE  = 18
	BLOCK_Q4_1_SIZE  = 20
	BLOCK_Q5_0_SIZE  = 22
	BLOCK_Q5_1_SIZE  = 24
	BLOCK_Q8_0_SIZE  = 34
	BLOCK_Q2_K_SIZE  = 84
	BLOCK_Q3_K_SIZE  = 110
	BLOCK_Q4_K_SIZE  = 144
	BLOCK_Q5_K_SIZE  = 176
	BLOCK_Q6_K_SIZE  = 210
	BLOCK_Q8_K_SIZE  = 292
	BLOCK_NVFP4_SIZE = 34
	BLOCK_IQ1_S_SIZE = 50
	BLOCK_IQ1_M_SIZE = 56
	BLOCK_IQ2_XXS_SIZE = 66
	BLOCK_IQ2_XS_SIZE  = 74
	BLOCK_IQ2_S_SIZE   = 82
	BLOCK_IQ3_XXS_SIZE = 98
	BLOCK_IQ3_S_SIZE   = 110
	BLOCK_IQ4_NL_SIZE  = 18
	BLOCK_IQ4_XS_SIZE  = 34
)

// E2M1DoubledValues is the FP4 (NVFP4 sub-block) lookup table mirroring
// E2M1_DOUBLED_VALUES in oxidize-core/src/compute/quantization.rs.
var E2M1DoubledValues = [16]float32{
	0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6,
}

// Error mirrors QuantizationError.
type Error struct{ Message string }

func (e *Error) Error() string { return "quantization: " + e.Message }

// String returns the canonical name of the type (matching llama.cpp/GGUF).
func (t Type) String() string {
	switch t {
	case TypeF32:
		return "F32"
	case TypeF16:
		return "F16"
	case TypeQ4_0:
		return "Q4_0"
	case TypeQ4_1:
		return "Q4_1"
	case TypeQ5_0:
		return "Q5_0"
	case TypeQ5_1:
		return "Q5_1"
	case TypeQ8_0:
		return "Q8_0"
	case TypeQ2_K:
		return "Q2_K"
	case TypeQ3_K_S:
		return "Q3_K_S"
	case TypeQ3_K_M:
		return "Q3_K_M"
	case TypeQ3_K_L:
		return "Q3_K_L"
	case TypeQ4_K_S:
		return "Q4_K_S"
	case TypeQ4_K_M:
		return "Q4_K_M"
	case TypeQ5_K_S:
		return "Q5_K_S"
	case TypeQ5_K_M:
		return "Q5_K_M"
	case TypeQ6_K:
		return "Q6_K"
	case TypeIQ2_XXS:
		return "IQ2_XXS"
	case TypeIQ2_XS:
		return "IQ2_XS"
	case TypeIQ3_XXS:
		return "IQ3_XXS"
	case TypeIQ1_S:
		return "IQ1_S"
	case TypeIQ4_NL:
		return "IQ4_NL"
	case TypeIQ3_S:
		return "IQ3_S"
	case TypeIQ2_S:
		return "IQ2_S"
	case TypeIQ4_XS:
		return "IQ4_XS"
	case TypeIQ1_M:
		return "IQ1_M"
	case TypeNVFP4:
		return "NVFP4"
	case TypeQ8_K:
		return "Q8_K"
	default:
		return fmt.Sprintf("Unknown(%d)", uint32(t))
	}
}

// BlockSize returns the number of source values that fit in a single block.
func (t Type) BlockSize() int {
	switch t {
	case TypeF32, TypeF16:
		return 1
	case TypeQ4_0, TypeQ4_1, TypeQ5_0, TypeQ5_1, TypeQ8_0:
		return QK4_0
	case TypeQ2_K, TypeQ3_K_S, TypeQ3_K_M, TypeQ3_K_L,
		TypeQ4_K_S, TypeQ4_K_M, TypeQ5_K_S, TypeQ5_K_M,
		TypeQ6_K, TypeQ8_K:
		return QK_K
	case TypeNVFP4:
		return QK_NVFP4
	case TypeIQ1_S, TypeIQ1_M, TypeIQ2_XXS, TypeIQ2_XS, TypeIQ2_S,
		TypeIQ3_XXS, TypeIQ3_S, TypeIQ4_NL, TypeIQ4_XS:
		return QK_K
	default:
		return 0
	}
}

// BytesPerBlock returns the number of bytes per quantized block.
func (t Type) BytesPerBlock() int {
	switch t {
	case TypeF32:
		return 4
	case TypeF16:
		return 2
	case TypeQ4_0:
		return BLOCK_Q4_0_SIZE
	case TypeQ4_1:
		return BLOCK_Q4_1_SIZE
	case TypeQ5_0:
		return BLOCK_Q5_0_SIZE
	case TypeQ5_1:
		return BLOCK_Q5_1_SIZE
	case TypeQ8_0:
		return BLOCK_Q8_0_SIZE
	case TypeQ2_K:
		return BLOCK_Q2_K_SIZE
	case TypeQ3_K_S, TypeQ3_K_M, TypeQ3_K_L:
		return BLOCK_Q3_K_SIZE
	case TypeQ4_K_S, TypeQ4_K_M:
		return BLOCK_Q4_K_SIZE
	case TypeQ5_K_S, TypeQ5_K_M:
		return BLOCK_Q5_K_SIZE
	case TypeQ6_K:
		return BLOCK_Q6_K_SIZE
	case TypeQ8_K:
		return BLOCK_Q8_K_SIZE
	case TypeNVFP4:
		return BLOCK_NVFP4_SIZE
	case TypeIQ1_S:
		return BLOCK_IQ1_S_SIZE
	case TypeIQ1_M:
		return BLOCK_IQ1_M_SIZE
	case TypeIQ2_XXS:
		return BLOCK_IQ2_XXS_SIZE
	case TypeIQ2_XS:
		return BLOCK_IQ2_XS_SIZE
	case TypeIQ2_S:
		return BLOCK_IQ2_S_SIZE
	case TypeIQ3_XXS:
		return BLOCK_IQ3_XXS_SIZE
	case TypeIQ3_S:
		return BLOCK_IQ3_S_SIZE
	case TypeIQ4_NL:
		return BLOCK_IQ4_NL_SIZE
	case TypeIQ4_XS:
		return BLOCK_IQ4_XS_SIZE
	default:
		return 0
	}
}

// QuantizedSize computes the byte size of a quantized buffer for a given
// number of values. Mirrors `quantized_size` in quantization.rs.
func QuantizedSize(t Type, n int) (int, error) {
	if n < 0 {
		return 0, &Error{Message: "negative value count"}
	}
	if t == TypeF32 {
		return n * 4, nil
	}
	if t == TypeF16 {
		return n * 2, nil
	}
	block := t.BlockSize()
	if block == 0 {
		return 0, &Error{Message: fmt.Sprintf("unsupported type %s", t)}
	}
	bytes := t.BytesPerBlock()
	blocks := (n + block - 1) / block
	return blocks * bytes, nil
}

// ParseType parses a quantization type by name (matching the Rust enum).
func ParseType(name string) (Type, error) {
	switch name {
	case "F32", "f32":
		return TypeF32, nil
	case "F16", "f16":
		return TypeF16, nil
	case "Q4_0":
		return TypeQ4_0, nil
	case "Q4_1":
		return TypeQ4_1, nil
	case "Q5_0":
		return TypeQ5_0, nil
	case "Q5_1":
		return TypeQ5_1, nil
	case "Q8_0":
		return TypeQ8_0, nil
	case "Q2_K":
		return TypeQ2_K, nil
	case "Q3_K_S":
		return TypeQ3_K_S, nil
	case "Q3_K_M":
		return TypeQ3_K_M, nil
	case "Q3_K_L":
		return TypeQ3_K_L, nil
	case "Q4_K_S":
		return TypeQ4_K_S, nil
	case "Q4_K_M":
		return TypeQ4_K_M, nil
	case "Q5_K_S":
		return TypeQ5_K_S, nil
	case "Q5_K_M":
		return TypeQ5_K_M, nil
	case "Q6_K":
		return TypeQ6_K, nil
	case "IQ2_XXS":
		return TypeIQ2_XXS, nil
	case "IQ2_XS":
		return TypeIQ2_XS, nil
	case "IQ3_XXS":
		return TypeIQ3_XXS, nil
	case "IQ1_S":
		return TypeIQ1_S, nil
	case "IQ4_NL":
		return TypeIQ4_NL, nil
	case "IQ3_S":
		return TypeIQ3_S, nil
	case "IQ2_S":
		return TypeIQ2_S, nil
	case "IQ4_XS":
		return TypeIQ4_XS, nil
	case "IQ1_M":
		return TypeIQ1_M, nil
	case "NVFP4":
		return TypeNVFP4, nil
	case "Q8_K":
		return TypeQ8_K, nil
	default:
		return TypeUnknown, &Error{Message: fmt.Sprintf("unknown type %q", name)}
	}
}

// FromLLamaFType mirrors GgufQuantizationType::from_llama_ftype.
func FromLLamaFType(ftype uint32) Type {
	switch ftype {
	case 0:
		return TypeF32
	case 1:
		return TypeF16
	case 2:
		return TypeQ4_0
	case 3:
		return TypeQ4_1
	case 6:
		return TypeQ5_0
	case 7:
		return TypeQ5_1
	case 8:
		return TypeQ8_0
	default:
		return TypeUnknown
	}
}

// FromGGMLType maps the raw GGML type id (used in GGUF tensor infos) to a
// quantization Type.
func FromGGMLType(id uint32) Type {
	switch id {
	case 0:
		return TypeF32
	case 1:
		return TypeF16
	case 2:
		return TypeQ4_0
	case 3:
		return TypeQ4_1
	case 6:
		return TypeQ5_0
	case 7:
		return TypeQ5_1
	case 8:
		return TypeQ8_0
	case 10:
		return TypeQ2_K
	case 11:
		return TypeQ3_K_S
	case 12:
		return TypeQ4_K_S
	case 13:
		return TypeQ5_K_S
	case 14:
		return TypeQ6_K
	case 15:
		return TypeIQ2_XXS
	case 16:
		return TypeIQ2_XS
	case 17:
		return TypeIQ3_XXS
	case 18:
		return TypeIQ1_S
	case 19:
		return TypeIQ4_NL
	case 20:
		return TypeIQ3_S
	case 21:
		return TypeIQ2_S
	case 22:
		return TypeIQ4_XS
	case 23:
		return TypeIQ1_M
	case 24:
		return TypeNVFP4
	default:
		return TypeUnknown
	}
}

// MaxAbs returns the maximum absolute value of a buffer.
func MaxAbs(values []float32) float32 {
	var max float32
	for _, v := range values {
		a := float32(math.Abs(float64(v)))
		if a > max {
			max = a
		}
	}
	return max
}

// MinMax returns the (min, max) of a buffer.
func MinMax(values []float32) (float32, float32) {
	if len(values) == 0 {
		return 0, 0
	}
	min, max := values[0], values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max
}
