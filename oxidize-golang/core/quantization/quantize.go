package quantization

import (
	"encoding/binary"
	"math"
)

// F32ToF16Bits converts a float32 to a 16-bit half-precision bit pattern.
func F32ToF16Bits(v float32) uint16 {
	bits := math.Float32bits(v)
	sign := uint16((bits >> 31) & 0x1)
	exp := int32((bits >> 23) & 0xFF) - 127 + 15
	mant := bits & 0x7FFFFF
	switch {
	case exp <= 0:
		if exp < -10 {
			return sign << 15
		}
		mant = (mant | 0x800000) >> uint32(1-exp)
		return (sign << 15) | uint16(mant>>13)
	case exp >= 0x1F:
		return (sign << 15) | 0x7C00
	default:
		return (sign << 15) | (uint16(exp) << 10) | uint16(mant>>13)
	}
}

// clampInt clamps a float32 to the [lo, hi] integer range.
func clampInt(v float32, lo, hi int) int {
	iv := int(v)
	if iv < lo {
		return lo
	}
	if iv > hi {
		return hi
	}
	return iv
}

// QuantizeScalar quantizes `input` (float32) into the target type. For the
// K-quants, IMatrix may be nil; for non-K-quants IMatrix is unused.
func QuantizeScalar(t Type, input []float32, output []byte, imatrix *IMatrix) error {
	switch t {
	case TypeF32:
		return quantizeF32(input, output)
	case TypeF16:
		return quantizeF16(input, output)
	case TypeQ4_0:
		return quantizeQ4_0(input, output)
	case TypeQ4_1:
		return quantizeQ4_1(input, output)
	case TypeQ5_0:
		return quantizeQ5_0(input, output)
	case TypeQ5_1:
		return quantizeQ5_1(input, output)
	case TypeQ8_0:
		return quantizeQ8_0(input, output)
	case TypeQ2_K:
		return quantizeQ2_K(input, output, imatrix)
	case TypeQ3_K_S, TypeQ3_K_M, TypeQ3_K_L:
		return quantizeQ3_K(input, output, imatrix)
	case TypeQ4_K_S, TypeQ4_K_M:
		return quantizeQ4_K(input, output, imatrix)
	case TypeQ5_K_S, TypeQ5_K_M:
		return quantizeQ5_K(input, output, imatrix)
	case TypeQ6_K:
		return quantizeQ6_K(input, output, imatrix)
	default:
		return &Error{Message: "quantize unsupported type " + t.String()}
	}
}

// QuantizeScalarWithIMatrix is a convenience alias for QuantizeScalar that
// requires the caller to provide an IMatrix.
func QuantizeScalarWithIMatrix(t Type, input []float32, output []byte, imatrix *IMatrix) error {
	return QuantizeScalar(t, input, output, imatrix)
}

func quantizeF32(input []float32, output []byte) error {
	if len(output) < len(input)*4 {
		return &Error{Message: "output too small for f32"}
	}
	for i, v := range input {
		binary.LittleEndian.PutUint32(output[i*4:], math.Float32bits(v))
	}
	return nil
}

func quantizeF16(input []float32, output []byte) error {
	if len(output) < len(input)*2 {
		return &Error{Message: "output too small for f16"}
	}
	for i, v := range input {
		binary.LittleEndian.PutUint16(output[i*2:], F32ToF16Bits(v))
	}
	return nil
}

func quantizeQ4_0(input []float32, output []byte) error {
	if len(input)%QK4_0 != 0 {
		return &Error{Message: "input length not aligned to Q4_0 block"}
	}
	blocks := len(input) / QK4_0
	if len(output) < blocks*BLOCK_Q4_0_SIZE {
		return &Error{Message: "output too small for Q4_0"}
	}
	for b := 0; b < blocks; b++ {
		src := input[b*QK4_0 : (b+1)*QK4_0]
		dst := output[b*BLOCK_Q4_0_SIZE:]
		max := MaxAbs(src)
		scale := max / 7
		if scale == 0 {
			scale = 1
		}
		inv := 7 / scale
		binary.LittleEndian.PutUint16(dst[0:2], F32ToF16Bits(scale))
		for i := 0; i < QK4_0/2; i++ {
			v0 := clampInt(src[i*2]*inv+8, 0, 15)
			v1 := clampInt(src[i*2+1]*inv+8, 0, 15)
			dst[2+i] = byte(v0) | byte(v1)<<4
		}
	}
	return nil
}

func quantizeQ4_1(input []float32, output []byte) error {
	if len(input)%QK4_1 != 0 {
		return &Error{Message: "input length not aligned to Q4_1 block"}
	}
	blocks := len(input) / QK4_1
	if len(output) < blocks*BLOCK_Q4_1_SIZE {
		return &Error{Message: "output too small for Q4_1"}
	}
	for b := 0; b < blocks; b++ {
		src := input[b*QK4_1 : (b+1)*QK4_1]
		dst := output[b*BLOCK_Q4_1_SIZE:]
		min, max := MinMax(src)
		scale := (max - min) / 15
		if scale == 0 {
			scale = 1
		}
		inv := 15 / scale
		binary.LittleEndian.PutUint16(dst[0:2], F32ToF16Bits(scale))
		binary.LittleEndian.PutUint16(dst[2:4], F32ToF16Bits(min))
		for i := 0; i < QK4_1/2; i++ {
			v0 := clampInt((src[i*2]-min)*inv, 0, 15)
			v1 := clampInt((src[i*2+1]-min)*inv, 0, 15)
			dst[4+i] = byte(v0) | byte(v1)<<4
		}
	}
	return nil
}

func quantizeQ5_0(input []float32, output []byte) error {
	if len(input)%QK5_0 != 0 {
		return &Error{Message: "input length not aligned to Q5_0 block"}
	}
	blocks := len(input) / QK5_0
	if len(output) < blocks*BLOCK_Q5_0_SIZE {
		return &Error{Message: "output too small for Q5_0"}
	}
	for b := 0; b < blocks; b++ {
		src := input[b*QK5_0 : (b+1)*QK5_0]
		dst := output[b*BLOCK_Q5_0_SIZE:]
		max := MaxAbs(src)
		scale := max / 15
		if scale == 0 {
			scale = 1
		}
		inv := 15 / scale
		binary.LittleEndian.PutUint16(dst[0:2], F32ToF16Bits(scale))
		var qh uint32
		for i := 0; i < QK5_0; i++ {
			v := clampInt(src[i]*inv+16, 0, 31)
			if v >= 16 {
				qh |= 1 << i
			}
			lo := uint8(v & 0x0F)
			pos := 6 + i/2
			if i%2 == 0 {
				dst[pos] = (dst[pos] & 0xF0) | lo
			} else {
				dst[pos] = (dst[pos] & 0x0F) | (lo << 4)
			}
		}
		binary.LittleEndian.PutUint32(dst[2:6], qh)
	}
	return nil
}

func quantizeQ5_1(input []float32, output []byte) error {
	if len(input)%QK5_1 != 0 {
		return &Error{Message: "input length not aligned to Q5_1 block"}
	}
	blocks := len(input) / QK5_1
	if len(output) < blocks*BLOCK_Q5_1_SIZE {
		return &Error{Message: "output too small for Q5_1"}
	}
	for b := 0; b < blocks; b++ {
		src := input[b*QK5_1 : (b+1)*QK5_1]
		dst := output[b*BLOCK_Q5_1_SIZE:]
		min, max := MinMax(src)
		scale := (max - min) / 31
		if scale == 0 {
			scale = 1
		}
		inv := 31 / scale
		binary.LittleEndian.PutUint16(dst[0:2], F32ToF16Bits(scale))
		binary.LittleEndian.PutUint16(dst[2:4], F32ToF16Bits(min))
		var qh uint32
		for i := 0; i < QK5_1; i++ {
			v := clampInt((src[i]-min)*inv, 0, 31)
			if v >= 16 {
				qh |= 1 << i
			}
			lo := uint8(v & 0x0F)
			pos := 8 + i/2
			if i%2 == 0 {
				dst[pos] = (dst[pos] & 0xF0) | lo
			} else {
				dst[pos] = (dst[pos] & 0x0F) | (lo << 4)
			}
		}
		binary.LittleEndian.PutUint32(dst[4:8], qh)
	}
	return nil
}

func quantizeQ8_0(input []float32, output []byte) error {
	if len(input)%QK8_0 != 0 {
		return &Error{Message: "input length not aligned to Q8_0 block"}
	}
	blocks := len(input) / QK8_0
	if len(output) < blocks*BLOCK_Q8_0_SIZE {
		return &Error{Message: "output too small for Q8_0"}
	}
	for b := 0; b < blocks; b++ {
		src := input[b*QK8_0 : (b+1)*QK8_0]
		dst := output[b*BLOCK_Q8_0_SIZE:]
		max := MaxAbs(src)
		scale := max / 127
		if scale == 0 {
			scale = 1
		}
		inv := 127 / scale
		binary.LittleEndian.PutUint16(dst[0:2], F32ToF16Bits(scale))
		for i, v := range src {
			q := int8(clampInt(v*inv, -128, 127))
			dst[2+i] = byte(q)
		}
	}
	return nil
}

func quantizeQ2_K(input []float32, output []byte, imatrix *IMatrix) error {
	if len(input)%QK_K != 0 {
		return &Error{Message: "input length not aligned to Q2_K block"}
	}
	blocks := len(input) / QK_K
	if len(output) < blocks*BLOCK_Q2_K_SIZE {
		return &Error{Message: "output too small for Q2_K"}
	}
	for b := 0; b < blocks; b++ {
		src := input[b*QK_K : (b+1)*QK_K]
		dst := output[b*BLOCK_Q2_K_SIZE:]
		max := MaxAbs(src)
		if max == 0 {
			max = 1
		}
		scale := max / 3
		if scale == 0 {
			scale = 1
		}
		inv := 3 / scale
		binary.LittleEndian.PutUint16(dst[80:82], F32ToF16Bits(scale))
		binary.LittleEndian.PutUint16(dst[82:84], F32ToF16Bits(0))
		for i := 0; i < QK_K/4; i++ {
			var packed uint8
			for j := 0; j < 4; j++ {
				v := uint8(clampInt(src[i*4+j]*inv, 0, 3))
				packed |= v << (j * 2)
			}
			dst[i] = packed
		}
	}
	return nil
}

func quantizeQ3_K(input []float32, output []byte, imatrix *IMatrix) error {
	if len(input)%QK_K != 0 {
		return &Error{Message: "input length not aligned to Q3_K block"}
	}
	blocks := len(input) / QK_K
	if len(output) < blocks*BLOCK_Q3_K_SIZE {
		return &Error{Message: "output too small for Q3_K"}
	}
	for b := 0; b < blocks; b++ {
		src := input[b*QK_K : (b+1)*QK_K]
		dst := output[b*BLOCK_Q3_K_SIZE:]
		max := MaxAbs(src)
		if max == 0 {
			max = 1
		}
		scale := max / 7
		if scale == 0 {
			scale = 1
		}
		inv := 7 / scale
		binary.LittleEndian.PutUint16(dst[108:110], F32ToF16Bits(scale))
		var qh [32]byte
		for i := 0; i < QK_K/2; i++ {
			var packed uint8
			for j := 0; j < 2; j++ {
				v := int(src[i*2+j] * inv)
				if v < 0 {
					v = 0
				}
				if v > 7 {
					v = 7
				}
				packed |= uint8(v) << (j * 4)
			}
			dst[32+i] = packed
		}
		copy(dst[0:32], qh[:])
		for i := 0; i < QK_K/16; i++ {
			dst[96+i] = byte(32)
		}
	}
	return nil
}

func quantizeQ4_K(input []float32, output []byte, imatrix *IMatrix) error {
	if len(input)%QK_K != 0 {
		return &Error{Message: "input length not aligned to Q4_K block"}
	}
	blocks := len(input) / QK_K
	if len(output) < blocks*BLOCK_Q4_K_SIZE {
		return &Error{Message: "output too small for Q4_K"}
	}
	for b := 0; b < blocks; b++ {
		src := input[b*QK_K : (b+1)*QK_K]
		dst := output[b*BLOCK_Q4_K_SIZE:]
		min, max := MinMax(src)
		scale := (max - min) / 63
		if scale == 0 {
			scale = 1
		}
		inv := 63 / scale
		binary.LittleEndian.PutUint16(dst[0:2], F32ToF16Bits(scale))
		binary.LittleEndian.PutUint16(dst[2:4], F32ToF16Bits(min))
		for i := 0; i < QK_K/2; i++ {
			v0 := clampInt((src[i*2]-min)*inv, 0, 63)
			v1 := clampInt((src[i*2+1]-min)*inv, 0, 63)
			dst[16+i] = byte(v0) | byte(v1)<<4
		}
	}
	return nil
}

func quantizeQ5_K(input []float32, output []byte, imatrix *IMatrix) error {
	if len(input)%QK_K != 0 {
		return &Error{Message: "input length not aligned to Q5_K block"}
	}
	blocks := len(input) / QK_K
	if len(output) < blocks*BLOCK_Q5_K_SIZE {
		return &Error{Message: "output too small for Q5_K"}
	}
	for b := 0; b < blocks; b++ {
		src := input[b*QK_K : (b+1)*QK_K]
		dst := output[b*BLOCK_Q5_K_SIZE:]
		min, max := MinMax(src)
		scale := (max - min) / 63
		if scale == 0 {
			scale = 1
		}
		inv := 63 / scale
		binary.LittleEndian.PutUint16(dst[0:2], F32ToF16Bits(scale))
		binary.LittleEndian.PutUint16(dst[2:4], F32ToF16Bits(min))
		for i := 0; i < QK_K; i++ {
			v := clampInt((src[i]-min)*inv, 0, 63)
			if i%2 == 0 {
				dst[48+i/2] = (dst[48+i/2] & 0xF0) | byte(v&0x0F)
			} else {
				dst[48+i/2] = (dst[48+i/2] & 0x0F) | (byte(v&0x0F) << 4)
			}
			if v >= 32 {
				dst[16+i/8] |= 1 << (i % 8)
			}
		}
	}
	return nil
}

func quantizeQ6_K(input []float32, output []byte, imatrix *IMatrix) error {
	if len(input)%QK_K != 0 {
		return &Error{Message: "input length not aligned to Q6_K block"}
	}
	blocks := len(input) / QK_K
	if len(output) < blocks*BLOCK_Q6_K_SIZE {
		return &Error{Message: "output too small for Q6_K"}
	}
	for b := 0; b < blocks; b++ {
		src := input[b*QK_K : (b+1)*QK_K]
		dst := output[b*BLOCK_Q6_K_SIZE:]
		max := MaxAbs(src)
		if max == 0 {
			max = 1
		}
		scale := max / 127
		if scale == 0 {
			scale = 1
		}
		inv := 127 / scale
		binary.LittleEndian.PutUint16(dst[208:210], F32ToF16Bits(scale))
		for i := 0; i < QK_K/16; i++ {
			dst[192+i] = byte(64)
		}
		for i := 0; i < QK_K/2; i++ {
			var packed uint8
			for j := 0; j < 2; j++ {
				v := int(src[i*2+j] * inv)
				if v < -32 {
					v = -32
				}
				if v > 31 {
					v = 31
				}
				if v < 0 {
					v += 64
				}
				packed |= uint8(v) << (j * 4)
			}
			dst[i] = packed
		}
		for i := 0; i < QK_K/4; i++ {
			var packed uint8
			for j := 0; j < 4; j++ {
				v := int(src[i*4+j] * inv)
				if v < -32 {
					v = -32
				}
				if v > 31 {
					v = 31
				}
				if v < 0 {
					v += 64
				}
				packed |= uint8(v&0x3) << (j * 2)
			}
			dst[128+i] = packed
		}
	}
	return nil
}

// DequantizeScalar dequantizes a single buffer from one type to f32.
func DequantizeScalar(t Type, input []byte, output []float32) error {
	return DQuantize(t, input, output)
}
