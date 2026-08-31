package flash_attention

import "math"

// f16BitsToF32 converts IEEE-754 half-precision bits (as stored in a KV cache)
// to float32. Mirrors the f16c hardware conversion used by the Rust AVX2 path.
func f16BitsToF32(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exp := uint32(bits>>10) & 0x1F
	mant := uint32(bits & 0x03FF)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: normalize. Shift until the implicit 1 bit appears.
		shift := 0
		for mant&0x0400 == 0 {
			mant <<= 1
			shift++
		}
		mant &= 0x03FF
		// f32 exponent = (1 - 15) - shift + 127 = 113 - shift.
		expF := uint32(113-shift) << 23
		return math.Float32frombits(sign | expF | (mant << 13))
	case 0x1F:
		// Inf / NaN.
		return math.Float32frombits(sign | 0x7F800000 | (mant << 13))
	default:
		expF := (exp + (127 - 15)) << 23
		return math.Float32frombits(sign | expF | (mant << 13))
	}
}

// DotProductF32F16 computes the dot product of an f32 query and an f16 key
func DotProductF32F16(a []float32, b []uint16) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s0, s1 float32
	i := 0
	for ; i+2 <= n; i += 2 {
		s0 += a[i] * f16BitsToF32(b[i])
		s1 += a[i+1] * f16BitsToF32(b[i+1])
	}
	for ; i < n; i++ {
		s0 += a[i] * f16BitsToF32(b[i])
	}
	return s0 + s1
}

// AxpyF32 computes out[i] += scale * row[i] in a single fused pass. Mirrors
// axpy_f32_avx2.
func AxpyF32(out []float32, scale float32, row []float32) {
	n := len(out)
	if len(row) < n {
		n = len(row)
	}
	i := 0
	for ; i+8 <= n; i += 8 {
		out[i] += scale * row[i]
		out[i+1] += scale * row[i+1]
		out[i+2] += scale * row[i+2]
		out[i+3] += scale * row[i+3]
		out[i+4] += scale * row[i+4]
		out[i+5] += scale * row[i+5]
		out[i+6] += scale * row[i+6]
		out[i+7] += scale * row[i+7]
	}
	for ; i < n; i++ {
		out[i] += scale * row[i]
	}
}

// AxpyF32F16 computes out[i] += scale * f16_to_f32(row[i]) in a single fused
// pass over an f16 value row. Mirrors axpy_f32_f16_avx2.
func AxpyF32F16(out []float32, scale float32, row []uint16) {
	n := len(out)
	if len(row) < n {
		n = len(row)
	}
	for i := 0; i < n; i++ {
		out[i] += scale * f16BitsToF32(row[i])
	}
}
