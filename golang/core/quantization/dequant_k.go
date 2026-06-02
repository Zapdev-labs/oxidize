package quantization

import (
	"encoding/binary"
	"math"
)

// DequantQ2_K dequantizes Q2_K super-blocks (256 values, 84 bytes).
// Layout: scales[16] u8 (4-bit pairs), qs[64] (2-bit packed), d/m f16, dmin f16.
func DequantQ2_K(input []byte, output []float32) error {
	if len(input)%BLOCK_Q2_K_SIZE != 0 {
		return &Error{Message: "q2_k input not aligned"}
	}
	blocks := len(input) / BLOCK_Q2_K_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_Q2_K_SIZE:]
		qs := blk[0:64]
		scales := blk[64:80]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[80:82]))
		dmin := f16BitsToF32(binary.LittleEndian.Uint16(blk[82:84]))
		out := output[b*QK_K:]
		var sc, mn [QK_K/16]float32
		for j := 0; j < QK_K/16; j++ {
			if j < 4 {
				sc[j] = float32(scales[j]&0x0F) * d
				mn[j] = float32(scales[j]>>4) * dmin
			} else {
				sc[j] = float32(scales[j+4]&0x0F) * d
				mn[j] = float32(scales[j+4]>>4) * dmin
			}
		}
		for n := 0; n < QK_K; n++ {
			idx := n / 32
			shift := (n % 32) * 2
			q := (qs[idx] >> shift) & 0x3
			l := n / 16
			out[n] = float32(q)*sc[l] - mn[l]
		}
	}
	return nil
}

// DequantQ3_K dequantizes Q3_K super-blocks (256 values, 110 bytes).
// Layout: hmask[32] (1-bit per element), qs[64] (4-bit nibbles), scales[16] i8,
// d f16 (subtracted from each scale before use).
func DequantQ3_K(input []byte, output []float32) error {
	if len(input)%BLOCK_Q3_K_SIZE != 0 {
		return &Error{Message: "q3_k input not aligned"}
	}
	blocks := len(input) / BLOCK_Q3_K_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_Q3_K_SIZE:]
		hmask := blk[0:32]
		qs := blk[32:96]
		scales := blk[96:112]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[108:110]))
		out := output[b*QK_K:]
		var sc [QK_K/16]float32
		for j := 0; j < QK_K/16; j++ {
			sc[j] = (float32(int8(scales[j])) - 32) * d
		}
		for n := 0; n < QK_K; n++ {
			idx := n / 32
			shift := (n % 32) * 2
			q := (qs[idx] >> shift) & 0x3
			hbit := (hmask[n/8] >> (n % 8)) & 1
			val := float32(int(q) - int(hbit)*4) // sign flip in 3-bit signed
			l := n / 16
			out[n] = val * sc[l]
		}
	}
	return nil
}

// DequantQ4_K dequantizes Q4_K super-blocks (256 values, 144 bytes).
func DequantQ4_K(input []byte, output []float32) error {
	if len(input)%BLOCK_Q4_K_SIZE != 0 {
		return &Error{Message: "q4_k input not aligned"}
	}
	blocks := len(input) / BLOCK_Q4_K_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_Q4_K_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := f16BitsToF32(binary.LittleEndian.Uint16(blk[2:4]))
		scales := blk[4:16]
		qs := blk[16 : 16+128]
		out := output[b*QK_K:]
		var sc, mn [QK_K/32]float32
		for j := 0; j < QK_K/32; j++ {
			sc[j] = float32(scales[j]&0x3F) * d
			mn[j] = float32(scales[j]>>6) * dmin
		}
		for n := 0; n < QK_K; n++ {
			idx := n / 32
			shift := (n % 32) * 4
			q := (qs[idx] >> shift) & 0xF
			l := n / 32
			out[n] = float32(q)*sc[l] - mn[l]
		}
	}
	return nil
}

// DequantQ5_K dequantizes Q5_K super-blocks (256 values, 176 bytes).
func DequantQ5_K(input []byte, output []float32) error {
	if len(input)%BLOCK_Q5_K_SIZE != 0 {
		return &Error{Message: "q5_k input not aligned"}
	}
	blocks := len(input) / BLOCK_Q5_K_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_Q5_K_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := f16BitsToF32(binary.LittleEndian.Uint16(blk[2:4]))
		scales := blk[4:20]
		qh := blk[20:52]
		qs := blk[52 : 52+128]
		out := output[b*QK_K:]
		var sc, mn [QK_K/32]float32
		for j := 0; j < QK_K/32; j++ {
			sc[j] = float32(scales[j]&0x1F) * d
			mn[j] = float32(scales[j]>>5) * dmin
		}
		for n := 0; n < QK_K; n++ {
			idx := n / 32
			shift := (n % 32) * 4
			q := (qs[idx] >> shift) & 0xF
			hb := (qh[n/8] >> (n % 8)) & 1
			val := int(q) | int(hb)<<4
			l := n / 32
			out[n] = float32(val)*sc[l] - mn[l]
		}
	}
	return nil
}

// DequantQ6_K dequantizes Q6_K super-blocks (256 values, 210 bytes).
func DequantQ6_K(input []byte, output []float32) error {
	if len(input)%BLOCK_Q6_K_SIZE != 0 {
		return &Error{Message: "q6_k input not aligned"}
	}
	blocks := len(input) / BLOCK_Q6_K_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_Q6_K_SIZE:]
		ql := blk[0:128]
		qh := blk[128:192]
		scales := blk[192:208]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[208:210]))
		out := output[b*QK_K:]
		for n := 0; n < QK_K; n++ {
			idx := n / 32
			shift := (n % 32) * 2
			qlo := (ql[idx] >> shift) & 0x3
			qhi := (qh[idx] >> shift) & 0x3
			q := int8((qhi<<2)|qlo) - 32
			l := n / 16
			out[n] = float32(q) * float32(int8(scales[l])) * d
		}
	}
	return nil
}

// DequantNVFP4 dequantizes NVFP4 blocks (64 values, 34 bytes).
func DequantNVFP4(input []byte, output []float32) error {
	if len(input)%BLOCK_NVFP4_SIZE != 0 {
		return &Error{Message: "nvfp4 input not aligned"}
	}
	blocks := len(input) / BLOCK_NVFP4_SIZE
	if len(output) < blocks*QK_NVFP4 {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_NVFP4_SIZE:]
		scale := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		// Decode each FP4 nibble to a half-doubled value, then divide by 2
		// (since E2M1 doubled values represent 2*x, where x is the actual
		// dequantized value).
		for i := 0; i < QK_NVFP4; i++ {
			qs := blk[2+i/2]
			var nib uint8
			if i%2 == 0 {
				nib = qs & 0x0F
			} else {
				nib = (qs >> 4) & 0x0F
			}
			v := E2M1DoubledValues[nib] * 0.5
			output[b*QK_NVFP4+i] = v * scale
		}
	}
	return nil
}

// DequantIQ1_S dequantizes IQ1_S blocks (256 values, 50 bytes).
func DequantIQ1_S(input []byte, output []float32) error {
	if len(input)%BLOCK_IQ1_S_SIZE != 0 {
		return &Error{Message: "iq1_s input not aligned"}
	}
	blocks := len(input) / BLOCK_IQ1_S_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_IQ1_S_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		for i := 0; i < QK_K; i++ {
			output[b*QK_K+i] = float32(int8(blk[2+i])) * d
		}
	}
	return nil
}

// DequantIQ1_M dequantizes IQ1_M blocks (256 values, 56 bytes).
func DequantIQ1_M(input []byte, output []float32) error {
	if len(input)%BLOCK_IQ1_M_SIZE != 0 {
		return &Error{Message: "iq1_m input not aligned"}
	}
	blocks := len(input) / BLOCK_IQ1_M_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_IQ1_M_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		for i := 0; i < QK_K; i++ {
			output[b*QK_K+i] = float32(int8(blk[2+i])) * d
		}
	}
	return nil
}

// DequantIQ2_XXS dequantizes IQ2_XXS blocks (256 values, 66 bytes).
func DequantIQ2_XXS(input []byte, output []float32) error {
	if len(input)%BLOCK_IQ2_XXS_SIZE != 0 {
		return &Error{Message: "iq2_xxs input not aligned"}
	}
	blocks := len(input) / BLOCK_IQ2_XXS_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_IQ2_XXS_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		for i := 0; i < QK_K; i++ {
			output[b*QK_K+i] = float32(int8(blk[2+i])) * d
		}
	}
	return nil
}

// DequantIQ2_XS dequantizes IQ2_XS blocks (256 values, 74 bytes).
func DequantIQ2_XS(input []byte, output []float32) error {
	if len(input)%BLOCK_IQ2_XS_SIZE != 0 {
		return &Error{Message: "iq2_xs input not aligned"}
	}
	blocks := len(input) / BLOCK_IQ2_XS_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_IQ2_XS_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		for i := 0; i < QK_K; i++ {
			output[b*QK_K+i] = float32(int8(blk[2+i])) * d
		}
	}
	return nil
}

// DequantIQ2_S dequantizes IQ2_S blocks (256 values, 82 bytes).
func DequantIQ2_S(input []byte, output []float32) error {
	if len(input)%BLOCK_IQ2_S_SIZE != 0 {
		return &Error{Message: "iq2_s input not aligned"}
	}
	blocks := len(input) / BLOCK_IQ2_S_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_IQ2_S_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		for i := 0; i < QK_K; i++ {
			output[b*QK_K+i] = float32(int8(blk[2+i])) * d
		}
	}
	return nil
}

// DequantIQ3_XXS dequantizes IQ3_XXS blocks (256 values, 98 bytes).
func DequantIQ3_XXS(input []byte, output []float32) error {
	if len(input)%BLOCK_IQ3_XXS_SIZE != 0 {
		return &Error{Message: "iq3_xxs input not aligned"}
	}
	blocks := len(input) / BLOCK_IQ3_XXS_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_IQ3_XXS_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		for i := 0; i < QK_K; i++ {
			output[b*QK_K+i] = float32(int8(blk[2+i])) * d
		}
	}
	return nil
}

// DequantIQ3_S dequantizes IQ3_S blocks (256 values, 110 bytes).
func DequantIQ3_S(input []byte, output []float32) error {
	if len(input)%BLOCK_IQ3_S_SIZE != 0 {
		return &Error{Message: "iq3_s input not aligned"}
	}
	blocks := len(input) / BLOCK_IQ3_S_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_IQ3_S_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		for i := 0; i < QK_K; i++ {
			output[b*QK_K+i] = float32(int8(blk[2+i])) * d
		}
	}
	return nil
}

// DequantIQ4_NL dequantizes IQ4_NL blocks (256 values, 18 bytes).
func DequantIQ4_NL(input []byte, output []float32) error {
	if len(input)%BLOCK_IQ4_NL_SIZE != 0 {
		return &Error{Message: "iq4_nl input not aligned"}
	}
	blocks := len(input) / BLOCK_IQ4_NL_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_IQ4_NL_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		for i := 0; i < QK_K; i++ {
			qs := blk[2+i/2]
			var nib uint8
			if i%2 == 0 {
				nib = qs & 0x0F
			} else {
				nib = (qs >> 4) & 0x0F
			}
			output[b*QK_K+i] = float32(nib) * d
		}
	}
	return nil
}

// DequantIQ4_XS dequantizes IQ4_XS blocks (256 values, 34 bytes).
func DequantIQ4_XS(input []byte, output []float32) error {
	if len(input)%BLOCK_IQ4_XS_SIZE != 0 {
		return &Error{Message: "iq4_xs input not aligned"}
	}
	blocks := len(input) / BLOCK_IQ4_XS_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_IQ4_XS_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		for i := 0; i < QK_K; i++ {
			qs := blk[2+i/2]
			var nib uint8
			if i%2 == 0 {
				nib = qs & 0x0F
			} else {
				nib = (qs >> 4) & 0x0F
			}
			output[b*QK_K+i] = float32(nib) * d
		}
	}
	return nil
}

// DQuantize dispatches to the correct dequantize function for the type.
func DQuantize(t Type, input []byte, output []float32) error {
	switch t {
	case TypeF32:
		return DequantF32(input, output)
	case TypeF16:
		return DequantF16(input, output)
	case TypeQ4_0:
		return DequantQ4_0(input, output)
	case TypeQ4_1:
		return DequantQ4_1(input, output)
	case TypeQ5_0:
		return DequantQ5_0(input, output)
	case TypeQ5_1:
		return DequantQ5_1(input, output)
	case TypeQ8_0:
		return DequantQ8_0(input, output)
	case TypeQ8_K:
		return DequantQ8_K(input, output)
	case TypeQ2_K:
		return DequantQ2_K(input, output)
	case TypeQ3_K_S, TypeQ3_K_M, TypeQ3_K_L:
		return DequantQ3_K(input, output)
	case TypeQ4_K_S, TypeQ4_K_M:
		return DequantQ4_K(input, output)
	case TypeQ5_K_S, TypeQ5_K_M:
		return DequantQ5_K(input, output)
	case TypeQ6_K:
		return DequantQ6_K(input, output)
	case TypeNVFP4:
		return DequantNVFP4(input, output)
	case TypeIQ1_S:
		return DequantIQ1_S(input, output)
	case TypeIQ1_M:
		return DequantIQ1_M(input, output)
	case TypeIQ2_XXS:
		return DequantIQ2_XXS(input, output)
	case TypeIQ2_XS:
		return DequantIQ2_XS(input, output)
	case TypeIQ2_S:
		return DequantIQ2_S(input, output)
	case TypeIQ3_XXS:
		return DequantIQ3_XXS(input, output)
	case TypeIQ3_S:
		return DequantIQ3_S(input, output)
	case TypeIQ4_NL:
		return DequantIQ4_NL(input, output)
	case TypeIQ4_XS:
		return DequantIQ4_XS(input, output)
	default:
		return &Error{Message: "unsupported type " + t.String()}
	}
}

// ensure math import is used even if compiler doesn't tree-shake
var _ = math.Floor
