package quantization

import (
	"encoding/binary"
	"math"
)

// DequantQ2_K dequantizes Q2_K super-blocks (256 values, 84 bytes).
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
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[80:82]))
		min := f16BitsToF32(binary.LittleEndian.Uint16(blk[82:84]))
		scales := blk[0:16]
		qs := blk[16:80]
		out := output[b*QK_K:]
		qPtr := 0
		is := 0
		for outer := 0; outer < 2; outer++ {
			qsBase := outer * 32
			for range 4 {
				sc1 := scales[is]
				dl1 := d * float32(sc1&0xF)
				ml1 := min * float32(sc1>>4)
				is++
				sc2 := scales[is]
				dl2 := d * float32(sc2&0xF)
				ml2 := min * float32(sc2>>4)
				is++
				shift := ((is/2 - 1) % 4) * 2
				for l := 0; l < 16; l++ {
					out[qPtr+l] = dl1*float32((qs[qsBase+l]>>shift)&3) - ml1
				}
				for l := 0; l < 16; l++ {
					out[qPtr+16+l] = dl2*float32((qs[qsBase+16+l]>>shift)&3) - ml2
				}
				qPtr += 32
			}
		}
	}
	return nil
}

// DequantQ3_K dequantizes Q3_K super-blocks (256 values, 110 bytes).
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
		dAll := f16BitsToF32(binary.LittleEndian.Uint16(blk[108:110]))
		hmask := blk[0:32]
		qs := blk[32:96]
		scalesRaw := [4]uint32{
			binary.LittleEndian.Uint32(blk[96:100]),
			binary.LittleEndian.Uint32(blk[100:104]),
			binary.LittleEndian.Uint32(blk[104:108]),
			0,
		}
		tmp := scalesRaw[2]
		scalesRaw[2] = ((scalesRaw[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4)
		scalesRaw[3] = ((scalesRaw[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4)
		scalesRaw[0] = (scalesRaw[0] & 0x0F0F0F0F) | ((tmp & 0x03030303) << 4)
		scalesRaw[1] = (scalesRaw[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4)
		var scaleBytes [16]byte
		for i := 0; i < 4; i++ {
			binary.LittleEndian.PutUint32(scaleBytes[i*4:(i+1)*4], scalesRaw[i])
		}
		scales := make([]int8, 16)
		for i := range scales {
			scales[i] = int8(scaleBytes[i])
		}
		out := output[b*QK_K:]
		qPtr := 0
		is := 0
		m := uint8(1)
		for range 2 {
			for range 4 {
				dl := dAll * float32(int32(scales[is])-32)
				is++
				shift := ((is - 1) % 4) * 2
				for l := 0; l < 16; l++ {
					qv := int32((qs[l] >> shift) & 3)
					hbit := int32(0)
					if (hmask[l] & m) == 0 {
						hbit = 4
					}
					out[qPtr+l] = dl * float32(qv-hbit)
				}
				dl2 := dAll * float32(int32(scales[is])-32)
				is++
				for l := 0; l < 16; l++ {
					qv := int32((qs[l+16] >> shift) & 3)
					hbit := int32(0)
					if (hmask[l+16] & m) == 0 {
						hbit = 4
					}
					out[qPtr+16+l] = dl2 * float32(qv-hbit)
				}
				qPtr += 32
				m <<= 1
			}
		}
	}
	return nil
}

func scaleMinK4(j int, scales []byte) (uint8, uint8) {
	if j < 4 {
		return scales[j] & 63, scales[j+4] & 63
	}
	return (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4), (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
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
		min := f16BitsToF32(binary.LittleEndian.Uint16(blk[2:4]))
		scales := blk[4:16]
		qs := blk[16:144]
		out := output[b*QK_K:]
		qPtr := 0
		is := 0
		for range 4 {
			sc1, m1 := scaleMinK4(is, scales)
			sc2, m2 := scaleMinK4(is+1, scales)
			d1 := d * float32(sc1)
			min1 := min * float32(m1)
			d2 := d * float32(sc2)
			min2 := min * float32(m2)
			for l := 0; l < 32; l++ {
				out[qPtr+l] = d1*float32(qs[l]&0xF) - min1
			}
			for l := 0; l < 32; l++ {
				out[qPtr+32+l] = d2*float32(qs[l]>>4) - min2
			}
			qPtr += 64
			is += 2
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
		min := f16BitsToF32(binary.LittleEndian.Uint16(blk[2:4]))
		scales := blk[4:16]
		qh := blk[16:48]
		qs := blk[48:176]
		out := output[b*QK_K:]
		qPtr := 0
		is := 0
		u1 := uint8(1)
		u2 := uint8(2)
		for range 4 {
			sc1, m1 := scaleMinK4(is, scales)
			sc2, m2 := scaleMinK4(is+1, scales)
			d1 := d * float32(sc1)
			min1 := min * float32(m1)
			d2 := d * float32(sc2)
			min2 := min * float32(m2)
			for l := 0; l < 32; l++ {
				qv1 := uint32(qs[l]&0xF)
				if (qh[l] & u1) != 0 {
					qv1 += 16
				}
				out[qPtr+l] = d1*float32(qv1) - min1
			}
			for l := 0; l < 32; l++ {
				qv2 := uint32(qs[l] >> 4)
				if (qh[l] & u2) != 0 {
					qv2 += 16
				}
				out[qPtr+32+l] = d2*float32(qv2) - min2
			}
			qPtr += 64
			is += 2
			u1 <<= 2
			u2 <<= 2
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
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[208:210]))
		ql := blk[0:128]
		qh := blk[128:192]
		sc := make([]int8, 16)
		for i := range sc {
			sc[i] = int8(blk[192+i])
		}
		out := output[b*QK_K:]
		qPtr := 0
		for range 2 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int32((ql[l]&0xF)|(((qh[l]&3)<<4))) - 32
				q2 := int32((ql[l+32]&0xF)|((((qh[l]>>2)&3)<<4))) - 32
				q3 := int32((ql[l]>>4)|((((qh[l]>>4)&3)<<4))) - 32
				q4 := int32((ql[l+32]>>4)|((((qh[l]>>6)&3)<<4))) - 32
				out[qPtr+l] = d * float32(sc[is]) * float32(q1)
				out[qPtr+32+l] = d * float32(sc[is+2]) * float32(q2)
				out[qPtr+64+l] = d * float32(sc[is+4]) * float32(q3)
				out[qPtr+96+l] = d * float32(sc[is+6]) * float32(q4)
			}
			qPtr += 128
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
