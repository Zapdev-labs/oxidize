package quantization

import (
	"encoding/binary"
	"math"
)

// DequantF32 copies a 32-bit float buffer.
func DequantF32(input []byte, output []float32) error {
	if len(output)*4 < len(input) {
		return &Error{Message: "output too small"}
	}
	n := len(input) / 4
	for i := 0; i < n; i++ {
		output[i] = math.Float32frombits(binary.LittleEndian.Uint32(input[i*4:]))
	}
	return nil
}

// DequantF16 converts a 16-bit half-precision buffer to float32.
func DequantF16(input []byte, output []float32) error {
	if len(input) == 0 {
		return nil
	}
	if len(output)*2 < len(input) {
		return &Error{Message: "output too small"}
	}
	n := len(input) / 2
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint16(input[i*2:])
		output[i] = f16BitsToF32(bits)
	}
	return nil
}

func f16BitsToF32(bits uint16) float32 {
	sign := uint32(bits>>15) & 0x1
	exp := uint32(bits>>10) & 0x1F
	mant := uint32(bits) & 0x3FF
	var out uint32
	switch {
	case exp == 0:
		if mant == 0 {
			out = sign << 31
		} else {
			for mant&0x400 == 0 {
				mant <<= 1
				exp--
			}
			exp++
			mant &= 0x3FF
			out = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
		}
	case exp == 0x1F:
		out = (sign << 31) | (0xFF << 23) | (mant << 13)
	default:
		out = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
	}
	return math.Float32frombits(out)
}

// DequantQ4_0 dequantizes Q4_0 blocks. Each block is 18 bytes: 2-byte f16
// scale + 32 4-bit quantized values.
func DequantQ4_0(input []byte, output []float32) error {
	if len(input) == 0 {
		return nil
	}
	if len(input)%BLOCK_Q4_0_SIZE != 0 {
		return &Error{Message: "q4_0 input length not aligned"}
	}
	blocks := len(input) / BLOCK_Q4_0_SIZE
	if len(output) < blocks*QK4_0 {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_Q4_0_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		qs := blk[2 : 2+QK4_0/2]
		out := output[b*QK4_0:]
		for i := 0; i < QK4_0/2; i++ {
			v0 := int8((qs[i] & 0x0F) - 8)
			v1 := int8((qs[i]>>4)&0x0F - 8)
			out[i*2] = float32(v0) * d
			out[i*2+1] = float32(v1) * d
		}
	}
	return nil
}

// DequantQ4_1 dequantizes Q4_1 blocks. Each block is 20 bytes: f16 scale +
// f16 min + 32 4-bit values.
func DequantQ4_1(input []byte, output []float32) error {
	if len(input)%BLOCK_Q4_1_SIZE != 0 {
		return &Error{Message: "q4_1 input not aligned"}
	}
	blocks := len(input) / BLOCK_Q4_1_SIZE
	if len(output) < blocks*QK4_1 {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_Q4_1_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		m := f16BitsToF32(binary.LittleEndian.Uint16(blk[2:4]))
		qs := blk[4 : 4+QK4_1/2]
		out := output[b*QK4_1:]
		for i := 0; i < QK4_1/2; i++ {
			v0 := float32(qs[i] & 0x0F)
			v1 := float32((qs[i] >> 4) & 0x0F)
			out[i*2] = v0*d + m
			out[i*2+1] = v1*d + m
		}
	}
	return nil
}

// DequantQ5_0 dequantizes Q5_0 blocks. Each block is 22 bytes: f16 scale +
// 4 bytes of high bits + 16 bytes of low nibbles.
func DequantQ5_0(input []byte, output []float32) error {
	if len(input)%BLOCK_Q5_0_SIZE != 0 {
		return &Error{Message: "q5_0 input not aligned"}
	}
	blocks := len(input) / BLOCK_Q5_0_SIZE
	if len(output) < blocks*QK5_0 {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_Q5_0_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		qh := binary.LittleEndian.Uint32(blk[2:6])
		qs := blk[6 : 6+QK5_0/2]
		out := output[b*QK5_0:]
		for i := 0; i < QK5_0; i++ {
			high := (qh >> i) & 1
			var lo uint8
			if i%2 == 0 {
				lo = qs[i/2] & 0x0F
			} else {
				lo = (qs[i/2] >> 4) & 0x0F
			}
			v := float32(int8((uint32(high)<<4)|uint32(lo)) - 16)
			out[i] = v * d
		}
	}
	return nil
}

// DequantQ5_1 dequantizes Q5_1 blocks.
func DequantQ5_1(input []byte, output []float32) error {
	if len(input)%BLOCK_Q5_1_SIZE != 0 {
		return &Error{Message: "q5_1 input not aligned"}
	}
	blocks := len(input) / BLOCK_Q5_1_SIZE
	if len(output) < blocks*QK5_1 {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_Q5_1_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		m := f16BitsToF32(binary.LittleEndian.Uint16(blk[2:4]))
		qh := binary.LittleEndian.Uint32(blk[4:8])
		qs := blk[8 : 8+QK5_1/2]
		out := output[b*QK5_1:]
		for i := 0; i < QK5_1; i++ {
			high := (qh >> i) & 1
			var lo uint8
			if i%2 == 0 {
				lo = qs[i/2] & 0x0F
			} else {
				lo = (qs[i/2] >> 4) & 0x0F
			}
			v := float32((uint32(high)<<4)|uint32(lo)) * d + m
			out[i] = v
		}
	}
	return nil
}

// DequantQ8_0 dequantizes Q8_0 blocks: f16 scale + 32 int8 quants.
func DequantQ8_0(input []byte, output []float32) error {
	if len(input)%BLOCK_Q8_0_SIZE != 0 {
		return &Error{Message: "q8_0 input not aligned"}
	}
	blocks := len(input) / BLOCK_Q8_0_SIZE
	if len(output) < blocks*QK8_0 {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_Q8_0_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		qs := blk[2 : 2+QK8_0]
		out := output[b*QK8_0:]
		for i, q := range qs {
			out[i] = float32(int8(q)) * d
		}
	}
	return nil
}

// DequantQ8_K is a non-standard but commonly seen format - 292-byte block.
func DequantQ8_K(input []byte, output []float32) error {
	if len(input)%BLOCK_Q8_K_SIZE != 0 {
		return &Error{Message: "q8_k input not aligned"}
	}
	blocks := len(input) / BLOCK_Q8_K_SIZE
	if len(output) < blocks*QK_K {
		return &Error{Message: "output too small"}
	}
	for b := 0; b < blocks; b++ {
		blk := input[b*BLOCK_Q8_K_SIZE:]
		d := f16BitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		qs := blk[2 : 2+QK_K]
		out := output[b*QK_K:]
		for i, q := range qs {
			out[i] = float32(int8(q)) * d
		}
	}
	return nil
}
