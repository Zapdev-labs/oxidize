package quantization

import (
	"encoding/binary"
	"math"
)

// QuantizeVectorQ8KInto quantizes `vector` (length nBlocks*256) into nBlocks
// Q8_K blocks written to `out`. Mirrors quantize_vector_q8_k_into in
// oxidize-core/src/compute/tensor/kernels/q_kernels.rs.
//
// Each Q8_K block is 292 bytes: 4 bytes d (f32 LE), 256 bytes qs (i8), then
// 16 int16 LE bsums (one per 16-element group).
func QuantizeVectorQ8KInto(vector []float32, nBlocks int, out []byte) error {
	if len(vector) < nBlocks*QK_K {
		return &Error{Message: "q8_k: input vector too small"}
	}
	if len(out) < nBlocks*BLOCK_Q8_K_SIZE {
		return &Error{Message: "q8_k: output buffer too small"}
	}
	for b := 0; b < nBlocks; b++ {
		in := vector[b*QK_K : (b+1)*QK_K]
		blk := out[b*BLOCK_Q8_K_SIZE : (b+1)*BLOCK_Q8_K_SIZE]
		quantizeBlockQ8KScalar(in, blk)
	}
	return nil
}

func quantizeBlockQ8KScalar(in []float32, out []byte) {
	var amax, max float32
	for _, v := range in {
		av := float32(math.Abs(float64(v)))
		if av > amax {
			amax = av
			max = v
		}
	}
	if amax == 0 {
		for i := range out {
			out[i] = 0
		}
		return
	}
	// iscale = -128 / max (sign-preserving), d = 1/iscale.
	iscale := float32(-128) / max
	d := 1 / iscale
	binary.LittleEndian.PutUint32(out[0:4], math.Float32bits(d))
	const qsOff = 4
	for i, v := range in {
		q := int32(math.Round(float64(iscale * v)))
		if q < -128 {
			q = -128
		} else if q > 127 {
			q = 127
		}
		out[qsOff+i] = byte(int8(q))
	}
	bsumsOff := qsOff + QK_K
	for g := 0; g < QK_K/16; g++ {
		var sum int32
		for i := 0; i < 16; i++ {
			sum += int32(int8(out[qsOff+g*16+i]))
		}
		if sum < math.MinInt16 {
			sum = math.MinInt16
		} else if sum > math.MaxInt16 {
			sum = math.MaxInt16
		}
		binary.LittleEndian.PutUint16(out[bsumsOff+g*2:bsumsOff+g*2+2], uint16(int16(sum)))
	}
}

func readQ8KBsum(bsums []byte, idx int) int32 {
	return int32(int16(binary.LittleEndian.Uint16(bsums[idx*2 : idx*2+2])))
}

// Q4KQ8KRowDot computes one output-row dot product of a Q4_K weight row against
// a Q8_K-quantized input vector, using integer multiply-adds. This mirrors the
// math of q4_k_q8_k_row_dot_avx2 (and its scalar fallback) in Rust: the result
// is bit-equivalent to dequantizing the Q4_K row and the Q8_K vector and taking
// a float dot product, but stays in integer arithmetic per super-block.
//
//   row            : blocksPerRow Q4_K blocks (144 bytes each)
//   blocksPerRow   : number of 256-element super-blocks in the row
//   q8k            : blocksPerRow Q8_K blocks (292 bytes each)
func Q4KQ8KRowDot(row []byte, blocksPerRow int, q8k []byte) float32 {
	var acc float32
	for blockIdx := 0; blockIdx < blocksPerRow; blockIdx++ {
		wOff := blockIdx * BLOCK_Q4_K_SIZE
		q8Off := blockIdx * BLOCK_Q8_K_SIZE
		block := row[wOff : wOff+BLOCK_Q4_K_SIZE]
		q8Block := q8k[q8Off : q8Off+BLOCK_Q8_K_SIZE]

		dW := f16BitsToF32(binary.LittleEndian.Uint16(block[0:2]))
		dminW := f16BitsToF32(binary.LittleEndian.Uint16(block[2:4]))
		dQ8 := math.Float32frombits(binary.LittleEndian.Uint32(q8Block[0:4]))
		scales := block[4:16]
		qs := block[16:144]
		q8 := q8Block[4 : 4+QK_K]
		bsums := q8Block[4+QK_K:]

		var posAcc int32
		var minAcc int32
		// 8 sub-groups of 32 elements: low/high nibble of each 32-byte chunk.
		for gp := 0; gp < 4; gp++ {
			g1 := gp * 2
			g2 := g1 + 1
			s1, ms1 := scaleMinK4(g1, scales)
			s2, ms2 := scaleMinK4(g2, scales)
			base := gp * 32
			var p1, p2 int32
			for l := 0; l < 32; l++ {
				packed := qs[base+l]
				lo := int32(packed & 0x0F)
				hi := int32(packed >> 4)
				p1 += lo * int32(int8(q8[g1*32+l]))
				p2 += hi * int32(int8(q8[g2*32+l]))
			}
			posAcc += int32(s1) * p1
			posAcc += int32(s2) * p2

			bs1 := readQ8KBsum(bsums, g1*2) + readQ8KBsum(bsums, g1*2+1)
			bs2 := readQ8KBsum(bsums, g2*2) + readQ8KBsum(bsums, g2*2+1)
			minAcc += int32(ms1) * bs1
			minAcc += int32(ms2) * bs2
		}
		acc += dW*dQ8*float32(posAcc) - dminW*dQ8*float32(minAcc)
	}
	return acc
}

// Q4KQ8KRowDotX4 computes 4 consecutive Q4_K weight rows against one shared
// Q8_K input vector, writing the four dot products to out. The Q8_K sub-group
// loads and bsum pair-sums are computed once per block and reused across all
// four rows (cache-locality win), mirroring q4_k_q8_k_row_dot_x4_avx2.
//
//   rows         : at least 4*rowBytes bytes (4 Q4_K rows spaced rowBytes apart)
//   rowBytes     : stride between rows (blocksPerRow*144 typically)
//   blocksPerRow : super-blocks per row
//   q8k          : blocksPerRow Q8_K blocks
func Q4KQ8KRowDotX4(rows []byte, rowBytes, blocksPerRow int, q8k []byte, out *[4]float32) {
	var acc [4]int32Pair
	for r := 0; r < 4; r++ {
		acc[r] = int32Pair{}
	}
	for blockIdx := 0; blockIdx < blocksPerRow; blockIdx++ {
		q8Off := blockIdx * BLOCK_Q8_K_SIZE
		q8Block := q8k[q8Off : q8Off+BLOCK_Q8_K_SIZE]
		dQ8 := math.Float32frombits(binary.LittleEndian.Uint32(q8Block[0:4]))
		q8 := q8Block[4 : 4+QK_K]
		bsums := q8Block[4+QK_K:]

		// Shared bsum pair-sums for all 8 groups.
		var bs [8]int32
		for g := 0; g < 8; g++ {
			bs[g] = readQ8KBsum(bsums, g*2) + readQ8KBsum(bsums, g*2+1)
		}

		for r := 0; r < 4; r++ {
			wOff := r*rowBytes + blockIdx*BLOCK_Q4_K_SIZE
			block := rows[wOff : wOff+BLOCK_Q4_K_SIZE]
			dW := f16BitsToF32(binary.LittleEndian.Uint16(block[0:2]))
			dminW := f16BitsToF32(binary.LittleEndian.Uint16(block[2:4]))
			scales := block[4:16]
			qs := block[16:144]

			var posAcc int32
			var minAcc int32
			for gp := 0; gp < 4; gp++ {
				g1 := gp * 2
				g2 := g1 + 1
				s1, ms1 := scaleMinK4(g1, scales)
				s2, ms2 := scaleMinK4(g2, scales)
				base := gp * 32
				var p1, p2 int32
				for l := 0; l < 32; l++ {
					packed := qs[base+l]
					p1 += int32(packed&0x0F) * int32(int8(q8[g1*32+l]))
					p2 += int32(packed>>4) * int32(int8(q8[g2*32+l]))
				}
				posAcc += int32(s1) * p1
				posAcc += int32(s2) * p2
				minAcc += int32(ms1) * bs[g1]
				minAcc += int32(ms2) * bs[g2]
			}
			acc[r].pos += dW * dQ8 * float32(posAcc)
			acc[r].neg += dminW * dQ8 * float32(minAcc)
		}
	}
	for r := 0; r < 4; r++ {
		out[r] = acc[r].pos - acc[r].neg
	}
}

type int32Pair struct {
	pos float32
	neg float32
}
