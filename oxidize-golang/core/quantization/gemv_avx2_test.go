package quantization

import (
	"math"
	"testing"
)

// dequantQ8KRef dequantizes one Q8_K block to f32 for the reference dot product.
func dequantQ8KRef(blk []byte) []float32 {
	d := math.Float32frombits(uint32(blk[0]) | uint32(blk[1])<<8 | uint32(blk[2])<<16 | uint32(blk[3])<<24)
	out := make([]float32, QK_K)
	for i := 0; i < QK_K; i++ {
		out[i] = d * float32(int8(blk[4+i]))
	}
	return out
}

func TestQuantizeVectorQ8KRoundTrip(t *testing.T) {
	vec := make([]float32, QK_K)
	for i := range vec {
		vec[i] = float32(math.Sin(float64(i) * 0.3))
	}
	out := make([]byte, BLOCK_Q8_K_SIZE)
	if err := QuantizeVectorQ8KInto(vec, 1, out); err != nil {
		t.Fatal(err)
	}
	deq := dequantQ8KRef(out)
	var maxErr float64
	for i := range vec {
		e := math.Abs(float64(vec[i] - deq[i]))
		if e > maxErr {
			maxErr = e
		}
	}
	if maxErr > 0.02 {
		t.Fatalf("q8_k round-trip error too high: %v", maxErr)
	}
	// bsums must match recomputed group sums.
	bsumsOff := 4 + QK_K
	for g := 0; g < QK_K/16; g++ {
		var want int32
		for i := 0; i < 16; i++ {
			want += int32(int8(out[4+g*16+i]))
		}
		got := readQ8KBsum(out[bsumsOff:], g)
		if got != want {
			t.Fatalf("bsum[%d] = %d, want %d", g, got, want)
		}
	}
}

func TestQ4KQ8KRowDotMatchesReference(t *testing.T) {
	const blocks = 3
	const cols = blocks * QK_K

	// Build a weight row and an input vector.
	weights := make([]float32, cols)
	vec := make([]float32, cols)
	for i := range weights {
		weights[i] = float32(math.Cos(float64(i)*0.11)) * 1.5
		vec[i] = float32(math.Sin(float64(i)*0.07)) * 0.8
	}

	rowBytes := blocks * BLOCK_Q4_K_SIZE
	q4 := make([]byte, rowBytes)
	if err := QuantizeScalar(TypeQ4_K_M, weights, q4, nil); err != nil {
		t.Fatal(err)
	}
	q8 := make([]byte, blocks*BLOCK_Q8_K_SIZE)
	if err := QuantizeVectorQ8KInto(vec, blocks, q8); err != nil {
		t.Fatal(err)
	}

	// Reference: dequant both, take float dot.
	wDeq := make([]float32, cols)
	if err := DequantQ4_K(q4, wDeq); err != nil {
		t.Fatal(err)
	}
	var ref float32
	for b := 0; b < blocks; b++ {
		vDeq := dequantQ8KRef(q8[b*BLOCK_Q8_K_SIZE : (b+1)*BLOCK_Q8_K_SIZE])
		for i := 0; i < QK_K; i++ {
			ref += wDeq[b*QK_K+i] * vDeq[i]
		}
	}

	got := Q4KQ8KRowDot(q4, blocks, q8)
	if rel := math.Abs(float64(got-ref)) / (math.Abs(float64(ref)) + 1e-6); rel > 1e-4 {
		t.Fatalf("Q4KQ8KRowDot = %v, ref = %v (rel %v)", got, ref, rel)
	}
}

func TestQ4KQ8KRowDotX4MatchesScalar(t *testing.T) {
	const blocks = 2
	const cols = blocks * QK_K
	rowBytes := blocks * BLOCK_Q4_K_SIZE

	rows := make([]byte, 4*rowBytes)
	for r := 0; r < 4; r++ {
		w := make([]float32, cols)
		for i := range w {
			w[i] = float32(math.Sin(float64(i+r*7)*0.13)) * (1 + float32(r)*0.3)
		}
		if err := QuantizeScalar(TypeQ4_K_M, w, rows[r*rowBytes:(r+1)*rowBytes], nil); err != nil {
			t.Fatal(err)
		}
	}
	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = float32(math.Cos(float64(i) * 0.05))
	}
	q8 := make([]byte, blocks*BLOCK_Q8_K_SIZE)
	if err := QuantizeVectorQ8KInto(vec, blocks, q8); err != nil {
		t.Fatal(err)
	}

	var out4 [4]float32
	Q4KQ8KRowDotX4(rows, rowBytes, blocks, q8, &out4)
	for r := 0; r < 4; r++ {
		want := Q4KQ8KRowDot(rows[r*rowBytes:(r+1)*rowBytes], blocks, q8)
		if math.Abs(float64(out4[r]-want)) > 1e-3 {
			t.Fatalf("row %d: x4=%v scalar=%v", r, out4[r], want)
		}
	}
}
