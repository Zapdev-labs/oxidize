package flash_attention

import (
	"math"
	"testing"
)

// f32ToF16Bits is a reference conversion used only in tests.
func f32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF
	if exp <= 0 {
		return sign
	}
	if exp >= 0x1F {
		return sign | 0x7C00
	}
	return sign | uint16(exp)<<10 | uint16(mant>>13)
}

func TestF16BitsToF32RoundTrip(t *testing.T) {
	vals := []float32{0, 1, -1, 0.5, -0.25, 2, 3.5, 100, -64, 0.001}
	for _, v := range vals {
		got := f16BitsToF32(f32ToF16Bits(v))
		if math.Abs(float64(got-v)) > math.Abs(float64(v))*0.01+1e-3 {
			t.Fatalf("f16 round trip %v -> %v", v, got)
		}
	}
}

func TestDotProductF32F16MatchesScalar(t *testing.T) {
	n := 40
	a := make([]float32, n)
	bf := make([]float32, n)
	b := make([]uint16, n)
	for i := 0; i < n; i++ {
		a[i] = float32(math.Sin(float64(i) * 0.3))
		bf[i] = float32(math.Cos(float64(i) * 0.2))
		b[i] = f32ToF16Bits(bf[i])
	}
	got := DotProductF32F16(a, b)
	var want float32
	for i := 0; i < n; i++ {
		want += a[i] * f16BitsToF32(b[i])
	}
	if math.Abs(float64(got-want)) > 1e-4 {
		t.Fatalf("dot f16: got %v want %v", got, want)
	}
}

func TestAxpyF32(t *testing.T) {
	out := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	row := []float32{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	want := make([]float32, len(out))
	for i := range out {
		want[i] = out[i] + 2*row[i]
	}
	AxpyF32(out, 2, row)
	for i := range out {
		if out[i] != want[i] {
			t.Fatalf("axpy[%d] = %v want %v", i, out[i], want[i])
		}
	}
}

// TestFlashAttentionDecodeF16MatchesF32 verifies the f16 decode path agrees
// with the f32 GQA decode within f16 precision.
func TestFlashAttentionDecodeF16MatchesF32(t *testing.T) {
	headDim := 8
	kvHeads := 2
	kvLen := headDim * kvHeads
	seqLen := 5
	kvHead := 1

	query := make([]float32, headDim)
	for i := range query {
		query[i] = float32(math.Sin(float64(i) * 0.5))
	}
	keyF := make([]float32, seqLen*kvLen)
	valF := make([]float32, seqLen*kvLen)
	keyH := make([]uint16, seqLen*kvLen)
	valH := make([]uint16, seqLen*kvLen)
	for i := range keyF {
		keyF[i] = float32(math.Sin(float64(i) * 0.13))
		valF[i] = float32(math.Cos(float64(i) * 0.07))
		keyH[i] = f32ToF16Bits(keyF[i])
		valH[i] = f32ToF16Bits(valF[i])
		// Round f32 reference through f16 too for a fair comparison.
		keyF[i] = f16BitsToF32(keyH[i])
		valF[i] = f16BitsToF32(valH[i])
	}

	outF := make([]float32, headDim)
	if err := FlashAttentionDecodeGQA(query, keyF, valF, outF, seqLen, headDim, kvLen, kvHead); err != nil {
		t.Fatal(err)
	}
	outH := make([]float32, headDim)
	if err := FlashAttentionDecodeF16(query, keyH, valH, outH, seqLen, headDim, kvLen, kvHead); err != nil {
		t.Fatal(err)
	}
	for d := 0; d < headDim; d++ {
		if math.Abs(float64(outF[d]-outH[d])) > 1e-4 {
			t.Fatalf("dim %d: f32=%v f16=%v", d, outF[d], outH[d])
		}
	}
}
