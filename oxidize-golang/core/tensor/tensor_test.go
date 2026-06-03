package tensor

import (
	"math"
	"testing"
)

func TestGemvF32(t *testing.T) {
	matrix := []float32{1, 2, 3, 4}
	vector := []float32{5, 6}
	output := make([]float32, 2)
	if err := GemvF32(matrix, 2, 2, vector, output); err != nil {
		t.Fatalf("err: %v", err)
	}
	if math.Abs(float64(output[0]-(1*5+2*6))) > 1e-6 || math.Abs(float64(output[1]-(3*5+4*6))) > 1e-6 {
		t.Fatalf("output = %v", output)
	}
}

func TestGemvF32Transposed(t *testing.T) {
	matrix := []float32{1, 2, 3, 4, 5, 6} // 2x3
	vector := []float32{7, 8}
	output := make([]float32, 3)
	if err := GemvF32Transposed(matrix, 2, 3, vector, output); err != nil {
		t.Fatalf("err: %v", err)
	}
	want := []float32{1*7 + 4*8, 2*7 + 5*8, 3*7 + 6*8}
	for i, v := range want {
		if math.Abs(float64(output[i]-v)) > 1e-6 {
			t.Fatalf("output[%d] = %f, want %f", i, output[i], v)
		}
	}
}

func TestGemmF32(t *testing.T) {
	left := []float32{1, 2, 3, 4}   // 2x2
	right := []float32{5, 6, 7, 8}  // 2x2
	output := make([]float32, 4)    // 2x2
	if err := GemmF32(left, right, 2, 2, 2, output); err != nil {
		t.Fatalf("err: %v", err)
	}
	want := []float32{19, 22, 43, 50}
	for i, v := range want {
		if math.Abs(float64(output[i]-v)) > 1e-6 {
			t.Fatalf("output[%d] = %f, want %f", i, output[i], v)
		}
	}
}

func TestRmsNorm(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	output := make([]float32, 4)
	if err := RMSNormF32(input, weight, output, 1e-6); err != nil {
		t.Fatalf("err: %v", err)
	}
	// input[0] * inv * weight[0] where inv = 1 / sqrt(mean + eps) and
	// mean = (1+4+9+16)/4. The test must compute the mean as a float to
	// avoid integer truncation.
	const mean = (1.0 + 4.0 + 9.0 + 16.0) / 4.0
	want := 1.0 / float32(math.Sqrt(mean))
	if math.Abs(float64(output[0]-want)) > 1e-3 {
		t.Fatalf("rms[0] = %f, want %f", output[0], want)
	}
}

func TestLayerNorm(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	bias := []float32{0, 0, 0, 0}
	output := make([]float32, 4)
	if err := LayerNormF32(input, weight, bias, output, 1e-6); err != nil {
		t.Fatalf("err: %v", err)
	}
	if math.Abs(float64(output[0]+1.3416407)) > 1e-3 {
		t.Fatalf("layer_norm[0] = %f, want ~-1.34", output[0])
	}
}

func TestSoftmax(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	output := make([]float32, 4)
	if err := SoftmaxF32(input, output, 4); err != nil {
		t.Fatalf("err: %v", err)
	}
	var sum float32
	for _, v := range output {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-6 {
		t.Fatalf("softmax sum = %f", sum)
	}
}

func TestSwiGLU(t *testing.T) {
	gate := []float32{0, 1, 2}
	up := []float32{1, 1, 1}
	output := make([]float32, 3)
	if err := ApplySwiGLUF32(gate, up, output); err != nil {
		t.Fatalf("err: %v", err)
	}
	if output[0] != 0 {
		t.Fatalf("swiglu[0] = %f", output[0])
	}
}

func TestRope(t *testing.T) {
	input := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	output := make([]float32, 8)
	if err := ApplyRopeF32(input, output, 0, 4, 10000); err != nil {
		t.Fatalf("err: %v", err)
	}
	if math.Abs(float64(output[0])-1) > 1e-5 {
		t.Fatalf("rope[0] = %f, want 1", output[0])
	}
}

func TestF16Roundtrip(t *testing.T) {
	values := []float32{0, 1, -1, 0.5, -0.5, 100, -100, 0.001}
	for _, v := range values {
		bits := F32ToF16Bits(v)
		round := F16BitsToF32(bits)
		if math.Abs(float64(v-round)) > 0.01 {
			t.Fatalf("f16 roundtrip %f -> %f", v, round)
		}
	}
}

func TestExtractBits(t *testing.T) {
	stream := []byte{0x12, 0x34, 0x56, 0x78}
	// 4-bit field at index 0 = 0x2
	if got := ExtractBits(stream, 0, 4); got != 0x2 {
		t.Fatalf("extract[0] = %x", got)
	}
	// 4-bit field at index 1 = 0x1
	if got := ExtractBits(stream, 1, 4); got != 0x1 {
		t.Fatalf("extract[1] = %x", got)
	}
	// 8-bit field at index 0 = 0x12
	if got := ExtractBits(stream, 0, 8); got != 0x12 {
		t.Fatalf("extract[0,8] = %x", got)
	}
}

func TestTensorRoundtrip(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	t1 := New(data, []int{2, 3})
	if t1 == nil {
		t.Fatal("nil tensor")
	}
	if t1.NumElements() != 6 {
		t.Fatalf("num = %d", t1.NumElements())
	}
	if t1.At(0) != 1 || t1.At(5) != 6 {
		t.Fatalf("values mismatch: %v", t1.Data())
	}
	t1.Set(0, 42)
	if t1.At(0) != 42 {
		t.Fatalf("set failed")
	}
	if t1.DType() != DTypeF32 {
		t.Fatalf("dtype = %v", t1.DType())
	}
}
