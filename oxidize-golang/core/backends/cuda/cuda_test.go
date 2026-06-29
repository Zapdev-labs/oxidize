package cudabackend

import (
	"math"
	"testing"

	quant "github.com/Zapdev-labs/oxidize/golang/core/quantization"
)

func resetState(t *testing.T) {
	t.Helper()
	if err := SetLayerConfig(CudaLayerConfig{}); err != nil {
		t.Fatalf("set config: %v", err)
	}
	if err := ClearResidentCache(); err != nil {
		t.Fatalf("clear: %v", err)
	}
}

func TestBuildInfo(t *testing.T) {
	info := Info()
	if info.DetectedAtBuild && info.CudaPath == "" {
		t.Fatal("native cuda build should set CudaPath")
	}
}

func TestValidateGemvDims(t *testing.T) {
	if err := ValidateGemvDims(0, 0); err == nil {
		t.Fatal("expected error for zero dims")
	}
	if err := ValidateGemvDims(1, 1); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestValidateQ8_0GemvDims(t *testing.T) {
	if err := ValidateQ8_0GemvDims(1, 33); err == nil {
		t.Fatal("expected misalignment error")
	}
	if err := ValidateQ8_0GemvDims(1, 32); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestF32PoolReuse(t *testing.T) {
	s := newGpuState()
	a := s.getF32Buffer(16)
	for i := range a {
		a[i] = float32(i)
	}
	s.returnF32Buffer(a)
	b := s.getF32Buffer(16)
	if &a[0] != &b[0] {
		t.Fatal("expected pooled buffer to be reused")
	}
	for i := range b {
		if b[i] != 0 {
			t.Fatalf("pooled buffer not zeroed at %d", i)
		}
	}
}

func TestLayerLRUEviction(t *testing.T) {
	resetState(t)
	defer resetState(t)
	if err := SetLayerConfig(CudaLayerConfig{MaxResidentLayers: 2}); err != nil {
		t.Fatal(err)
	}
	w := func() []F32Weight { return []F32Weight{{Data: []float32{1, 2, 3, 4}, Rows: 2, Cols: 2}} }
	for i := LayerID(0); i < 3; i++ {
		// distinct data so keys differ
		d := []float32{float32(i), 2, 3, 4}
		if err := PreloadLayer(i, []F32Weight{{Data: d, Rows: 2, Cols: 2}}); err != nil {
			t.Fatal(err)
		}
	}
	_ = w
	if err := withGPU(func(st *GpuState) error {
		if len(st.layerMap) != 2 {
			t.Fatalf("expected 2 resident layers after eviction, got %d", len(st.layerMap))
		}
		if _, ok := st.layerMap[0]; ok {
			t.Fatal("layer 0 should have been evicted (LRU)")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestVramBudgetEnforcement(t *testing.T) {
	resetState(t)
	defer resetState(t)
	// Each layer = 4 f32 = 16 bytes. Budget 40 bytes -> at most 2 layers.
	if err := SetLayerConfig(CudaLayerConfig{MaxVramBytes: 40}); err != nil {
		t.Fatal(err)
	}
	for i := LayerID(0); i < 4; i++ {
		d := []float32{float32(i) + 0.5, 2, 3, 4}
		if err := PreloadLayer(i, []F32Weight{{Data: d, Rows: 2, Cols: 2}}); err != nil {
			t.Fatal(err)
		}
	}
	if got := ResidentVramBytes(); got > 40 {
		t.Fatalf("resident bytes %d exceeds budget 40", got)
	}
}

func TestGemmF32CudaCorrectness(t *testing.T) {
	// [2x3] * [3x2] = [2x2]
	left := []float32{1, 2, 3, 4, 5, 6}
	right := []float32{7, 8, 9, 10, 11, 12}
	out := make([]float32, 4)
	if err := GemmF32Cuda(left, right, 2, 3, 2, out); err != nil {
		t.Fatal(err)
	}
	// reference
	want := make([]float32, 4)
	for r := 0; r < 2; r++ {
		for c := 0; c < 2; c++ {
			var s float32
			for k := 0; k < 3; k++ {
				s += left[r*3+k] * right[k*2+c]
			}
			want[r*2+c] = s
		}
	}
	for i := range out {
		if math.Abs(float64(out[i]-want[i])) > 1e-4 {
			t.Fatalf("gemm[%d] = %v want %v", i, out[i], want[i])
		}
	}
}

func TestGemvQuantizedDispatchQ8_0(t *testing.T) {
	resetState(t)
	defer resetState(t)
	rows, cols := 2, 32
	matrix := make([]float32, rows*cols)
	for i := range matrix {
		matrix[i] = float32(i%7) - 3
	}
	vector := make([]float32, cols)
	for i := range vector {
		vector[i] = float32(i%5) - 2
	}
	qsize, err := quant.QuantizedSize(quant.TypeQ8_0, rows*cols)
	if err != nil {
		t.Fatal(err)
	}
	qbytes := make([]byte, qsize)
	if err := quant.QuantizeScalar(quant.TypeQ8_0, matrix, qbytes, nil); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, rows)
	if err := GemvQuantizedCuda(qbytes, int(GgmlTypeQ8_0), vector, rows, cols, out, nil); err != nil {
		t.Fatal(err)
	}
	// reference against dequantized matrix
	deq := make([]float32, rows*cols)
	if err := quant.DequantizeScalar(quant.TypeQ8_0, qbytes, deq); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < rows; r++ {
		var want float32
		for c := 0; c < cols; c++ {
			want += deq[r*cols+c] * vector[c]
		}
		if math.Abs(float64(out[r]-want)) > 1e-3 {
			t.Fatalf("gemv[%d] = %v want %v", r, out[r], want)
		}
	}
}

func TestSupportsQuantizedGpu(t *testing.T) {
	if !SupportsQuantizedGpu(GgmlTypeQ8_0) || !SupportsQuantizedGpu(GgmlTypeQ4_K) {
		t.Fatal("Q8_0/Q4_K should be GPU-supported")
	}
	if SupportsQuantizedGpu(GgmlTypeF32) {
		t.Fatal("F32 has no dequant kernel")
	}
}

func TestActivationBuffersAndRmsNorm(t *testing.T) {
	resetState(t)
	defer resetState(t)
	hidden := 8
	if err := GpuInitActivationBuffers(hidden, 16); err != nil {
		t.Fatal(err)
	}
	in := []float32{1, -2, 3, -4, 5, -6, 7, -8}
	if err := GpuUploadHidden(in); err != nil {
		t.Fatal(err)
	}
	weight := make([]float32, hidden)
	for i := range weight {
		weight[i] = 1
	}
	if err := GpuRmsNorm(weight, 1e-5); err != nil {
		t.Fatal(err)
	}
	// verify normed against reference
	var ss float64
	for _, v := range in {
		ss += float64(v) * float64(v)
	}
	scale := float32(1.0 / math.Sqrt(ss/float64(hidden)+1e-5))
	got := make([]float32, hidden)
	if err := withGPU(func(st *GpuState) error {
		copy(got, st.activation.Normed)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		want := in[i] * scale
		if math.Abs(float64(got[i]-want)) > 1e-4 {
			t.Fatalf("normed[%d] = %v want %v", i, got[i], want)
		}
	}
	// download round-trip
	out := make([]float32, hidden)
	if err := GpuDownloadHidden(out); err != nil {
		t.Fatal(err)
	}
	for i := range out {
		if out[i] != in[i] {
			t.Fatalf("download[%d] = %v want %v", i, out[i], in[i])
		}
	}
}

func TestGpuAttnRmsAndQkvQ4K(t *testing.T) {
	resetState(t)
	defer resetState(t)
	hidden := 32
	qLen, kvLen := 32, 32
	if err := GpuInitActivationBuffers(hidden, 64); err != nil {
		t.Fatal(err)
	}
	in := make([]float32, hidden)
	for i := range in {
		in[i] = float32(i%9) - 4
	}
	if err := GpuUploadHidden(in); err != nil {
		t.Fatal(err)
	}
	attnNorm := make([]float32, hidden)
	for i := range attnNorm {
		attnNorm[i] = 1
	}
	mkQ8 := func(rows, cols int, seed float32) []byte {
		m := make([]float32, rows*cols)
		for i := range m {
			m[i] = float32(i%6)*seed - 1
		}
		sz, _ := quant.QuantizedSize(quant.TypeQ8_0, rows*cols)
		b := make([]byte, sz)
		_ = quant.QuantizeScalar(quant.TypeQ8_0, m, b, nil)
		return b
	}
	wq := mkQ8(qLen, hidden, 0.5)
	wk := mkQ8(kvLen, hidden, 0.3)
	wv := mkQ8(kvLen, hidden, 0.2)
	qOut := make([]float32, qLen)
	kOut := make([]float32, kvLen)
	vOut := make([]float32, kvLen)
	if err := GpuAttnRmsAndQkvQ4K(attnNorm, 1e-5, wq, qLen, wk, kvLen, wv, GgmlTypeQ8_0, hidden, qOut, kOut, vOut); err != nil {
		t.Fatal(err)
	}
	// weights should now be resident (cached) for reuse across tokens
	if ResidentVramBytes() == 0 {
		t.Fatal("expected quantized weights to be cached resident")
	}
}
