package tensor

import "testing"

// BenchmarkGemvF32Transposed exercises the dense transposed GEMV at lm_head
// scale (the largest single matmul in a decode step). Guards against regressing
// the single-barrier, cache-friendly implementation.
//
//	go test -bench BenchmarkGemvF32Transposed ./core/tensor/
func BenchmarkGemvF32Transposed(b *testing.B) {
	const rows, cols = 896, 151936 // matrix is [inDim][outDim]; out length = cols
	mat := make([]float32, rows*cols)
	for i := range mat {
		mat[i] = float32(i%7) * 0.01
	}
	vec := make([]float32, rows)
	for i := range vec {
		vec[i] = float32(i%5) * 0.1
	}
	out := make([]float32, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := GemvF32Transposed(mat, rows, cols, vec, out); err != nil {
			b.Fatal(err)
		}
	}
}
