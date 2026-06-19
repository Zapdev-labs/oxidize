package tensor

import (
	"math"
	"testing"
)

// naiveTransposed is the reference definition of GemvF32Transposed:
// output[c] = sum_r matrix[r*cols+c] * vector[r].
func naiveTransposed(matrix []float32, rows, cols int, vector []float32) []float32 {
	out := make([]float32, cols)
	for c := 0; c < cols; c++ {
		var sum float32
		for r := 0; r < rows; r++ {
			sum += matrix[r*cols+c] * vector[r]
		}
		out[c] = sum
	}
	return out
}

// TestGemvF32TransposedMatchesReference guards the cache-friendly refactor:
// the optimized kernel must produce identical results to the naive definition
// across odd shapes (including cols not divisible by the worker count / 64).
func TestGemvF32TransposedMatchesReference(t *testing.T) {
	shapes := [][2]int{
		{1, 1}, {3, 5}, {7, 64}, {64, 65}, {128, 896}, {896, 151}, {251, 4864}, {333, 257},
	}
	for _, s := range shapes {
		rows, cols := s[0], s[1]
		mat := make([]float32, rows*cols)
		for i := range mat {
			mat[i] = float32((i*13)%17) * 0.013
		}
		vec := make([]float32, rows)
		for i := range vec {
			vec[i] = float32((i*7)%11) * 0.07
		}
		got := make([]float32, cols)
		if err := GemvF32Transposed(mat, rows, cols, vec, got); err != nil {
			t.Fatalf("rows=%d cols=%d: %v", rows, cols, err)
		}
		want := naiveTransposed(mat, rows, cols, vec)
		for c := 0; c < cols; c++ {
			if math.Abs(float64(got[c]-want[c])) > 1e-4 {
				t.Fatalf("rows=%d cols=%d: mismatch at c=%d got=%g want=%g", rows, cols, c, got[c], want[c])
			}
		}
	}
}

// TestGemvF32TransposedAccumulatesFresh verifies the kernel does not depend on
// the caller pre-zeroing the output buffer (callers reuse buffers across tokens).
func TestGemvF32TransposedAccumulatesFresh(t *testing.T) {
	rows, cols := 16, 40
	mat := make([]float32, rows*cols)
	for i := range mat {
		mat[i] = float32(i%9) * 0.1
	}
	vec := make([]float32, rows)
	for i := range vec {
		vec[i] = 1
	}
	out := make([]float32, cols)
	for i := range out {
		out[i] = 999 // stale garbage
	}
	if err := GemvF32Transposed(mat, rows, cols, vec, out); err != nil {
		t.Fatal(err)
	}
	want := naiveTransposed(mat, rows, cols, vec)
	for c := 0; c < cols; c++ {
		if math.Abs(float64(out[c]-want[c])) > 1e-4 {
			t.Fatalf("stale buffer not overwritten at c=%d got=%g want=%g", c, out[c], want[c])
		}
	}
}
