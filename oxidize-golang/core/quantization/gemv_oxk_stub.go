//go:build !cgo || !oxk

package quantization

// GemvOxk is unavailable without CGO and the oxk build tag.
func GemvOxk(qbytes []byte, qtype Type, rows, cols int, vector, output []float32) (bool, error) {
	return false, nil
}

// OxkHasAVX2 reports false when OXK is not linked.
func OxkHasAVX2() bool {
	return false
}

// OxkDotF32 falls back to a scalar dot product.
func OxkDotF32(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}
