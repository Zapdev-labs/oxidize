//go:build !cgo || !rust_ffi

package quantization

// GemvRust is the fallback used when the project is built without the
func GemvRust(_ []byte, _ Type, _, _ int, _, _ []float32) (bool, error) {
	return false, nil
}
