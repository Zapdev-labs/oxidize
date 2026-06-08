//go:build !cgo || !rust_ffi

package quantization

// GemvRust is the fallback used when the project is built without the
// `cgo,rust_ffi` tags (i.e. no Rust FFI kernel is linked). It reports
// ok=false so callers transparently fall back to the pure-Go GEMV path.
//
// This mirrors the Rust workspace's stub convention (e.g. vulkan_stub.rs):
// the real implementation lives in gemv_rust.go behind `+build cgo,rust_ffi`.
func GemvRust(_ []byte, _ Type, _, _ int, _, _ []float32) (bool, error) {
	return false, nil
}
