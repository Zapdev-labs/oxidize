//go:build !cgo

package quantization

import "errors"

// RustModel is unavailable without CGO.
type RustModel struct{}

func (r *RustModel) Close()                          {}
func (r *RustModel) ResetSession()                     {}
func (r *RustModel) Forward([]uint32) ([]float32, error) { return nil, errors.New("rust ffi unavailable") }
func (r *RustModel) SampleArgmax() uint32              { return 0 }

// LoadRustModel returns an error when CGO is disabled.
func LoadRustModel(string) (*RustModel, error) {
	return nil, errors.New("rust ffi unavailable without cgo")
}
