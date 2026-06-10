package testutil

import (
	"os"

	"github.com/Zapdev-labs/oxidize/golang/internal/gguf"
)

// WriteValidFixture writes a minimal valid GGUF file to path.
func WriteValidFixture(path string) error {
	raw, err := gguf.Encode(gguf.WriterHeader{
		Version: 3,
		Metadata: map[string]gguf.MetadataValue{
			"general.architecture": {Type: gguf.MetadataString, String: "unknown"},
		},
		Tensors:   []gguf.TensorInfo{{Name: "weight", Dimensions: []uint64{1}, GGMLType: 0, RelativeOffset: 0}},
		Alignment: 32,
	}, []byte{1, 2, 3, 4})
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}
