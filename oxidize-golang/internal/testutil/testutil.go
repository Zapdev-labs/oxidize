// Package testutil provides shared test helpers for the oxidize Go module.
package testutil

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/internal/gguf"
)

// CopyFixture copies the canonical valid-v3.gguf fixture from the
// oxidize-core test fixtures into the supplied target path. It fails the test
// immediately if either the source cannot be read or the target cannot be
// written.
func CopyFixture(t *testing.T, target string) {
	t.Helper()
	source := filepath.Join("..", "..", "..", "oxidize-core", "tests", "fixtures", "valid-v3.gguf")
	raw, err := os.ReadFile(source)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	if err := os.WriteFile(target, raw, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
}

// WriteEncodedGGUF writes a minimal valid GGUF file suitable for parse/inspect tests.
func WriteEncodedGGUF(t *testing.T, target string) {
	t.Helper()
	raw, err := gguf.Encode(gguf.WriterHeader{
		Version: 3,
		Metadata: map[string]gguf.MetadataValue{
			"general.architecture": {Type: gguf.MetadataString, String: "demo"},
		},
		Tensors:   []gguf.TensorInfo{{Name: "weight", Dimensions: []uint64{1}, GGMLType: 0, RelativeOffset: 0}},
		Alignment: 32,
	}, []byte{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("encode gguf: %v", err)
	}
	if err := os.WriteFile(target, raw, 0o644); err != nil {
		t.Fatalf("write encoded gguf: %v", err)
	}
}
