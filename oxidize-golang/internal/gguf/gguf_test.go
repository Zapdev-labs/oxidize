package gguf

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseFixture(t *testing.T) {
	raw := encodedFixture(t)
	file, err := Parse(raw)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if file.Version != 3 || file.TensorCount != 1 {
		t.Fatalf("file = %#v", file)
	}
	if file.Metadata["general.architecture"].String != "demo" {
		t.Fatalf("metadata = %#v", file.Metadata)
	}
	if len(file.TensorInfos) != 1 || file.TensorInfos[0].Name != "weight" {
		t.Fatalf("tensor infos = %#v", file.TensorInfos)
	}
}

func TestRejectsInvalidFixtures(t *testing.T) {
	tests := []struct {
		name string
		file string
		want string
	}{
		{name: "magic", file: "invalid-magic.gguf", want: "invalid gguf magic"},
		{name: "version", file: "unsupported-version.gguf", want: "unsupported gguf version: 1"},
		{name: "alignment", file: "invalid-alignment.gguf", want: "invalid alignment: 3"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := LoadFile(fixturePath(tt.file))
			if err == nil || err.Error() != tt.want {
				t.Fatalf("err = %v", err)
			}
		})
	}
}

func TestEncodeArrayMetadataRoundTrip(t *testing.T) {
	raw, err := Encode(WriterHeader{
		Version: 3,
		Metadata: map[string]MetadataValue{
			"tokenizer.ggml.tokens": {
				Type: MetadataArray,
				Array: []MetadataValue{
					{Type: MetadataString, String: "a"},
					{Type: MetadataString, String: "b"},
				},
			},
		},
		Tensors:   []TensorInfo{{Name: "weight", Dimensions: []uint64{1}, GGMLType: 0, RelativeOffset: 0}},
		Alignment: 32,
	}, []byte{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	file, err := Parse(raw)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	arr := file.Metadata["tokenizer.ggml.tokens"].Array
	if len(arr) != 2 || arr[0].String != "a" || arr[1].String != "b" {
		t.Fatalf("array = %#v", arr)
	}
}

func TestValidateFile(t *testing.T) {
	path := t.TempDir() + "/encoded.gguf"
	raw := encodedFixture(t)
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	if err := ValidateFile(path); err != nil {
		t.Fatalf("validate: %v", err)
	}
}

func TestLoadMetadata(t *testing.T) {
	path := t.TempDir() + "/encoded.gguf"
	raw := encodedFixture(t)
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	header, err := LoadMetadata(path)
	if err != nil {
		t.Fatalf("load metadata: %v", err)
	}
	if header.Version != 3 {
		t.Fatalf("version = %d", header.Version)
	}
	if got := header.Metadata["general.architecture"].String; got != "demo" {
		t.Fatalf("architecture = %q", got)
	}
}

func TestRejectsTruncatedTensorData(t *testing.T) {
	raw := encodedFixture(t)
	if _, err := Parse(raw[:len(raw)-1]); err == nil {
		t.Fatal("expected parse error")
	}
}

// TestAsUint64AcceptsZeroForSigned guards against a regression where a
// signed-typed metadata value of 0 was treated as "not convertible" and
// returned (_, false), even though 0 is a valid non-negative integer.
func TestAsUint64AcceptsZeroForSigned(t *testing.T) {
	for _, mt := range []MetadataType{MetadataInt8, MetadataInt16, MetadataInt32, MetadataInt64} {
		v := MetadataValue{Type: mt, Int64: 0}
		got, ok := v.AsUint64()
		if !ok || got != 0 {
			t.Fatalf("AsUint64 for %d(0) = (%d, %v), want (0, true)", mt, got, ok)
		}
	}
	// And confirm that negative values are still rejected.
	v := MetadataValue{Type: MetadataInt32, Int64: -1}
	if _, ok := v.AsUint64(); ok {
		t.Fatal("AsUint64 for negative int32 must be false")
	}
}

func fixturePath(name string) string {
	return filepath.Join("..", "..", "..", "oxidize-core", "tests", "fixtures", name)
}

func encodedFixture(t *testing.T) []byte {
	t.Helper()
	raw, err := Encode(WriterHeader{
		Version: 3,
		Metadata: map[string]MetadataValue{
			"general.architecture": {Type: MetadataString, String: "demo"},
		},
		Tensors:   []TensorInfo{{Name: "weight", Dimensions: []uint64{1}, GGMLType: 0, RelativeOffset: 0}},
		Alignment: 32,
	}, []byte{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return raw
}
