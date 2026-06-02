package gguf

import (
	"path/filepath"
	"testing"
)

func TestParseFixture(t *testing.T) {
	path := fixturePath("valid-v3.gguf")
	file, err := LoadFile(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if file.Version != 3 || file.TensorCount != 1 {
		t.Fatalf("file = %#v", file)
	}
	if file.Alignment != 64 || file.DataSectionStart != 128 {
		t.Fatalf("alignment/data = %#v", file)
	}
	if len(file.TensorInfos) != 1 || file.TensorInfos[0].Name != "tok_embeddings.weight" {
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

func fixturePath(name string) string {
	return filepath.Join("..", "..", "..", "oxidize-core", "tests", "fixtures", name)
}
