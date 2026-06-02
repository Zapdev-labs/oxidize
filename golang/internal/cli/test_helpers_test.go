package cli

import (
	"os"
	"path/filepath"
	"testing"
)

func copyFixture(t *testing.T, target string) {
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
