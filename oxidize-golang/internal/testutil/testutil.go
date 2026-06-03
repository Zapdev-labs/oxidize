// Package testutil provides shared test helpers for the oxidize Go module.
package testutil

import (
	"os"
	"path/filepath"
	"testing"
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
