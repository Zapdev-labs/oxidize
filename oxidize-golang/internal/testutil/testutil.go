// Package testutil provides shared test helpers for the oxidize Go module.
package testutil

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/internal/gguf"
)

const (
	// QwenModelFileName is the canonical integration-test GGUF in oxidize/models/.
	QwenModelFileName = "Qwen3-4B-Q4_K_M.gguf"
	// QwenModelID is the model id exposed by list/serve (filename without .gguf).
	QwenModelID = "Qwen3-4B-Q4_K_M"
)

// OxidizeRepoRoot walks upward from the working directory until it finds the
// oxidize workspace root (oxidize-core + models/).
func OxidizeRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if stat, err := os.Stat(filepath.Join(dir, "oxidize-core")); err == nil && stat.IsDir() {
			if models, err := os.Stat(filepath.Join(dir, "models")); err == nil && models.IsDir() {
				return dir
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skip("oxidize repo root not found")
		}
		dir = parent
	}
}

// SlowTestsEnabled reports whether full-model prompt integration should run.
func SlowTestsEnabled() bool {
	return os.Getenv("OXIDIZE_SLOW_TESTS") != ""
}

// RequireSlowTests skips unless OXIDIZE_SLOW_TESTS is set (Qwen3-4B CPU inference is slow).
func RequireSlowTests(t *testing.T) {
	t.Helper()
	if !SlowTestsEnabled() {
		t.Skip("set OXIDIZE_SLOW_TESTS=1 to run full-model prompt integration")
	}
}

// QwenModelPath returns the path to models/Qwen3-4B-Q4_K_M.gguf, skipping when absent.
func QwenModelPath(t *testing.T) string {
	t.Helper()
	path := filepath.Join(OxidizeRepoRoot(t), "models", QwenModelFileName)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("qwen model not found at %s", path)
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		t.Fatalf("abs qwen model path: %v", err)
	}
	return abs
}

// LinkQwenModel symlinks the canonical Qwen GGUF into targetDir for list/serve tests.
func LinkQwenModel(t *testing.T, targetDir string) string {
	t.Helper()
	src := QwenModelPath(t)
	dst := filepath.Join(targetDir, QwenModelFileName)
	if err := os.Symlink(src, dst); err != nil {
		t.Fatalf("symlink qwen model: %v", err)
	}
	return dst
}

// AssertGenerationText checks CLI output contains non-degenerate generated text.
func AssertGenerationText(t *testing.T, raw string) {
	t.Helper()
	text := strings.TrimSpace(raw)
	if idx := strings.Index(text, "generation stats:"); idx >= 0 {
		text = strings.TrimSpace(text[:idx])
	}
	if text == "" {
		t.Fatalf("expected generated text, got %q", raw)
	}
	if strings.Contains(text, "DateFormat") {
		t.Fatalf("degenerate output: %q", text)
	}
}

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
