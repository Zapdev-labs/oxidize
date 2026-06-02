package scripts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestManualQAScriptShape(t *testing.T) {
	path := filepath.Join("manual_qa.sh")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read script: %v", err)
	}
	text := string(raw)
	for _, needle := range []string{
		"trap cleanup EXIT",
		"--prompt",
		"list --models-dir",
		"/healthz",
		"/v1/models",
		"/v1/chat/completions",
		"/v1/completions",
		"/v1/embeddings",
		"[DONE]",
		"task-10-",
	} {
		if !strings.Contains(text, needle) {
			t.Fatalf("script missing %q", needle)
		}
	}
}
