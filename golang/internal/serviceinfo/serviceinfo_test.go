package serviceinfo

import (
	"path/filepath"
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/internal/testutil"
)

func TestDefaultModelID(t *testing.T) {
	if got := DefaultModelID(nil); got != "oxidize-default" {
		t.Fatalf("default id = %q", got)
	}

	models := []ModelInfo{{ID: "valid-v3", Path: "/tmp/valid-v3.gguf"}}
	if got := DefaultModelID(models); got != "valid-v3" {
		t.Fatalf("model id = %q", got)
	}
}

func TestDiscoverModels(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "valid-v3.gguf")
	testutil.CopyFixture(t, path)

	models, err := DiscoverModels(dir)
	if err != nil {
		t.Fatalf("discover: %v", err)
	}
	if len(models) != 1 || models[0].ID != "valid-v3" {
		t.Fatalf("models = %#v", models)
	}
}

func TestOpenAPIContainsPublicRoutes(t *testing.T) {
	spec := OpenAPI("dev")
	for _, path := range []string{"/healthz", "/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/embeddings"} {
		if _, ok := spec.Paths[path]; !ok {
			t.Fatalf("missing path %q", path)
		}
	}
}

func TestMetricsSnapshot(t *testing.T) {
	text := MetricsSnapshot(MetricsData{RequestsTotal: 2, RequestsInflight: 1})
	if text == "" {
		t.Fatal("metrics text must not be empty")
	}
}
