package serviceinfo

import (
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/internal/testutil"
)

func TestDefaultModelID(t *testing.T) {
	if got := DefaultModelID(nil); got != "oxidize-default" {
		t.Fatalf("default id = %q", got)
	}

	models := []ModelInfo{{ID: testutil.QwenModelID, Path: "/tmp/" + testutil.QwenModelFileName}}
	if got := DefaultModelID(models); got != testutil.QwenModelID {
		t.Fatalf("model id = %q", got)
	}
}

func TestDiscoverModels(t *testing.T) {
	dir := t.TempDir()
	testutil.LinkQwenModel(t, dir)

	models, err := DiscoverModels(dir)
	if err != nil {
		t.Fatalf("discover: %v", err)
	}
	if len(models) != 1 || models[0].ID != testutil.QwenModelID {
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
