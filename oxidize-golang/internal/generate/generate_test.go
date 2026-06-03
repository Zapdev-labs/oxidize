package generate

import (
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/internal/api"
)

func TestServerPlaceholderPriority(t *testing.T) {
	format := &api.ResponseFormat{Type: api.ResponseFormatJSONObject}
	spec := PlaceholderSpec{
		ResponseFormat: format,
		GuidedJSON:     []byte(`{"type":"object"}`),
		JSONSchema:     []byte(`{"type":"object"}`),
		GuidedRegex:    "[a-z]+",
		GuidedChoice:   []string{"first", "second"},
	}

	if got := PlaceholderText(spec); got != "first" {
		t.Fatalf("placeholder = %q", got)
	}
}

func TestCLITranscript(t *testing.T) {
	got := CLITranscript("hello")
	want := "generation progress: 1/2 tokens\ngeneration progress: 2/2 tokens\noxidize-cli: hello\ngeneration stats: tokens=2 speed=4.00 tok/s\n"
	if got != want {
		t.Fatalf("transcript = %q", got)
	}
}

func TestNoRealInferenceHooks(t *testing.T) {
	spec := PlaceholderSpec{GuidedRegex: "[0-9]+"}
	if first, second := PlaceholderText(spec), PlaceholderText(spec); first != second {
		t.Fatalf("non-deterministic output: %q %q", first, second)
	}
}
