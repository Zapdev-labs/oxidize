package generate

import "github.com/Zapdev-labs/oxidize/golang/internal/api"

type PlaceholderSpec struct {
	ResponseFormat *api.ResponseFormat
	GuidedJSON     []byte
	JSONSchema     []byte
	GuidedRegex    string
	GuidedChoice   []string
}

func PlaceholderText(spec PlaceholderSpec) string {
	if len(spec.GuidedChoice) > 0 {
		return spec.GuidedChoice[0]
	}
	if len(spec.GuidedJSON) > 0 || len(spec.JSONSchema) > 0 {
		return "{}"
	}
	if spec.GuidedRegex != "" {
		return spec.GuidedRegex
	}
	if spec.ResponseFormat != nil {
		return spec.ResponseFormat.OutputText()
	}
	return ""
}

func CLIText(prompt string) string {
	return "oxidize-cli: " + prompt
}

func CLITranscript(prompt string) string {
	return "generation progress: 1/2 tokens\ngeneration progress: 2/2 tokens\n" + CLIText(prompt) + "\ngeneration stats: tokens=2 speed=4.00 tok/s\n"
}
