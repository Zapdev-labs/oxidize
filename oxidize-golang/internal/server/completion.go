package server

import (
	"context"

	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
)

func (a *application) completionText(ctx context.Context, modelID, prompt string, maxTokens int) string {
	path := a.modelPath(modelID)
	if path == "" {
		return ""
	}
	text, err := generate.CompletionText(ctx, path, prompt, maxTokens)
	if err != nil {
		return ""
	}
	return text
}

func (a *application) modelPath(modelID string) string {
	for _, model := range a.models {
		if model.ID == modelID {
			return model.Path
		}
	}
	return ""
}
