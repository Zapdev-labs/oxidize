package server

import (
	"context"

	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
)

func (a *application) completionText(ctx context.Context, modelID, prompt string, maxTokens int, temperature, topP float32, topK int) string {
	path := a.modelPath(modelID)
	if path == "" {
		path = a.defaultModelPath
	}
	if path == "" {
		return ""
	}
	if maxTokens <= 0 {
		maxTokens = a.defaultMaxTokens
	}
	if temperature < 0 {
		temperature = a.defaultTemp
	}
	if topP < 0 {
		topP = a.defaultTopP
	}
	if topK == 0 {
		topK = a.defaultTopK
	}
	params := generate.CompletionParams{
		MaxTokens:   maxTokens,
		Temperature: temperature,
		TopP:        topP,
		TopK:        topK,
		Loader:      a.loader,
	}
	text, err := generate.CompletionText(ctx, path, prompt, params)
	if err != nil {
		return ""
	}
	return text
}

func (a *application) modelPath(modelID string) string {
	if modelID == "" {
		return a.defaultModelPath
	}
	for _, model := range a.models {
		if model.ID == modelID {
			return model.Path
		}
	}
	return ""
}
