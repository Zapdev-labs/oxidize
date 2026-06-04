package server

import (
	"net/http"

	"github.com/Zapdev-labs/oxidize/golang/internal/api"
	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
)

func (a *application) streamChatCompletion(
	w http.ResponseWriter,
	r *http.Request,
	modelID, prompt string,
	maxTokens int,
	temp, topP float32,
	topK int,
	payload api.ChatCompletionRequest,
) {
	path := a.modelPath(modelID)
	if path == "" {
		path = a.defaultModelPath
	}
	if path == "" {
		text := generate.PlaceholderText(generate.PlaceholderSpec{
			ResponseFormat: payload.ResponseFormat,
			GuidedJSON:     payload.GuidedJSON,
			JSONSchema:     payload.JSONSchema,
			GuidedRegex:    payload.GuidedRegex,
			GuidedChoice:   payload.GuidedChoice,
		})
		writeSSE(w, api.BuildChatChunk(modelID, text, false), api.BuildChatChunk(modelID, "", true))
		return
	}
	params := generate.CompletionParams{
		MaxTokens:   maxTokens,
		Temperature: temp,
		TopP:        topP,
		TopK:        topK,
		Loader:      a.loader,
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		text, _ := generate.StreamCompletion(r.Context(), path, prompt, params, nil)
		if text == "" {
			text = generate.PlaceholderText(generate.PlaceholderSpec{})
		}
		writeSSE(w, api.BuildChatChunk(modelID, text, false), api.BuildChatChunk(modelID, "", true))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	first := true
	_, _ = generate.StreamCompletion(r.Context(), path, prompt, params, func(piece string) error {
		if piece == "" {
			return nil
		}
		chunk := api.BuildChatChunk(modelID, piece, false)
		writeSSEChunk(w, chunk)
		if first {
			first = false
			flusher.Flush()
		}
		return nil
	})
	writeSSEChunk(w, api.BuildChatChunk(modelID, "", true))
	flusher.Flush()
}

func (a *application) streamTextCompletion(
	w http.ResponseWriter,
	r *http.Request,
	modelID, prompt string,
	maxTokens int,
	temp, topP float32,
	topK int,
	payload api.CompletionRequest,
) {
	path := a.modelPath(modelID)
	if path == "" {
		path = a.defaultModelPath
	}
	params := generate.CompletionParams{
		MaxTokens:   maxTokens,
		Temperature: temp,
		TopP:        topP,
		TopK:        topK,
		Loader:      a.loader,
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		text, _ := generate.StreamCompletion(r.Context(), path, prompt, params, nil)
		if text == "" {
			text = generate.PlaceholderText(generate.PlaceholderSpec{})
		}
		writeSSE(w, api.BuildTextChunk(modelID, text, false), api.BuildTextChunk(modelID, "", true))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	_, _ = generate.StreamCompletion(r.Context(), path, prompt, params, func(piece string) error {
		if piece == "" {
			return nil
		}
		writeSSEChunk(w, api.BuildTextChunk(modelID, piece, false))
		flusher.Flush()
		return nil
	})
	writeSSEChunk(w, api.BuildTextChunk(modelID, "", true))
	flusher.Flush()
}

func writeSSEChunk(w http.ResponseWriter, payload any) {
	writeSSE(w, payload)
}
