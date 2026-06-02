package server

import (
	"encoding/json"
	"net/http"

	"github.com/Zapdev-labs/oxidize/golang/internal/api"
	"github.com/Zapdev-labs/oxidize/golang/internal/buildinfo"
	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
	"github.com/Zapdev-labs/oxidize/golang/internal/serviceinfo"
)

func (a *application) health(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (a *application) openapi(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, serviceinfo.OpenAPI(buildinfo.Version))
}

func (a *application) metricsHandler(w http.ResponseWriter, _ *http.Request) {
	a.mu.Lock()
	snapshot := a.metrics
	a.mu.Unlock()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = w.Write([]byte(serviceinfo.MetricsSnapshot(snapshot)))
}

func (a *application) modelsHandler(w http.ResponseWriter, _ *http.Request) {
	modelIDs := make([]string, 0, len(a.models))
	for _, model := range a.models {
		modelIDs = append(modelIDs, model.ID)
	}
	writeJSON(w, http.StatusOK, api.BuildModelsResponse(modelIDs...))
}

func (a *application) chatCompletions(w http.ResponseWriter, r *http.Request) {
	var payload api.ChatCompletionRequest
	if !decodeJSON(w, r, &payload) {
		return
	}
	if errResp := api.ValidateCandidateCount(payload.N, payload.BestOf); errResp != nil {
		writeJSON(w, errResp.StatusCode, errResp)
		return
	}
	if !a.ensureModel(w, payload.Model) {
		return
	}
	text := generate.PlaceholderText(generate.PlaceholderSpec{ResponseFormat: payload.ResponseFormat, GuidedJSON: payload.GuidedJSON, JSONSchema: payload.JSONSchema, GuidedRegex: payload.GuidedRegex, GuidedChoice: payload.GuidedChoice})
	if payload.Stream {
		writeSSE(w, api.BuildChatChunk(payload.Model, text, false), api.BuildChatChunk(payload.Model, "", true))
		return
	}
	writeJSON(w, http.StatusOK, api.BuildChatCompletion(payload.Model, text))
}

func (a *application) completions(w http.ResponseWriter, r *http.Request) {
	var payload api.CompletionRequest
	if !decodeJSON(w, r, &payload) {
		return
	}
	if errResp := api.ValidateCandidateCount(payload.N, payload.BestOf); errResp != nil {
		writeJSON(w, errResp.StatusCode, errResp)
		return
	}
	if !a.ensureModel(w, payload.Model) {
		return
	}
	text := generate.PlaceholderText(generate.PlaceholderSpec{ResponseFormat: payload.ResponseFormat, GuidedJSON: payload.GuidedJSON, JSONSchema: payload.JSONSchema, GuidedRegex: payload.GuidedRegex, GuidedChoice: payload.GuidedChoice})
	if payload.Stream {
		writeSSE(w, api.BuildTextChunk(payload.Model, text, false), api.BuildTextChunk(payload.Model, text, true))
		return
	}
	writeJSON(w, http.StatusOK, api.BuildTextCompletion(payload.Model, text))
}

func (a *application) embeddings(w http.ResponseWriter, r *http.Request) {
	var payload api.EmbeddingsRequest
	if !decodeJSON(w, r, &payload) {
		return
	}
	if !a.ensureModel(w, payload.Model) {
		return
	}
	writeJSON(w, http.StatusOK, api.BuildEmbeddingsResponse(payload.Model))
}

func (a *application) ensureModel(w http.ResponseWriter, model string) bool {
	if len(a.modelID) == 0 {
		return true
	}
	if _, ok := a.modelID[model]; ok {
		return true
	}
	resp := api.ModelNotFound(model)
	writeJSON(w, resp.StatusCode, resp)
	return false
}

func decodeJSON(w http.ResponseWriter, r *http.Request, target any) bool {
	if err := json.NewDecoder(r.Body).Decode(target); err != nil {
		resp := api.MalformedJSON()
		writeJSON(w, resp.StatusCode, resp)
		return false
	}
	return true
}
