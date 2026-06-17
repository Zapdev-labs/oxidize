package server

import (
	"net/http"
	"os"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/mesh"
	"github.com/Zapdev-labs/oxidize/golang/internal/api"
)

func (a *application) meshChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	var payload api.ChatCompletionRequest
	if !decodeJSON(w, r, &payload) {
		return
	}
	rt := a.meshRuntime()
	if rt == nil {
		writeJSON(w, http.StatusServiceUnavailable, api.ErrorResponse{
			StatusCode: http.StatusServiceUnavailable,
			Error: api.APIError{
				Message: "mesh runtime is not configured",
				Type:    "service_unavailable",
			},
		})
		return
	}
	if !a.ensureModel(w, payload.Model) {
		return
	}
	prompt := payload.FirstUserMessage()
	temp, topP, topK := samplingFromChat(payload)
	maxTok := payload.MaxTokensOr(a.defaultMaxTokens)
	text, err := rt.RouteCompletion(payload.Model, prompt, func(modelID, p string) (string, error) {
		out := a.completionText(r.Context(), modelID, p, maxTok, temp, topP, topK)
		return out, nil
	})
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, api.ErrorResponse{
			StatusCode: http.StatusServiceUnavailable,
			Error:      api.APIError{Message: err.Error(), Type: "service_unavailable"},
		})
		return
	}
	if text == "" {
		text = prompt
	}
	writeJSON(w, http.StatusOK, api.BuildChatCompletion(payload.Model, text))
}

func (a *application) meshRuntime() *mesh.Runtime {
	addr := strings.TrimSpace(os.Getenv("OXIDIZE_MESH_ADDR"))
	if addr == "" {
		return nil
	}
	local := mesh.MeshNode{ID: "local", Addr: addr, Role: "worker", Healthy: true}
	rt := mesh.NewRuntime(local)
	_ = rt.StartListen()
	if peers := strings.TrimSpace(os.Getenv("OXIDIZE_MESH_PEERS")); peers != "" {
		for _, p := range strings.Split(peers, ",") {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			rt.Engine.Router.Update(mesh.MeshNode{ID: p, Addr: p, Role: "worker", Healthy: true})
		}
	}
	return rt
}
