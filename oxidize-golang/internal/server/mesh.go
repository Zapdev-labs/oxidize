package server

import (
	"net/http"

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
	writeJSON(w, http.StatusServiceUnavailable, api.ErrorResponse{
		StatusCode: http.StatusServiceUnavailable,
		Error: api.APIError{
			Message: "mesh runtime is not configured",
			Type:    "service_unavailable",
		},
	})
}
