package auth

import (
	"crypto/subtle"
	"encoding/json"
	"net/http"
	"os"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/internal/api"
)

func Middleware(next http.Handler) http.Handler {
	expected := strings.TrimSpace(os.Getenv("OXIDIZE_API_KEY"))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/v1/") || expected == "" || hasAPIKey(r, expected) {
			next.ServeHTTP(w, r)
			return
		}
		writeJSON(w, api.InvalidAPIKey())
	})
}

func hasAPIKey(r *http.Request, expected string) bool {
	if constantTimeEqual(r.Header.Get("x-api-key"), expected) {
		return true
	}
	header := r.Header.Get("Authorization")
	if rest, ok := strings.CutPrefix(header, "Bearer "); ok {
		return constantTimeEqual(rest, expected)
	}
	if key := r.URL.Query().Get("api_key"); key != "" {
		return constantTimeEqual(key, expected)
	}
	return false
}

func constantTimeEqual(actual string, expected string) bool {
	if len(actual) != len(expected) {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(actual), []byte(expected)) == 1
}

func writeJSON(w http.ResponseWriter, resp api.ErrorResponse) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_ = json.NewEncoder(w).Encode(resp)
}
