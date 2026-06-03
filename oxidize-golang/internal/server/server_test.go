package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/internal/testutil"
)

func TestHealthAndDocs(t *testing.T) {
	handler, err := NewHandler(Config{})
	if err != nil {
		t.Fatalf("handler: %v", err)
	}

	for _, path := range []string{"/healthz", "/livez", "/readyz", "/openapi.json", "/metrics"} {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s status = %d", path, rec.Code)
		}
	}
}

func TestModelsAndPlaceholderRoutes(t *testing.T) {
	dir := t.TempDir()
	testutil.CopyFixture(t, filepath.Join(dir, "valid-v3.gguf"))
	handler, err := NewHandler(Config{ModelsDir: dir})
	if err != nil {
		t.Fatalf("handler: %v", err)
	}

	assertStatus(t, handler, http.MethodGet, "/v1/models", nil, "", http.StatusOK)
	assertStatus(t, handler, http.MethodPost, "/v1/chat/completions", []byte(`{"model":"valid-v3","messages":[{"role":"user","content":"hi"}]}`), "application/json", http.StatusOK)
	assertStatus(t, handler, http.MethodPost, "/v1/completions", []byte(`{"model":"valid-v3","prompt":"hi"}`), "application/json", http.StatusOK)
	assertStatus(t, handler, http.MethodPost, "/v1/embeddings", []byte(`{"model":"valid-v3","input":"hi"}`), "application/json", http.StatusOK)
}

func TestAuthAndErrors(t *testing.T) {
	t.Setenv("OXIDIZE_API_KEY", "secret")
	handler, err := NewHandler(Config{})
	if err != nil {
		t.Fatalf("handler: %v", err)
	}

	assertStatus(t, handler, http.MethodGet, "/v1/models", nil, "", http.StatusUnauthorized)
	assertStatus(t, handler, http.MethodPost, "/v1/completions", []byte(`{"model":"demo","prompt":"hi","n":2}`), "application/json", http.StatusUnauthorized)

	req := httptest.NewRequest(http.MethodPost, "/v1/completions", bytes.NewReader([]byte(`{"model":"demo","prompt":"hi","n":2}`)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "secret")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestRejectsOversizedRequestBody(t *testing.T) {
	handler, err := NewHandler(Config{})
	if err != nil {
		t.Fatalf("handler: %v", err)
	}

	body := append([]byte(`{"model":"demo","prompt":"`), bytes.Repeat([]byte("a"), maxJSONBodyBytes)...)
	body = append(body, []byte(`"}`)...)
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
}

func TestStreamingRoute(t *testing.T) {
	handler, err := NewHandler(Config{})
	if err != nil {
		t.Fatalf("handler: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"model":"demo","messages":[{"role":"user","content":"hi"}],"stream":true}`)))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("content-type = %q", rec.Header().Get("Content-Type"))
	}
	if !bytes.Contains(rec.Body.Bytes(), []byte("[DONE]")) {
		t.Fatalf("body = %s", rec.Body.Bytes())
	}
}

func assertStatus(t *testing.T, handler http.Handler, method, path string, body []byte, contentType string, want int) {
	t.Helper()
	req := httptest.NewRequest(method, path, bytes.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != want {
		t.Fatalf("%s %s status = %d body = %s", method, path, rec.Code, rec.Body.Bytes())
	}
	if rec.Body.Len() > 0 && json.Valid(rec.Body.Bytes()) {
		return
	}
	if path == "/metrics" || path == "/healthz" || path == "/livez" || path == "/readyz" {
		return
	}
}
