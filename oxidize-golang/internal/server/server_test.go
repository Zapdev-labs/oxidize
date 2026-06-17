package server

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
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
	testutil.LinkQwenModel(t, dir)
	modelID := testutil.QwenModelID
	handler, err := NewHandler(Config{ModelsDir: dir})
	if err != nil {
		t.Fatalf("handler: %v", err)
	}

	assertStatus(t, handler, http.MethodGet, "/v1/models", nil, "", http.StatusOK)
	assertStatus(t, handler, http.MethodPost, "/v1/chat/completions", []byte(`{"model":"`+modelID+`","messages":[{"role":"user","content":"hi"}]}`), "application/json", http.StatusOK)
	assertStatus(t, handler, http.MethodPost, "/v1/completions", []byte(`{"model":"`+modelID+`","prompt":"hi"}`), "application/json", http.StatusOK)
	assertStatus(t, handler, http.MethodPost, "/v1/embeddings", []byte(`{"model":"`+modelID+`","input":"hi"}`), "application/json", http.StatusNotImplemented)
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

func TestMeshChatDisabledRoute(t *testing.T) {
	handler, err := NewHandler(Config{})
	if err != nil {
		t.Fatalf("handler: %v", err)
	}
	assertStatus(t, handler, http.MethodPost, "/v1/mesh/chat/completions", []byte(`{"model":"demo","messages":[{"role":"user","content":"hi"}]}`), "application/json", http.StatusServiceUnavailable)
}

func TestRealtimeLifecycle(t *testing.T) {
	handler, err := NewHandler(Config{})
	if err != nil {
		t.Fatalf("handler: %v", err)
	}
	server := httptest.NewServer(handler)
	defer server.Close()

	conn, rw := dialWebSocket(t, server.URL, "/v1/realtime")
	defer conn.Close()
	first := readWSJSON(t, rw)
	if first["type"] != "session.created" {
		t.Fatalf("first event = %#v", first)
	}
	writeWSJSON(t, conn, map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{
			"type":    "message",
			"role":    "user",
			"content": []map[string]any{{"type": "input_text", "text": "hi"}},
		},
	})
	created := readWSJSON(t, rw)
	if created["type"] != "conversation.item.created" {
		t.Fatalf("created event = %#v", created)
	}
	writeWSJSON(t, conn, map[string]any{"type": "response.create"})
	responseCreated := readWSJSON(t, rw)
	if responseCreated["type"] != "response.created" {
		t.Fatalf("response event = %#v", responseCreated)
	}
	errEvent := readWSJSON(t, rw)
	if errEvent["type"] != "error" {
		t.Fatalf("error event = %#v", errEvent)
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

func dialWebSocket(t *testing.T, rawURL, path string) (net.Conn, *bufio.Reader) {
	t.Helper()
	addr := strings.TrimPrefix(rawURL, "http://")
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	keyBytes := make([]byte, 16)
	if _, err := rand.Read(keyBytes); err != nil {
		t.Fatalf("rand: %v", err)
	}
	key := fmt.Sprintf("%x", keyBytes)
	req := fmt.Sprintf("GET %s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: %s\r\n\r\n", path, addr, key)
	if _, err := conn.Write([]byte(req)); err != nil {
		t.Fatalf("write handshake: %v", err)
	}
	rw := bufio.NewReader(conn)
	status, err := rw.ReadString('\n')
	if err != nil {
		t.Fatalf("read status: %v", err)
	}
	if !strings.Contains(status, "101") {
		t.Fatalf("status = %s", status)
	}
	for {
		line, err := rw.ReadString('\n')
		if err != nil {
			t.Fatalf("read header: %v", err)
		}
		if line == "\r\n" {
			break
		}
	}
	return conn, rw
}

func readWSJSON(t *testing.T, r *bufio.Reader) map[string]any {
	t.Helper()
	header := make([]byte, 2)
	if _, err := io.ReadFull(r, header); err != nil {
		t.Fatalf("read ws header: %v", err)
	}
	length := int(header[1] & 0x7f)
	if length == 126 {
		ext := make([]byte, 2)
		_, _ = io.ReadFull(r, ext)
		length = int(ext[0])<<8 | int(ext[1])
	}
	payload := make([]byte, length)
	if _, err := io.ReadFull(r, payload); err != nil {
		t.Fatalf("read ws payload: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(payload, &out); err != nil {
		t.Fatalf("json: %v payload=%s", err, payload)
	}
	return out
}

func writeWSJSON(t *testing.T, conn net.Conn, v any) {
	t.Helper()
	payload, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	mask := []byte{1, 2, 3, 4}
	frame := []byte{0x81, 0x80 | byte(len(payload))}
	frame = append(frame, mask...)
	for i, b := range payload {
		frame = append(frame, b^mask[i%4])
	}
	if _, err := conn.Write(frame); err != nil {
		t.Fatalf("write frame: %v", err)
	}
}
