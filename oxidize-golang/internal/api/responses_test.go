package api

import (
	"encoding/json"
	"testing"
)

func TestCandidateCountRejectsUnsupportedValues(t *testing.T) {
	two := 2

	errResp := ValidateCandidateCount(&two, nil)
	if errResp == nil {
		t.Fatal("expected validation error")
	}
	if errResp.StatusCode != 400 {
		t.Fatalf("status = %d", errResp.StatusCode)
	}
	if errResp.Error.Message != "oxidize-server currently supports only n=1 and best_of=1" {
		t.Fatalf("message = %q", errResp.Error.Message)
	}
}

func TestChatResponseFormatJsonObject(t *testing.T) {
	resp := BuildChatCompletion("demo", "{}")
	raw := mustJSON(t, resp)
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if decoded["object"] != "chat.completion" {
		t.Fatalf("object = %#v", decoded["object"])
	}
	assertJSONContains(t, raw, `"content":"{}"`)
}

func TestModelsResponseDefaultID(t *testing.T) {
	resp := BuildModelsResponse("oxidize-default")
	if len(resp.Data) != 1 || resp.Data[0].ID != "oxidize-default" {
		t.Fatalf("response = %#v", resp)
	}
}

func TestTextCompletionResponseShape(t *testing.T) {
	resp := BuildTextCompletion("demo", "hello")
	raw := mustJSON(t, resp)
	assertJSONContains(t, raw, `"object":"text_completion"`)
	assertJSONContains(t, raw, `"finish_reason":"stop"`)
	assertJSONContains(t, raw, `"completion_tokens":0`)
}

func TestChatChunkOmitsMessage(t *testing.T) {
	resp := BuildChatChunk("demo", "hello", false)
	raw := mustJSON(t, resp)
	if contains(string(raw), `"message"`) {
		t.Fatalf("chunk contains message field: %s", raw)
	}
	assertJSONContains(t, raw, `"delta":{"role":"assistant","content":"hello"}`)
}

func TestEmbeddingsResponseShape(t *testing.T) {
	resp := BuildEmbeddingsResponse("demo")
	raw := mustJSON(t, resp)
	assertJSONContains(t, raw, `"object":"list"`)
	assertJSONContains(t, raw, `"embedding":[0,0,0,0,0,0,0,0]`)
}

// TestEmbeddingsResponseAcceptsFloatValues guards against a regression where
// EmbeddingData.Embedding was typed as []int, which would cause a compile
// error when the OpenAI-shaped response is populated with float values.
func TestEmbeddingsResponseAcceptsFloatValues(t *testing.T) {
	resp := EmbeddingsResponse{
		Object: "list",
		Data:   []EmbeddingData{{Object: "embedding", Embedding: []float64{0.5, -1.25, 3.0}, Index: 0}},
		Model:  "demo",
	}
	raw := mustJSON(t, resp)
	assertJSONContains(t, raw, `"embedding":[0.5,-1.25,3]`)
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

func assertJSONContains(t *testing.T, raw []byte, needle string) {
	t.Helper()
	if !json.Valid(raw) {
		t.Fatalf("invalid json: %s", raw)
	}
	if string(raw) == "" || !contains(string(raw), needle) {
		t.Fatalf("json %s missing %q", raw, needle)
	}
}

func contains(haystack string, needle string) bool {
	return len(needle) == 0 || (len(haystack) >= len(needle) && indexOf(haystack, needle) >= 0)
}

func indexOf(haystack string, needle string) int {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}
