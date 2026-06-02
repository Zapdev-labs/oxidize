package api

import (
	"encoding/json"
	"testing"
)

func TestChatRequestAcceptsStructuredFields(t *testing.T) {
	payload := []byte(`{
		"model":"demo",
		"messages":[{"role":"user","content":"hello"}],
		"response_format":{"type":"json_object"},
		"guided_json":{"type":"object"},
		"json_schema":{"type":"object"},
		"guided_regex":"[a-z]+",
		"guided_choice":["alpha","beta"],
		"stream":true,
		"max_tokens":4,
		"max_completion_tokens":5,
		"temperature":0.2,
		"top_p":0.9,
		"top_k":8,
		"min_p":0.1,
		"typical_p":0.8,
		"tail_free_z":0.5,
		"stop":["done"],
		"seed":7,
		"n":1,
		"best_of":1
	}`)

	var req ChatCompletionRequest
	if err := json.Unmarshal(payload, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if req.ResponseFormat == nil || req.ResponseFormat.Type != ResponseFormatJSONObject {
		t.Fatalf("response format = %#v", req.ResponseFormat)
	}
	if len(req.GuidedChoice) != 2 || req.GuidedChoice[0] != "alpha" {
		t.Fatalf("guided choice = %#v", req.GuidedChoice)
	}
	if got := req.Stop.Values(); len(got) != 1 || got[0] != "done" {
		t.Fatalf("stop = %#v", got)
	}
}

func TestCompletionRequestAcceptsStringStop(t *testing.T) {
	payload := []byte(`{"model":"demo","prompt":"hello","stop":"done"}`)

	var req CompletionRequest
	if err := json.Unmarshal(payload, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got := req.Stop.Values()
	if len(got) != 1 || got[0] != "done" {
		t.Fatalf("stop = %#v", got)
	}
}
