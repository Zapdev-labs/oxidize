package api

import (
	"encoding/json"
	"fmt"
)

type ResponseFormatType string

const (
	ResponseFormatText       ResponseFormatType = "text"
	ResponseFormatJSONObject ResponseFormatType = "json_object"
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

type ChatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []ChatMessage   `json:"messages"`
	ResponseFormat      *ResponseFormat `json:"response_format,omitempty"`
	GuidedJSON          json.RawMessage `json:"guided_json,omitempty"`
	JSONSchema          json.RawMessage `json:"json_schema,omitempty"`
	GuidedRegex         string          `json:"guided_regex,omitempty"`
	GuidedChoice        []string        `json:"guided_choice,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	TopK                *int            `json:"top_k,omitempty"`
	MinP                *float64        `json:"min_p,omitempty"`
	TypicalP            *float64        `json:"typical_p,omitempty"`
	TailFreeZ           *float64        `json:"tail_free_z,omitempty"`
	Stop                StopSequences   `json:"stop,omitempty"`
	Seed                *uint64         `json:"seed,omitempty"`
	N                   *int            `json:"n,omitempty"`
	BestOf              *int            `json:"best_of,omitempty"`
}

type ChatMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

type CompletionRequest struct {
	Model          string          `json:"model"`
	Prompt         string          `json:"prompt"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	GuidedJSON     json.RawMessage `json:"guided_json,omitempty"`
	JSONSchema     json.RawMessage `json:"json_schema,omitempty"`
	GuidedRegex    string          `json:"guided_regex,omitempty"`
	GuidedChoice   []string        `json:"guided_choice,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	TopK           *int            `json:"top_k,omitempty"`
	MinP           *float64        `json:"min_p,omitempty"`
	TypicalP       *float64        `json:"typical_p,omitempty"`
	TailFreeZ      *float64        `json:"tail_free_z,omitempty"`
	Stop           StopSequences   `json:"stop,omitempty"`
	Seed           *uint64         `json:"seed,omitempty"`
	Echo           bool            `json:"echo,omitempty"`
	N              *int            `json:"n,omitempty"`
	BestOf         *int            `json:"best_of,omitempty"`
}

type EmbeddingsRequest struct {
	Model string          `json:"model"`
	Input json.RawMessage `json:"input"`
}

type ResponseFormat struct {
	Type       ResponseFormatType `json:"type"`
	JSONSchema json.RawMessage    `json:"json_schema,omitempty"`
}

func (f ResponseFormat) OutputText() string {
	switch f.Type {
	case ResponseFormatJSONObject, ResponseFormatJSONSchema:
		return "{}"
	default:
		return ""
	}
}

type StopSequences struct {
	values []string
}

func (s *StopSequences) UnmarshalJSON(data []byte) error {
	var single string
	if err := json.Unmarshal(data, &single); err == nil {
		s.values = []string{single}
		return nil
	}
	var many []string
	if err := json.Unmarshal(data, &many); err == nil {
		s.values = append([]string(nil), many...)
		return nil
	}
	return fmt.Errorf("decode stop sequences")
}

func (s StopSequences) Values() []string {
	return append([]string(nil), s.values...)
}
