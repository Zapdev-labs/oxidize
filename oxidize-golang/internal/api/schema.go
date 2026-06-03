package api

import (
	"encoding/json"
	"fmt"
	"strings"
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
	Role    string         `json:"role"`
	Content MessageContent `json:"content"`
	Images  []string       `json:"images,omitempty"`
}

type MessageContent struct {
	text  string
	parts []ContentPart
	mode  messageContentMode
}

type messageContentMode uint8

const (
	messageContentText messageContentMode = iota
	messageContentParts
)

type ContentPart struct {
	Type     string           `json:"type"`
	Text     string           `json:"text,omitempty"`
	ImageURL *ContentImageURL `json:"image_url,omitempty"`
}

type ContentImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

func NewMessageContentText(text string) MessageContent {
	return MessageContent{text: text, mode: messageContentText}
}

func NewMessageContentParts(parts []ContentPart) MessageContent {
	return MessageContent{parts: append([]ContentPart(nil), parts...), mode: messageContentParts}
}

func (c MessageContent) Text() string {
	if c.mode != messageContentParts {
		return c.text
	}
	var b strings.Builder
	for _, part := range c.parts {
		if part.Type == "text" {
			b.WriteString(part.Text)
		}
	}
	return b.String()
}

func (c MessageContent) Parts() []ContentPart {
	return append([]ContentPart(nil), c.parts...)
}

func (c MessageContent) MarshalJSON() ([]byte, error) {
	if c.mode == messageContentParts {
		return json.Marshal(c.parts)
	}
	return json.Marshal(c.text)
}

func (c *MessageContent) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*c = NewMessageContentText(text)
		return nil
	}
	var parts []ContentPart
	if err := json.Unmarshal(data, &parts); err == nil {
		for _, part := range parts {
			switch part.Type {
			case "text":
			case "image_url":
				if part.ImageURL == nil || strings.TrimSpace(part.ImageURL.URL) == "" {
					return fmt.Errorf("decode content parts")
				}
			default:
				return fmt.Errorf("decode content parts")
			}
		}
		*c = NewMessageContentParts(parts)
		return nil
	}
	return fmt.Errorf("decode chat message content")
}

// MaxTokensOr returns max_tokens or max_completion_tokens, else defaultMax.
func (r ChatCompletionRequest) MaxTokensOr(defaultMax int) int {
	if r.MaxCompletionTokens != nil && *r.MaxCompletionTokens > 0 {
		return *r.MaxCompletionTokens
	}
	if r.MaxTokens != nil && *r.MaxTokens > 0 {
		return *r.MaxTokens
	}
	return defaultMax
}

// FirstUserMessage returns the last user message text in the chat.
func (r ChatCompletionRequest) FirstUserMessage() string {
	for i := len(r.Messages) - 1; i >= 0; i-- {
		if r.Messages[i].Role == "user" {
			return r.Messages[i].Content.Text()
		}
	}
	if len(r.Messages) > 0 {
		return r.Messages[len(r.Messages)-1].Content.Text()
	}
	return ""
}

// MaxTokensOr returns max_tokens or defaultMax when unset.
func (r CompletionRequest) MaxTokensOr(defaultMax int) int {
	if r.MaxTokens != nil && *r.MaxTokens > 0 {
		return *r.MaxTokens
	}
	return defaultMax
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
