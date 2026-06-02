package api

type ErrorResponse struct {
	StatusCode int      `json:"-"`
	Error      APIError `json:"error"`
}

type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens"`
}

type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelData `json:"data"`
}

type ModelData struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}

type ChatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   Usage        `json:"usage"`
}

type ChatChoice struct {
	Index        int          `json:"index"`
	Message      *ChatMessage `json:"message,omitempty"`
	Delta        *ChatDelta   `json:"delta,omitempty"`
	FinishReason *string      `json:"finish_reason"`
}

type ChatDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type TextCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []TextChoice `json:"choices"`
	Usage   Usage        `json:"usage"`
}

type TextChoice struct {
	Index        int     `json:"index"`
	Text         string  `json:"text"`
	FinishReason *string `json:"finish_reason"`
}

type EmbeddingsResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  EmbeddingsUsage `json:"usage"`
}

type EmbeddingData struct {
	Object    string `json:"object"`
	Embedding []int  `json:"embedding"`
	Index     int    `json:"index"`
}

type EmbeddingsUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func BuildChatCompletion(model string, content string) ChatCompletionResponse {
	stop := "stop"
	return ChatCompletionResponse{
		ID: "chatcmpl-placeholder", Object: "chat.completion", Created: 0, Model: model,
		Choices: []ChatChoice{{Index: 0, Message: &ChatMessage{Role: "assistant", Content: content}, FinishReason: &stop}},
		Usage:   Usage{},
	}
}

func BuildChatChunk(model string, content string, finished bool) ChatCompletionResponse {
	var finish *string
	choice := ChatChoice{Index: 0}
	if finished {
		stop := "stop"
		finish = &stop
		choice.Delta = &ChatDelta{}
	} else {
		choice.Delta = &ChatDelta{Role: "assistant", Content: content}
	}
	choice.FinishReason = finish
	return ChatCompletionResponse{ID: "chatcmpl-placeholder", Object: "chat.completion.chunk", Created: 0, Model: model, Choices: []ChatChoice{choice}}
}

func BuildTextCompletion(model string, text string) TextCompletionResponse {
	stop := "stop"
	return TextCompletionResponse{
		ID: "cmpl-placeholder", Object: "text_completion", Created: 0, Model: model,
		Choices: []TextChoice{{Index: 0, Text: text, FinishReason: &stop}},
		Usage:   Usage{},
	}
}

func BuildTextChunk(model string, text string, finished bool) TextCompletionResponse {
	var finish *string
	out := text
	if finished {
		stop := "stop"
		finish = &stop
	}
	return TextCompletionResponse{ID: "cmpl-placeholder", Object: "text_completion", Created: 0, Model: model, Choices: []TextChoice{{Index: 0, Text: out, FinishReason: finish}}}
}

func BuildModelsResponse(modelIDs ...string) ModelsResponse {
	if len(modelIDs) == 0 {
		modelIDs = []string{"oxidize-default"}
	}
	data := make([]ModelData, 0, len(modelIDs))
	for _, modelID := range modelIDs {
		data = append(data, ModelData{ID: modelID, Object: "model", OwnedBy: "oxidize"})
	}
	return ModelsResponse{Object: "list", Data: data}
}

func BuildEmbeddingsResponse(model string) EmbeddingsResponse {
	return EmbeddingsResponse{
		Object: "list",
		Data:   []EmbeddingData{{Object: "embedding", Embedding: []int{}, Index: 0}},
		Model:  model,
		Usage:  EmbeddingsUsage{},
	}
}

func ValidateCandidateCount(n *int, bestOf *int) *ErrorResponse {
	actualN := 1
	actualBestOf := 1
	if n != nil {
		actualN = *n
		actualBestOf = actualN
	}
	if bestOf != nil {
		actualBestOf = *bestOf
	}
	if actualN == 1 && actualBestOf == 1 {
		return nil
	}
	return &ErrorResponse{StatusCode: 400, Error: APIError{Message: "oxidize-server currently supports only n=1 and best_of=1", Type: "unsupported_parameter"}}
}

func ModelNotFound(model string) ErrorResponse {
	return ErrorResponse{StatusCode: 404, Error: APIError{Message: "model '" + model + "' is not loaded", Type: "model_not_found"}}
}

func InvalidAPIKey() ErrorResponse {
	return ErrorResponse{StatusCode: 401, Error: APIError{Message: "invalid api key", Type: "invalid_api_key"}}
}

func MalformedJSON() ErrorResponse {
	return ErrorResponse{StatusCode: 400, Error: APIError{Message: "malformed json", Type: "invalid_request_error"}}
}
