package generate

import (
	"context"
	"fmt"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/model"
	"github.com/Zapdev-labs/oxidize/golang/core/tokenizer"
)

// CompletionParams controls API completion generation.
type CompletionParams struct {
	MaxTokens   int
	Temperature float32
	TopP        float32
	TopK        int
	Loader      LoaderConfig
}

// DefaultCompletionParams returns API defaults.
func DefaultCompletionParams() CompletionParams {
	return CompletionParams{
		MaxTokens:   64,
		Temperature: 0.8,
		TopP:        0.9,
		Loader:      LoaderConfig{Backend: "cpu", AllowFallback: true},
	}
}

// CompletionText runs generation for API handlers.
func CompletionText(ctx context.Context, modelPath, prompt string, params CompletionParams) (string, error) {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", nil
	}
	if params.MaxTokens <= 0 {
		params.MaxTokens = 64
	}

	inference, _, err := InferenceFromCache(modelPath, params.Loader)
	if err != nil {
		return "", err
	}
	if inference == nil || inference.Stack == nil || !inference.Stack.Loaded() {
		return PlaceholderText(PlaceholderSpec{}), nil
	}

	tok, err := tokenizer.LoadFromGGUFFile(modelPath)
	if err != nil {
		tok = tokenizer.NewBpeTokenizer(nil, nil, tokenizer.SpecialTokens{BOS: 1, EOS: 2})
	}
	promptTokens, err := tok.Encode(prompt, tokenizer.EncodeOptions{})
	if err != nil {
		return "", fmt.Errorf("encode: %w", err)
	}
	if len(promptTokens) == 0 {
		promptTokens = []model.Token{1}
	}

	session := model.NewSession()
	genCfg := model.DefaultGenerationConfig()
	genCfg.MaxNewTokens = params.MaxTokens
	genCfg.Sampling.Temperature = params.Temperature
	genCfg.Sampling.TopP = params.TopP
	if params.TopK > 0 {
		genCfg.Sampling.TopK = params.TopK
	}
	if params.Temperature <= 0 {
		genCfg.Sampling.Temperature = 0
	}

	stream := model.NewGenerationStream(inference, session, genCfg)
	stream.Seed(promptTokens)

	var out strings.Builder
	for i := 0; i < params.MaxTokens; i++ {
		if err := ctx.Err(); err != nil {
			return out.String(), err
		}
		token, done, err := stream.Next(ctx)
		if err != nil {
			return out.String(), err
		}
		if done {
			break
		}
		piece, err := tok.Decode([]model.Token{token})
		if err != nil {
			piece = fmt.Sprintf("<%d>", token)
		}
		out.WriteString(piece)
	}
	return out.String(), nil
}
