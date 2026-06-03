package generate

import (
	"context"
	"fmt"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/model"
	"github.com/Zapdev-labs/oxidize/golang/core/tokenizer"
)

// CompletionText runs greedy generation for API handlers.
func CompletionText(ctx context.Context, modelPath, prompt string, maxTokens int) (string, error) {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", nil
	}
	if maxTokens <= 0 {
		maxTokens = 64
	}

	loaded, err := model.LoadGGUFModelFromPath(modelPath, model.NewLoaderConfig())
	if err != nil {
		return "", err
	}
	inference, ok := loaded.(*model.InferenceModel)
	if !ok || inference.Stack == nil || !inference.Stack.Loaded() {
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
	genCfg.MaxNewTokens = maxTokens
	genCfg.Sampling.Temperature = 0

	stream := model.NewGenerationStream(inference, session, genCfg)
	stream.Seed(promptTokens)

	var out strings.Builder
	for i := 0; i < maxTokens; i++ {
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
