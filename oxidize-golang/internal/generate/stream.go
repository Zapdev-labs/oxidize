package generate

import (
	"context"
	"fmt"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/model"
	"github.com/Zapdev-labs/oxidize/golang/core/tokenizer"
)

// TokenHandler receives decoded text pieces during streaming generation.
type TokenHandler func(piece string) error

// StreamCompletion generates text token-by-token using the model cache.
func StreamCompletion(ctx context.Context, modelPath, prompt string, params CompletionParams, onToken TokenHandler) (string, error) {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", nil
	}
	if params.MaxTokens <= 0 {
		params.MaxTokens = 64
	}

	inference, entry, err := InferenceFromCache(modelPath, params.Loader)
	if err != nil {
		return "", err
	}
	if inference.Stack == nil || !inference.Stack.Loaded() {
		text := PlaceholderText(PlaceholderSpec{})
		if onToken != nil && text != "" {
			_ = onToken(text)
		}
		return text, nil
	}
	_ = entry

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
		if onToken != nil {
			if err := onToken(piece); err != nil {
				return out.String(), err
			}
		}
	}
	return out.String(), nil
}
