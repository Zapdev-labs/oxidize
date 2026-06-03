package generate

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/Zapdev-labs/oxidize/golang/core/model"
	"github.com/Zapdev-labs/oxidize/golang/core/tokenizer"
)

// RunConfig controls CLI / server text generation.
type RunConfig struct {
	ModelPath      string
	Prompt         string
	MaxNewTokens   int
	Temperature    float32
	TopP           float32
	StopToken      model.Token
}

// DefaultRunConfig returns sensible generation defaults.
func DefaultRunConfig() RunConfig {
	return RunConfig{
		MaxNewTokens: 64,
		Temperature:  0.8,
		TopP:         0.9,
		StopToken:    2,
	}
}

// RunFromGGUF loads a GGUF model and streams generated text to stdout.
func RunFromGGUF(ctx context.Context, cfg RunConfig, stdout io.Writer) error {
	if strings.TrimSpace(cfg.ModelPath) == "" {
		return fmt.Errorf("generate: empty model path")
	}
	if strings.TrimSpace(cfg.Prompt) == "" {
		return nil
	}

	loaded, err := model.LoadGGUFModelFromPath(cfg.ModelPath, model.NewLoaderConfig())
	if err != nil {
		return err
	}
	inference, ok := loaded.(*model.InferenceModel)
	if !ok {
		return fmt.Errorf("generate: expected InferenceModel, got %T", loaded)
	}
	if inference.Stack == nil || !inference.Stack.Loaded() {
		return fmt.Errorf("generate: model %q has no loadable transformer weights", cfg.ModelPath)
	}

	tok, err := tokenizer.LoadFromGGUFFile(cfg.ModelPath)
	if err != nil {
		tok = tokenizer.NewBpeTokenizer(nil, nil, tokenizer.SpecialTokens{BOS: 1, EOS: 2})
	}
	promptTokens, err := tok.Encode(cfg.Prompt, tokenizer.EncodeOptions{})
	if err != nil {
		return fmt.Errorf("generate: encode prompt: %w", err)
	}
	if len(promptTokens) == 0 {
		promptTokens = []model.Token{1}
	}

	session := model.NewSession()
	genCfg := model.DefaultGenerationConfig()
	if cfg.MaxNewTokens > 0 {
		genCfg.MaxNewTokens = cfg.MaxNewTokens
	}
	if cfg.StopToken != 0 {
		genCfg.StopToken = cfg.StopToken
	}
	genCfg.Sampling.Temperature = cfg.Temperature
	genCfg.Sampling.TopP = cfg.TopP

	stream := model.NewGenerationStream(inference, session, genCfg)
	stream.Seed(promptTokens)

	start := time.Now()
	for i := 0; i < genCfg.MaxNewTokens; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		token, done, err := stream.Next(ctx)
		if err != nil {
			return err
		}
		if done {
			break
		}
		piece, err := tok.Decode([]model.Token{token})
		if err != nil {
			piece = fmt.Sprintf("<%d>", token)
		}
		if _, err := io.WriteString(stdout, piece); err != nil {
			return err
		}
	}

	elapsed := time.Since(start).Seconds()
	tokens := session.ConsumedTokens()
	speed := 0.0
	if elapsed > 0 && tokens > 0 {
		speed = float64(tokens) / elapsed
	}
	stats := fmt.Sprintf("\ngeneration stats: tokens=%d speed=%.2f tok/s\n", tokens, speed)
	_, err = io.WriteString(stdout, stats)
	return err
}
