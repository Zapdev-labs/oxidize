package generate

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/Zapdev-labs/oxidize/golang/core/model"
	"github.com/Zapdev-labs/oxidize/golang/core/tokenizer"
	"github.com/Zapdev-labs/oxidize/golang/core/vision"
)

// RunConfig controls CLI / server text generation.
type RunConfig struct {
	ModelPath      string
	Prompt         string
	MaxNewTokens   int
	Temperature    float32
	TopP           float32
	TopK           int
	Threads        int
	ContextSize    int
	Backend        string
	NGPULayers     int
	GPUs           int
	Parallelism    string
	DraftModel     string
	DraftTokens    int
	TokenizerModel string
	HFFile         string
	StopToken      model.Token
	UsePaged        bool
	UseDFlashFusion bool
	Vision          bool
	ImagePath       string
	LayerWise       bool
	LayerCache      int
	RAMOffload      bool
}

// DefaultRunConfig returns sensible generation defaults.
func DefaultRunConfig() RunConfig {
	return RunConfig{
		MaxNewTokens: 64,
		Temperature:  0.8,
		TopP:         0.9,
		StopToken:    2,
		DraftTokens:  4,
		Backend:      "cpu",
	}
}

func (cfg RunConfig) loaderConfig() LoaderConfig {
	return LoaderConfig{
		Backend:       cfg.Backend,
		Threads:       cfg.Threads,
		ContextSize:   cfg.ContextSize,
		NGPULayers:    cfg.NGPULayers,
		GPUs:          cfg.GPUs,
		Parallelism:   cfg.Parallelism,
		HFFilename:    cfg.HFFile,
		AllowFallback: true,
	}
}

func (cfg RunConfig) generationConfig() model.GenerationConfig {
	genCfg := model.DefaultGenerationConfig()
	if cfg.MaxNewTokens > 0 {
		genCfg.MaxNewTokens = cfg.MaxNewTokens
	}
	if cfg.StopToken != 0 {
		genCfg.StopToken = cfg.StopToken
	}
	genCfg.Sampling.Temperature = cfg.Temperature
	genCfg.Sampling.TopP = cfg.TopP
	if cfg.TopK > 0 {
		genCfg.Sampling.TopK = cfg.TopK
	}
	return genCfg
}

func loadTokenizer(modelPath, tokenizerPath string) (tokenizer.Tokenizer, error) {
	if strings.TrimSpace(tokenizerPath) != "" {
		return tokenizer.LoadFromGGUFFile(tokenizerPath)
	}
	tok, err := tokenizer.LoadFromGGUFFile(modelPath)
	if err != nil {
		return tokenizer.NewBpeTokenizer(nil, nil, tokenizer.SpecialTokens{BOS: 1, EOS: 2}), nil
	}
	return tok, nil
}

// RunFromGGUF loads a GGUF model and streams generated text to stdout.
func RunFromGGUF(ctx context.Context, cfg RunConfig, stdout io.Writer) error {
	if strings.TrimSpace(cfg.ModelPath) == "" {
		return fmt.Errorf("generate: empty model path")
	}
	if strings.TrimSpace(cfg.Prompt) == "" {
		return nil
	}
	if cfg.UsePaged {
		return RunPagedFromGGUF(ctx, cfg, stdout)
	}
	if cfg.Vision && strings.TrimSpace(cfg.ImagePath) != "" {
		if raw, err := os.ReadFile(cfg.ImagePath); err == nil {
			cfgVision := vision.DefaultConfig()
			enc := vision.NewPatchEncoder(cfgVision)
			if vecs, err := enc.Encode(raw); err == nil {
				dims := enc.Dims()
				_, _ = fmt.Fprintf(stdout, "# vision: patch encoder dims=%v len=%d\n", dims, len(vecs))
			}
		}
	}

	result, err := DefaultModelCache.Load(cfg.ModelPath, cfg.loaderConfig())
	if err != nil {
		return err
	}
	if result.Warning != "" {
		_, _ = fmt.Fprintf(stdout, "# backend: %s (%s)\n", result.Backend, result.Warning)
	}

	inference, ok := result.Result.Model.(*model.InferenceModel)
	if !ok {
		return fmt.Errorf("generate: expected InferenceModel, got %T", result.Result.Model)
	}
	if inference.Stack == nil || !inference.Stack.Loaded() {
		return fmt.Errorf("generate: model %q has no loadable transformer weights", cfg.ModelPath)
	}

	tok, err := loadTokenizer(cfg.ModelPath, cfg.TokenizerModel)
	if err != nil {
		return fmt.Errorf("generate: tokenizer: %w", err)
	}
	promptTokens, err := tok.Encode(cfg.Prompt, tokenizer.EncodeOptions{})
	if err != nil {
		return fmt.Errorf("generate: encode prompt: %w", err)
	}
	if len(promptTokens) == 0 {
		promptTokens = []model.Token{1}
	}

	session := model.NewSession()
	genCfg := cfg.generationConfig()
	start := time.Now()

	streamModel := model.Model(inference)
	if cfg.LayerWise {
		if cfg.LayerCache <= 0 {
			cfg.LayerCache = 4
		}
		streamModel = model.NewLayerWiseFromInference(inference, cfg.LayerCache)
	}

	if strings.TrimSpace(cfg.DraftModel) != "" || cfg.UseDFlashFusion {
		draftPath := strings.TrimSpace(cfg.DraftModel)
		var draft model.Model
		var err error
		if draftPath != "" {
			draft, err = LoadDraftFromPath(draftPath, cfg.loaderConfig(), inference.Config.HiddenSize)
		} else {
			draft = model.NewHeuristicDFlashDraft(streamModel, model.DefaultDFlashConfig())
		}
		if err != nil {
			return fmt.Errorf("generate: draft model: %w", err)
		}
		if cfg.UseDFlashFusion {
			dec := model.NewSpeculativeDecoder(draft, streamModel, session, model.SpeculativeConfig{
				DraftTokensPerStep: cfg.DraftTokens,
				MaxNewTokens:       genCfg.MaxNewTokens,
				Sampling:           genCfg.Sampling,
				StopToken:          &genCfg.StopToken,
			})
			if cfg.DraftTokens > 0 {
				dec.Config.DraftTokensPerStep = cfg.DraftTokens
			}
			_, _ = streamModel.Forward(promptTokens, session)
			for i := 0; i < genCfg.MaxNewTokens; i++ {
				if err := ctx.Err(); err != nil {
					return err
				}
				accepted, err := dec.Step()
				if err != nil {
					return err
				}
				if len(accepted) == 0 {
					break
				}
				for _, token := range accepted {
					piece, err := tok.Decode([]model.Token{token})
					if err != nil {
						piece = fmt.Sprintf("<%d>", token)
					}
					if _, err := io.WriteString(stdout, piece); err != nil {
						return err
					}
				}
			}
			elapsed := time.Since(start).Seconds()
			tokens := session.ConsumedTokens()
			speed := 0.0
			if elapsed > 0 && tokens > 0 {
				speed = float64(tokens) / elapsed
			}
			stats := fmt.Sprintf("\ngeneration stats: tokens=%d speed=%.2f tok/s (dflash)\n", tokens, speed)
			_, err = io.WriteString(stdout, stats)
			return err
		}
		specCfg := model.DefaultSpeculativeGenerationConfig()
		specCfg.Generation = genCfg
		if cfg.DraftTokens > 0 {
			specCfg.DraftTokensPerStep = cfg.DraftTokens
		}
		stream := model.NewSpeculativeGenerationStream(draft, streamModel, session, specCfg)
		stream.Seed(promptTokens)
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
	} else if model.HasMTPWeights(cfg.ModelPath) {
		mtpStream := model.NewMtpGenerationStream(streamModel, session, genCfg)
		mtpStream.Seed(promptTokens)
		for i := 0; i < genCfg.MaxNewTokens; i++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			token, done, err := mtpStream.Next(ctx)
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
	} else {
		stream := model.NewGenerationStream(streamModel, session, genCfg)
		stream.Seed(promptTokens)
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
