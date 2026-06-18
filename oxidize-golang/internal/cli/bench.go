package cli

import (
	"context"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/Zapdev-labs/oxidize/golang/core/model"
	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
	"github.com/Zapdev-labs/oxidize/golang/core/tokenizer"
	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
)

func benchCommand(ctx context.Context, args []string, stdout io.Writer) error {
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		_, _ = fmt.Fprintln(stdout, `Usage: oxidize bench <model> [options]

Runs a short decode benchmark and reports tokens/sec.

Options:
  --engine NAME    inference (default) or dflash
  --iterations N   benchmark rounds (default 3)
  --max-tokens N   tokens generated per round (default 32)
  --prompt TEXT    prompt seed (default "benchmark")
  --file NAME      GGUF filename for Hugging Face repos
  (plus run flags: --backend, --temperature, --draft-model, ...)`)
		return nil
	}
	engine, rest, err := parseBenchEngine(args)
	if err != nil {
		return err
	}
	if err := validateBenchEngine(engine); err != nil {
		return err
	}
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	iterations := fs.Int("iterations", 3, "benchmark rounds")
	maxTokens := fs.Int("max-tokens", 32, "tokens per round")
	prompt := fs.String("prompt", "benchmark", "prompt seed")
	_, genOpts, _, flagRest, err := parseGenFlags("bench", rest)
	if err != nil {
		return err
	}
	if err := fs.Parse(flagRest); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		return fmt.Errorf("oxidize bench requires a model name or local .gguf path")
	}

	modelPath, err := resolveModelPathWithHF(fs.Arg(0), genOpts.HFFile)
	if err != nil {
		return err
	}

	loader := genOpts.loaderConfig()
	result, err := generate.LoadModelFromPath(modelPath, loader)
	if err != nil {
		return err
	}
	inference, ok := result.Model.(*model.InferenceModel)
	if !ok || inference.Stack == nil || !inference.Stack.Loaded() {
		return fmt.Errorf("bench: model %q has no loadable weights", modelPath)
	}
	tokPath := modelPath
	if genOpts.TokenizerModel != "" {
		tokPath = genOpts.TokenizerModel
	}
	tok, err := tokenizer.LoadFromGGUFFile(tokPath)
	if err != nil {
		tok = tokenizer.NewBpeTokenizer(nil, nil, tokenizer.SpecialTokens{BOS: 1, EOS: 2})
	}
	promptTokens, err := tok.Encode(*prompt, tokenizer.EncodeOptions{})
	if err != nil {
		return fmt.Errorf("bench: encode: %w", err)
	}
	if len(promptTokens) == 0 {
		promptTokens = []model.Token{1}
	}

	_, _ = fmt.Fprintf(
		stdout,
		"=== Oxidize bench ===\nmodel: %s\nengine: %s\niterations: %d max_tokens: %d\n\n",
		modelPath,
		engine,
		*iterations,
		*maxTokens,
	)

	// Fast path: use Rust FFI model (same AVX2+Rayon kernels as the Rust binary).
	if rm, err2 := quantization.LoadRustModel(modelPath); err2 == nil {
		defer rm.Close()
		var totalTokens int
		var totalSeconds float64
		for round := 1; round <= *iterations; round++ {
			rm.ResetSession()
			promptIDs := make([]uint32, len(promptTokens))
			for i, t := range promptTokens {
				promptIDs[i] = uint32(t)
			}
			start := time.Now()
			if _, ferr := rm.Forward(promptIDs); ferr != nil {
				_, _ = fmt.Fprintf(stdout, "rust forward failed: %v\n", ferr)
				break
			}
			tok := rm.SampleArgmax()
			generated := 1
			for i := 0; i < *maxTokens; i++ {
				if err3 := ctx.Err(); err3 != nil {
					return err3
				}
				if _, ferr := rm.Forward([]uint32{tok}); ferr != nil {
					_, _ = fmt.Fprintf(stdout, "rust forward failed: %v\n", ferr)
					break
				}
				tok = rm.SampleArgmax()
				generated++
			}
			elapsed := time.Since(start).Seconds()
			speed := float64(generated) / elapsed
			totalTokens += generated
			totalSeconds += elapsed
			_, _ = fmt.Fprintf(stdout, "round %d: tokens=%d elapsed=%.3fs speed=%.2f tok/s\n", round, generated, elapsed, speed)
		}
		avg := 0.0
		if totalSeconds > 0 {
			avg = float64(totalTokens) / totalSeconds
		}
		_, _ = fmt.Fprintf(stdout, "\naverage: %.2f tok/s over %d tokens\n", avg, totalTokens)
		return nil
	}

	genCfg := model.DefaultGenerationConfig()
	genCfg.MaxNewTokens = *maxTokens
	genCfg.Sampling.Temperature = float32(genOpts.Temperature)
	genCfg.Sampling.TopP = float32(genOpts.TopP)
	if genOpts.TopK > 0 {
		genCfg.Sampling.TopK = genOpts.TopK
	}

	var draftModel model.Model
	if engine == "dflash" {
		if genOpts.DraftModel != "" {
			draftModel, err = generate.LoadDraftFromPath(genOpts.DraftModel, loader, inference.Config.HiddenSize)
			if err != nil {
				return fmt.Errorf("bench: draft: %w", err)
			}
		} else {
			dcfg := model.DefaultDFlashConfig()
			draftModel = model.NewHeuristicDFlashDraft(inference, dcfg)
		}
	}

	var totalTokens int
	var totalSeconds float64
	for round := 1; round <= *iterations; round++ {
		session := model.NewSession()
		start := time.Now()
		generated := 0

		if engine == "dflash" && draftModel != nil {
			specCfg := model.DefaultSpeculativeGenerationConfig()
			specCfg.Generation = genCfg
			if genOpts.DraftTokens > 0 {
				specCfg.DraftTokensPerStep = genOpts.DraftTokens
			}
			stream := model.NewSpeculativeGenerationStream(draftModel, inference, session, specCfg)
			stream.Seed(promptTokens)
			for i := 0; i < *maxTokens; i++ {
				if err := ctx.Err(); err != nil {
					return err
				}
				_, done, err := stream.Next(ctx)
				if err != nil {
					return err
				}
				generated++
				if done {
					break
				}
			}
		} else {
			stream := model.NewGenerationStream(inference, session, genCfg)
			stream.Seed(promptTokens)
			for i := 0; i < *maxTokens; i++ {
				if err := ctx.Err(); err != nil {
					return err
				}
				_, done, err := stream.Next(ctx)
				if err != nil {
					return err
				}
				generated++
				if done {
					break
				}
			}
		}

		elapsed := time.Since(start).Seconds()
		tokens := session.ConsumedTokens()
		if tokens == 0 {
			tokens = generated
		}
		speed := float64(tokens) / elapsed
		totalTokens += tokens
		totalSeconds += elapsed
		_, _ = fmt.Fprintf(stdout, "round %d: tokens=%d elapsed=%.3fs speed=%.2f tok/s\n", round, tokens, elapsed, speed)
	}
	avg := float64(totalTokens) / totalSeconds
	_, _ = fmt.Fprintf(stdout, "\naverage: %.2f tok/s over %d tokens\n", avg, totalTokens)
	return nil
}
