package cli

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
	"github.com/Zapdev-labs/oxidize/golang/internal/server"
	"github.com/Zapdev-labs/oxidize/golang/internal/serviceinfo"
)

func Run(ctx context.Context, args []string, stdout io.Writer, stderr io.Writer) error {
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return runLegacy(ctx, args, stdout, stderr)
	}
	switch args[0] {
	case "run":
		return runCommand(ctx, args[1:], stdout, stderr)
	case "chat":
		return chatCommand(ctx, args[1:], stdout, stderr)
	case "inspect":
		return inspectCommand(args[1:], stdout)
	case "bench":
		return benchCommand(ctx, args[1:], stdout)
	case "list", "ls":
		return listCommand(args[1:], stdout)
	case "serve":
		return serveCommand(ctx, args[1:])
	case "gpu-cluster":
		return gpuClusterCommand(args[1:], stdout, stderr)
	case "-h", "--help", "help":
		printOllamaHelp(stdout)
		return nil
	default:
		_, _ = fmt.Fprintf(stderr, "unknown command: %s\n", args[0])
		printOllamaHelp(stderr)
		return fmt.Errorf("unknown command: %s", args[0])
	}
}

func runLegacy(ctx context.Context, args []string, stdout io.Writer, stderr io.Writer) error {
	fs := flag.NewFlagSet("oxidize", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	prompt := fs.String("prompt", "", "prompt")
	model := fs.String("model", "", "model path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	promptText := strings.TrimSpace(*prompt)
	if promptText == "" {
		return nil
	}
	modelPath := strings.TrimSpace(*model)
	if modelPath != "" {
		resolved, err := resolveModelPathWithHF(modelPath, "")
		if err != nil {
			return err
		}
		if err := validateGGUFPath(resolved); err == nil {
			cfg := generate.DefaultRunConfig()
			cfg.ModelPath = resolved
			cfg.Prompt = promptText
			cfg.MaxNewTokens = 128
			return generate.RunFromGGUF(ctx, cfg, stdout)
		}
	}
	_, err := io.WriteString(stdout, generate.CLITranscript(promptText))
	return err
}

func runCommand(ctx context.Context, args []string, stdout io.Writer, stderr io.Writer) error {
	return runOrChat(ctx, args, stdout, stderr, false)
}

func runOrChat(ctx context.Context, args []string, stdout io.Writer, stderr io.Writer, chat bool) error {
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		if chat {
			printChatHelp(stdout)
		} else {
			printRunHelp(stdout)
		}
		return nil
	}
	cmd := "run"
	if chat {
		cmd = "chat"
	}
	_, opts, rest, err := parseRunFlags(cmd, args)
	if err != nil {
		return err
	}
	modelArg, err := requireModelArg(rest)
	if err != nil {
		return fmt.Errorf("oxidize %s %w", cmd, err)
	}
	if !chat && strings.TrimSpace(opts.Prompt) == "" {
		return nil
	}
	modelPath, err := resolveModelPathWithHF(modelArg, opts.HFFile)
	if err != nil {
		return err
	}
	if done, err := maybeRunPipeline(ctx, opts, modelPath, stdout); done {
		return err
	}
	if opts.Mesh {
		if done, err := maybeRunMeshChat(ctx, opts, modelPath, stdout, stderr); done {
			return err
		}
	}
	cfg := opts.runConfig(modelPath)
	if chat {
		return chatREPL(ctx, cfg, stdout, stderr)
	}
	if err := generateRun(ctx, cfg, stdout, stderr); err != nil {
		_, _ = fmt.Fprintf(stderr, "generation failed: %v\n", err)
		return err
	}
	return nil
}

func listCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("list", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	modelsDir := fs.String("models-dir", "", "models directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	dir := resolveModelsDir(*modelsDir)
	if _, err := io.WriteString(stdout, fmt.Sprintf("%-48s %9s %s\n", "NAME", "SIZE", "PATH")); err != nil {
		return err
	}
	if dir == "" {
		return nil
	}
	models, err := serviceinfo.DiscoverModels(dir)
	if err != nil {
		return err
	}
	for _, model := range models {
		sizeGiB := "?"
		if stat, statErr := os.Stat(model.Path); statErr == nil {
			sizeGiB = fmt.Sprintf("%.2fG", float64(stat.Size())/1024/1024/1024)
		}
		name := model.ID
		if name == "" {
			name = model.Path
		}
		line := fmt.Sprintf("%-48s %9s %s\n", name, sizeGiB, model.Path)
		if _, writeErr := io.WriteString(stdout, line); writeErr != nil {
			return writeErr
		}
	}
	return nil
}

func serveCommand(ctx context.Context, args []string) error {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		printServeHelp(os.Stdout)
		return nil
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	host := fs.String("host", "127.0.0.1", "host")
	port := fs.Int("port", 8080, "port")
	modelsDir := fs.String("models-dir", "", "models directory")
	var genOpts genOptions
	registerGenFlags(fs, &genOpts)
	if err := fs.Parse(args); err != nil {
		return err
	}
	defaultModel := ""
	if len(fs.Args()) > 0 {
		resolved, resolveErr := resolveModelPathWithHF(fs.Args()[0], genOpts.HFFile)
		if resolveErr != nil {
			return resolveErr
		}
		defaultModel = resolved
	}
	loader := genOpts.loaderConfig()
	return server.Listen(ctx, server.Config{
		Host:         *host,
		Port:         *port,
		ModelsDir:    resolveModelsDir(*modelsDir),
		DefaultModel: defaultModel,
		Backend:      loader.Backend,
		MaxTokens:    genOpts.MaxTokens,
		Temperature:  float32(genOpts.Temperature),
		TopP:         float32(genOpts.TopP),
		TopK:         genOpts.TopK,
		Loader:       loader,
	})
}

func printOllamaHelp(w io.Writer) {
	_, _ = fmt.Fprintln(w, `Usage: oxidize <command> [args]

Commands:
  run <model> [prompt]     Run a model locally
  chat <model>             Interactive chat REPL
  bench <model>            Decode throughput benchmark
  inspect <model.gguf>     Print GGUF metadata and tensors
  serve [options]          Start the OpenAI-compatible server
  gpu-cluster <subcmd>     Generate or detect GPU cluster configs
  list                     List local GGUF models in ./models

Examples:
  oxidize run ./models/Qwen3-4B-Q4_K_M.gguf "hello"
  oxidize chat ./models/model.gguf
  oxidize bench ./models/model.gguf --iterations 5
  oxidize inspect ./models/model.gguf
  oxidize serve --host 0.0.0.0 --port 11434
  oxidize list`)
}

func printRunHelp(w io.Writer) {
	_, _ = fmt.Fprintln(w, `Usage: oxidize run <model> [prompt] [options]

Models can be local .gguf files or Hugging Face GGUF repos.

Examples:
  oxidize run ./models/model.gguf "hello"
  oxidize run Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf "hi"

Common options: --prompt, --max-tokens, --temperature, --top-p, --top-k, --backend, --threads, --ctx-size, --draft-model, --file`)
}

func printServeHelp(w io.Writer) {
	_, _ = fmt.Fprintln(w, `Usage: oxidize serve [model] [options]

Starts the OpenAI-compatible API server.

Examples:
  oxidize serve ./models/model.gguf --host 0.0.0.0 --port 11434
  oxidize serve --models-dir ./models --temperature 0.7

Common options: --host, --port, --models-dir, --max-tokens, --temperature, --backend, --file`)
}
