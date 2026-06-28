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
	_, opts, visits, rest, err := parseRunFlags(cmd, args)
	if err != nil {
		return err
	}
	modelArg, err := requireModelArg(rest)
	if err != nil {
		return fmt.Errorf("oxidize %s %w", cmd, err)
	}

	if !chat && strings.TrimSpace(opts.Prompt) == "" {
		if stdinPrompt, ok := readStdinPrompt(); ok {
			opts.Prompt = stdinPrompt
		}
	}

	modelPath, err := resolveModelPathWithHF(modelArg, opts.HFFile)
	if err != nil {
		return err
	}
	if err := applyAutotune(modelPath, &opts, visits, stderr); err != nil {
		_, _ = fmt.Fprintf(stderr, "autotune warning: %v\n", err)
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

	interactiveRun := !chat && strings.TrimSpace(opts.Prompt) == "" && isInteractiveTerminal()
	if chat || interactiveRun {
		return chatREPL(ctx, cfg, stdout, stderr)
	}
	if strings.TrimSpace(opts.Prompt) == "" {
		return nil
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
	if dir == "" {
		_, _ = fmt.Fprintln(stdout, "No models directory found. Use --models-dir or create ./models")
		return nil
	}
	models, err := serviceinfo.DiscoverModels(dir)
	if err != nil {
		return err
	}
	if len(models) == 0 {
		_, _ = fmt.Fprintf(stdout, "No GGUF models found in %s\n", dir)
		return nil
	}

	filter := ""
	if len(fs.Args()) > 0 {
		filter = strings.ToLower(fs.Arg(0))
	}

	var rows [][]string
	for _, model := range models {
		name := model.ID
		if name == "" {
			name = model.Path
		}
		if filter != "" && !strings.HasPrefix(strings.ToLower(name), filter) {
			continue
		}
		size := "?"
		modified := "Never"
		if stat, statErr := os.Stat(model.Path); statErr == nil {
			size = humanBytes(stat.Size())
			modified = humanTime(stat.ModTime(), "Never")
		}
		rows = append(rows, []string{name, modelDigest(model.Path), size, modified})
	}
	if len(rows) == 0 {
		_, _ = fmt.Fprintf(stdout, "No models matching %q in %s\n", filter, dir)
		return nil
	}
	renderTable(stdout, []string{"NAME", "ID", "SIZE", "MODIFIED"}, rows)
	return nil
}

func readStdinPrompt() (string, bool) {
	fi, err := os.Stdin.Stat()
	if err != nil || (fi.Mode()&os.ModeCharDevice) != 0 {
		return "", false
	}
	in, err := io.ReadAll(os.Stdin)
	if err != nil || len(in) == 0 {
		return "", false
	}
	return strings.TrimSpace(string(in)), true
}

func isInteractiveTerminal() bool {
	fi, err := os.Stdin.Stat()
	if err != nil || (fi.Mode()&os.ModeCharDevice) == 0 {
		return false
	}
	fo, err := os.Stdout.Stat()
	return err == nil && (fo.Mode()&os.ModeCharDevice) != 0
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
