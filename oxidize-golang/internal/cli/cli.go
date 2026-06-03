package cli

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/hf"
	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
	"github.com/Zapdev-labs/oxidize/golang/internal/server"
	"github.com/Zapdev-labs/oxidize/golang/internal/serviceinfo"
)

func Run(ctx context.Context, args []string, stdout io.Writer, stderr io.Writer) error {
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return runLegacy(args, stdout)
	}
	switch args[0] {
	case "run":
		return runCommand(args[1:], stdout, stderr)
	case "list", "ls":
		return listCommand(args[1:], stdout)
	case "serve":
		return serveCommand(ctx, args[1:])
	case "-h", "--help", "help":
		printOllamaHelp(stdout)
		return nil
	default:
		_, _ = fmt.Fprintf(stderr, "unknown command: %s\n", args[0])
		printOllamaHelp(stderr)
		return fmt.Errorf("unknown command: %s", args[0])
	}
}

func runLegacy(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("oxidize", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	prompt := fs.String("prompt", "", "prompt")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*prompt) == "" {
		return nil
	}
	_, err := io.WriteString(stdout, generate.CLITranscript(*prompt))
	return err
}

func runCommand(args []string, stdout io.Writer, stderr io.Writer) error {
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		printRunHelp(stdout)
		return nil
	}
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	prompt := fs.String("prompt", "", "prompt")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		return fmt.Errorf("oxidize run requires a model name or local .gguf path")
	}
	if strings.TrimSpace(*prompt) == "" {
		return nil
	}
	return generate.RunFromGGUF(context.Background(), generate.RunConfig{
		ModelPath:    resolveModelPath(fs.Arg(0)),
		Prompt:       *prompt,
		MaxNewTokens: 128,
	}, stdout)
}

func resolveModelPath(nameOrPath string) string {
	if strings.HasSuffix(strings.ToLower(nameOrPath), ".gguf") {
		return nameOrPath
	}
	if _, err := os.Stat(nameOrPath); err == nil {
		return nameOrPath
	}
	if strings.Contains(nameOrPath, "/") {
		path, err := hf.ResolveGGUF(hf.ResolveOptions{Repo: nameOrPath})
		if err == nil {
			return path
		}
	}
	dir := resolveModelsDir("")
	if dir == "" {
		return nameOrPath
	}
	candidate := filepath.Join(dir, nameOrPath+".gguf")
	if _, err := os.Stat(candidate); err == nil {
		return candidate
	}
	return nameOrPath
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
		_, err := io.WriteString(stdout, "NAME                                             SIZE PATH\n")
		return err
	}
	models, err := serviceinfo.DiscoverModels(dir)
	if err != nil {
		return err
	}
	if _, err := io.WriteString(stdout, fmt.Sprintf("%-48s %9s %s\n", "NAME", "SIZE", "PATH")); err != nil {
		return err
	}
	if len(models) == 0 {
		return nil
	}
	for _, model := range models {
		sizeGiB := "?"
		if stat, statErr := os.Stat(model.Path); statErr == nil {
			sizeGiB = fmt.Sprintf("%.2fG", float64(stat.Size())/1024/1024/1024)
		}
		name := filepath.Base(model.Path)
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
	if err := fs.Parse(args); err != nil {
		return err
	}
	return server.Listen(ctx, server.Config{
		Host:      *host,
		Port:      *port,
		ModelsDir: resolveModelsDir(*modelsDir),
	})
}

func resolveModelsDir(dir string) string {
	if strings.TrimSpace(dir) != "" {
		return dir
	}
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	candidate := filepath.Join(cwd, "models")
	if _, err := os.Stat(candidate); err != nil {
		return ""
	}
	return candidate
}

func printOllamaHelp(w io.Writer) {
	_, _ = fmt.Fprintln(w, `Usage: oxidize <command> [args]

Commands:
  run <model> [prompt]     Run a model locally
  serve [options]          Start the OpenAI-compatible server
  list                     List local GGUF models in ./models

Examples:
  oxidize run ./models/Qwen3-4B-Q4_K_M.gguf "hello"
  oxidize serve --host 0.0.0.0 --port 11434
  oxidize list`)
}

func printRunHelp(w io.Writer) {
	_, _ = fmt.Fprintln(w, `Usage: oxidize run <model> [prompt] [options]

Models can be local .gguf files or Hugging Face GGUF repos.

Examples:
  oxidize run ./models/model.gguf "hello"
  oxidize run Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf

Common options: --prompt, --max-tokens, --temperature, --backend, --threads`)
}

func printServeHelp(w io.Writer) {
	_, _ = fmt.Fprintln(w, `Usage: oxidize serve [options]

Starts the OpenAI-compatible API server.

Examples:
  oxidize serve --host 0.0.0.0 --port 11434
  oxidize serve --models-dir ./models

Common options: --host, --port, --models-dir`)
}

func fallbackArch(value string) string {
	if strings.TrimSpace(value) == "" {
		return "unknown"
	}
	return value
}

func ParsePort(raw string) (int, error) {
	return strconv.Atoi(raw)
}
