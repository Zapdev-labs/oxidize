package cli

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
)

func chatCommand(ctx context.Context, args []string, stdout io.Writer, stderr io.Writer) error {
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		printChatHelp(stdout)
		return nil
	}
	return runOrChat(ctx, args, stdout, stderr, true)
}

func chatREPL(ctx context.Context, cfg generate.RunConfig, stdout, stderr io.Writer) error {
	_, _ = fmt.Fprintf(stdout, ">>> oxidize · %s\n", shortModelName(cfg.ModelPath))
	_, _ = fmt.Fprintln(stdout, "Send a message. /help for commands, /bye to exit.")
	scanner := bufio.NewScanner(os.Stdin)
	for {
		if _, err := io.WriteString(stdout, "\n>>> "); err != nil {
			return err
		}
		if !scanner.Scan() {
			_, _ = io.WriteString(stdout, "\n")
			return nil
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		lower := strings.ToLower(line)
		switch lower {
		case "exit", "quit", "/bye", "/exit":
			return nil
		case "/help", "help":
			_, _ = fmt.Fprintln(stdout, "  /help  show commands")
			_, _ = fmt.Fprintln(stdout, "  /bye   exit chat")
			continue
		case "/clear":
			continue
		}
		cfg.Prompt = line
		if err := generate.RunFromGGUF(ctx, cfg, stdout); err != nil {
			_, _ = fmt.Fprintf(stderr, "generation failed: %v\n", err)
		}
	}
}

func shortModelName(path string) string {
	base := path
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' || path[i] == '\\' {
			base = path[i+1:]
			break
		}
	}
	return strings.TrimSuffix(base, ".gguf")
}

func printChatHelp(w io.Writer) {
	_, _ = fmt.Fprintln(w, `Usage: oxidize chat <model> [options]

Interactive REPL for local GGUF models.

Examples:
  oxidize chat ./models/model.gguf
  oxidize chat Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf

Common options: --max-tokens, --temperature, --top-p, --top-k, --backend, --threads, --ctx-size, --draft-model, --file`)
}
