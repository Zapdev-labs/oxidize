package cli

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
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
	return chatREPLWithIO(ctx, cfg, os.Stdin, stdout, stderr, isInteractiveTerminal(), generate.RunFromGGUF)
}

type chatGenerator func(context.Context, generate.RunConfig, io.Writer) error

func chatREPLWithIO(ctx context.Context, cfg generate.RunConfig, input io.Reader, stdout, stderr io.Writer, interactive bool, run chatGenerator) error {
	_, _ = fmt.Fprintf(stdout, "oxidize chat · %s\n", shortModelName(cfg.ModelPath))
	printChatSettings(stdout, cfg)
	_, _ = fmt.Fprintln(stdout, "Type a message, or /help for commands.")
	scanner := bufio.NewScanner(input)
	for {
		if _, err := io.WriteString(stdout, "\nyou> "); err != nil {
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
		command := strings.Fields(line)
		switch strings.ToLower(command[0]) {
		case "exit", "quit", "/bye", "/exit":
			return nil
		case "/help", "help":
			printChatCommands(stdout)
			continue
		case "/settings":
			printChatSettings(stdout, cfg)
			continue
		case "/set":
			if err := setChatOption(&cfg, command); err != nil {
				_, _ = fmt.Fprintf(stderr, "chat: %v\n", err)
			} else {
				printChatSettings(stdout, cfg)
			}
			continue
		case "/clear":
			if interactive {
				_, _ = io.WriteString(stdout, "\x1b[2J\x1b[H")
			} else {
				_, _ = fmt.Fprintln(stdout, "--- clear ---")
			}
			continue
		}
		if strings.HasPrefix(line, "/") {
			_, _ = fmt.Fprintf(stderr, "chat: unknown command %q; use /help\n", command[0])
			continue
		}
		cfg.Prompt = line
		_, _ = io.WriteString(stdout, "assistant> ")
		if run != nil {
			if err := run(ctx, cfg, stdout); err != nil {
				_, _ = fmt.Fprintf(stderr, "generation failed: %v\n", err)
			}
		}
	}
}

func printChatCommands(w io.Writer) {
	_, _ = fmt.Fprintln(w, "  /settings                 show generation settings")
	_, _ = fmt.Fprintln(w, "  /set temperature VALUE   set temperature (0 or greater)")
	_, _ = fmt.Fprintln(w, "  /set top-p VALUE         set top-p (0 to 1)")
	_, _ = fmt.Fprintln(w, "  /set top-k VALUE         set top-k (0 or greater)")
	_, _ = fmt.Fprintln(w, "  /set max-tokens VALUE    set response token limit")
	_, _ = fmt.Fprintln(w, "  /clear                    clear the screen")
	_, _ = fmt.Fprintln(w, "  /bye                      exit chat")
}

func printChatSettings(w io.Writer, cfg generate.RunConfig) {
	_, _ = fmt.Fprintf(w, "settings: temperature=%.2f top-p=%.2f top-k=%d max-tokens=%d\n",
		cfg.Temperature, cfg.TopP, cfg.TopK, cfg.MaxNewTokens)
}

func setChatOption(cfg *generate.RunConfig, fields []string) error {
	if len(fields) != 3 {
		return fmt.Errorf("usage: /set temperature|top-p|top-k|max-tokens VALUE")
	}
	name, value := strings.ToLower(fields[1]), fields[2]
	switch name {
	case "temperature", "top-p":
		parsed, err := strconv.ParseFloat(value, 32)
		if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
			return fmt.Errorf("%s must be a number", name)
		}
		if name == "temperature" {
			if parsed < 0 {
				return fmt.Errorf("temperature must be 0 or greater")
			}
			cfg.Temperature = float32(parsed)
			return nil
		}
		if parsed < 0 || parsed > 1 {
			return fmt.Errorf("top-p must be between 0 and 1")
		}
		cfg.TopP = float32(parsed)
		return nil
	case "top-k", "max-tokens":
		parsed, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("%s must be an integer", name)
		}
		if parsed < 0 || (name == "max-tokens" && parsed == 0) {
			return fmt.Errorf("%s must be %s", name, map[bool]string{true: "greater than 0", false: "0 or greater"}[name == "max-tokens"])
		}
		if name == "top-k" {
			cfg.TopK = parsed
		} else {
			cfg.MaxNewTokens = parsed
		}
		return nil
	default:
		return fmt.Errorf("unknown setting %q", name)
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

In chat: /settings, /set, /clear, /help, /bye.

Examples:
  oxidize chat ./models/model.gguf
  oxidize chat Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf

Common options: --max-tokens, --temperature, --top-p, --top-k, --backend, --threads, --ctx-size, --draft-model, --file`)
}
