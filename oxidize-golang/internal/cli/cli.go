package cli

import (
	"context"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"

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
	case "list":
		return listCommand(args[1:], stdout)
	case "serve":
		return serveCommand(ctx, args[1:])
	default:
		_, _ = fmt.Fprintf(stderr, "unknown command: %s\n", args[0])
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
	_, err := io.WriteString(stdout, generate.CLITranscript(*prompt))
	return err
}

func listCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("list", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	modelsDir := fs.String("models-dir", "", "models directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	models, err := serviceinfo.DiscoverModels(*modelsDir)
	if err != nil {
		return err
	}
	if len(models) == 0 {
		_, err = io.WriteString(stdout, "oxidize-default\n")
		return err
	}
	for _, model := range models {
		line := fmt.Sprintf("%s\t%s\tversion=%d\tarch=%s\n", model.ID, filepath.Base(model.Path), model.Version, fallbackArch(model.Architecture))
		if _, writeErr := io.WriteString(stdout, line); writeErr != nil {
			return writeErr
		}
	}
	return nil
}

func serveCommand(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	host := fs.String("host", "127.0.0.1", "host")
	port := fs.Int("port", 8080, "port")
	modelsDir := fs.String("models-dir", "", "models directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	return server.Listen(ctx, server.Config{Host: *host, Port: *port, ModelsDir: *modelsDir})
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
