package cli

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func rmCommand(args []string, stdout io.Writer) error {
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		_, _ = fmt.Fprintln(stdout, `Usage: oxidize rm MODEL [MODEL...] [options]

Remove local GGUF models from the models directory.

Examples:
  oxidize rm qwen2.5-0.5b
  oxidize rm ./models/old-model.gguf`)
		return nil
	}

	fs := flag.NewFlagSet("rm", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	modelsDir := fs.String("models-dir", "", "models directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		return fmt.Errorf("oxidize rm requires at least one model name")
	}

	dir := resolveModelsDir(*modelsDir)
	for _, name := range fs.Args() {
		path, err := resolveDeletePath(name, dir)
		if err != nil {
			return err
		}
		if err := os.Remove(path); err != nil {
			return fmt.Errorf("rm %q: %w", name, err)
		}
		_, _ = fmt.Fprintf(stdout, "deleted %q\n", name)
	}
	return nil
}

func resolveDeletePath(name, modelsDir string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return "", fmt.Errorf("empty model name")
	}
	if strings.HasSuffix(strings.ToLower(name), ".gguf") {
		if _, err := os.Stat(name); err != nil {
			return "", fmt.Errorf("model %q not found", name)
		}
		return name, nil
	}
	if modelsDir == "" {
		return "", fmt.Errorf("models directory not found; use --models-dir")
	}
	candidate := filepath.Join(modelsDir, name+".gguf")
	if _, err := os.Stat(candidate); err != nil {
		return "", fmt.Errorf("model %q not found in %s", name, modelsDir)
	}
	return candidate, nil
}
