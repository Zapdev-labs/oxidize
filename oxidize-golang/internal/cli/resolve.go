package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/hf"
)

func resolveModelPathWithHF(nameOrPath, hfFile string) (string, error) {
	if strings.HasSuffix(strings.ToLower(nameOrPath), ".gguf") {
		return nameOrPath, nil
	}
	if _, err := os.Stat(nameOrPath); err == nil {
		return nameOrPath, nil
	}
	if strings.Contains(nameOrPath, "/") {
		path, err := hf.ResolveGGUF(hf.ResolveOptions{Repo: nameOrPath, Filename: hfFile})
		if err == nil {
			return path, nil
		}
		if hfFile != "" || !strings.Contains(err.Error(), "multiple .gguf") {
			return "", err
		}
	}
	dir := resolveModelsDir("")
	if dir == "" {
		return nameOrPath, nil
	}
	candidate := filepath.Join(dir, nameOrPath+".gguf")
	if _, err := os.Stat(candidate); err == nil {
		return candidate, nil
	}
	return nameOrPath, nil
}

func resolveModelPath(nameOrPath string) string {
	path, err := resolveModelPathWithHF(nameOrPath, "")
	if err != nil {
		return nameOrPath
	}
	return path
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

func validateGGUFPath(path string) error {
	if !strings.HasSuffix(strings.ToLower(path), ".gguf") {
		return fmt.Errorf("expected a .gguf path, got %q", path)
	}
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("model not found: %w", err)
	}
	return nil
}
