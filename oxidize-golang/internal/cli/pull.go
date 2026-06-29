package cli

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/hf"
)

func pullCommand(args []string, stdout, stderr io.Writer) error {
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		_, _ = fmt.Fprintln(stdout, `Usage: oxidize pull MODEL [options]

Download a GGUF model from Hugging Face.

Examples:
  oxidize pull Qwen/Qwen2.5-0.5B-Instruct-GGUF
  oxidize pull Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf
  oxidize pull org/model --models-dir ./models

Options:
  --file NAME        specific .gguf filename in the repo
  --models-dir DIR   also link the model into this directory`)
		return nil
	}

	fs := flag.NewFlagSet("pull", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	hfFile := fs.String("file", "", "GGUF filename in the Hugging Face repo")
	modelsDir := fs.String("models-dir", "", "also link into models directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		return fmt.Errorf("oxidize pull requires a model name (e.g. org/model-GGUF)")
	}
	repo := strings.TrimSpace(fs.Arg(0))
	if repo == "" {
		return fmt.Errorf("oxidize pull requires a model name")
	}

	_, _ = fmt.Fprintf(stderr, "pulling %s...\n", repo)

	guessName := pathOrGuess(repo, *hfFile)
	var lastPct int
	path, err := hf.ResolveGGUF(hf.ResolveOptions{
		Repo:     repo,
		Filename: *hfFile,
		OnProgress: func(downloaded, total int64) {
			pct := pullPercent(downloaded, total)
			if pct == lastPct && downloaded < total {
				return
			}
			lastPct = pct
			_, _ = fmt.Fprintf(stderr, "\rpulling %s: %3d%% %s / %s",
				guessName,
				pct,
				humanBytes(downloaded),
				humanBytes(total),
			)
		},
	})
	if err != nil {
		_, _ = fmt.Fprintln(stderr)
		return err
	}
	_, _ = fmt.Fprintf(stderr, "\rpulling %s: 100%% %s\n", filepath.Base(path), humanBytes(fileSize(path)))

	targetDir := resolveModelsDir(*modelsDir)
	if targetDir != "" {
		name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		if link := filepath.Join(targetDir, name+".gguf"); link != path {
			if err := linkModelIntoDir(path, link); err != nil {
				_, _ = fmt.Fprintf(stderr, "warning: could not link into %s: %v\n", targetDir, err)
			} else {
				_, _ = fmt.Fprintf(stderr, "linked to %s\n", link)
			}
		}
	}

	_, _ = fmt.Fprintf(stdout, "success %s\n", path)
	return nil
}

func pathOrGuess(repo, file string) string {
	if file != "" {
		return file
	}
	parts := strings.Split(repo, "/")
	if len(parts) >= 3 && strings.HasSuffix(strings.ToLower(parts[len(parts)-1]), ".gguf") {
		return parts[len(parts)-1]
	}
	return repo
}

func fileSize(path string) int64 {
	st, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return st.Size()
}

func linkModelIntoDir(src, dest string) error {
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	_ = os.Remove(dest)
	return os.Link(src, dest)
}
