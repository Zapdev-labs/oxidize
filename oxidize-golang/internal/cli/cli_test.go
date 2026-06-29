package cli

import (
	"bytes"
	"context"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Zapdev-labs/oxidize/golang/internal/testutil"
)

func TestLegacyPromptFlag(t *testing.T) {
	testutil.RequireSlowTests(t)
	model := testutil.QwenModelPath(t)
	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{
		"--prompt", "Write a Python function that returns the factorial of n.",
		"--model", model,
	}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	testutil.AssertGenerationText(t, stdout.String())
}

func TestRunCommandQwenPrompt(t *testing.T) {
	testutil.RequireSlowTests(t)
	model := testutil.QwenModelPath(t)
	var stdout bytes.Buffer
	err := Run(context.Background(), []string{
		"run", model, "Write a Python function that returns the factorial of n.",
		"--max-tokens", "32", "--temperature", "0.7", "--top-p", "0.9",
	}, &stdout, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	testutil.AssertGenerationText(t, stdout.String())
}

func TestListCommand(t *testing.T) {
	dir := t.TempDir()
	testutil.LinkQwenModel(t, dir)

	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{"list", "--models-dir", dir}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	got := stdout.String()
	if got == "" {
		t.Fatal("expected list output")
	}
	if !strings.Contains(got, testutil.QwenModelID) {
		t.Fatalf("expected %q in list output: %q", testutil.QwenModelID, got)
	}
}

func TestRunCommandRejectsMissingModel(t *testing.T) {
	var stderr bytes.Buffer
	err := Run(context.Background(), []string{"run"}, &bytes.Buffer{}, &stderr)
	if err == nil {
		t.Fatal("expected missing model error")
	}
	if !strings.Contains(err.Error(), "requires a model name or local .gguf path") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestHelpCommand(t *testing.T) {
	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{"help"}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(stdout.String(), "serve") {
		t.Fatal("expected serve in help output")
	}
	if !strings.Contains(stdout.String(), "chat") {
		t.Fatal("expected chat in help output")
	}
}

func TestVersionCommand(t *testing.T) {
	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{"version"}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(stdout.String(), "0.1.0") {
		t.Fatalf("expected version in output: %q", stdout.String())
	}
}

func TestRootHelpFlag(t *testing.T) {
	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{"--help"}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(stdout.String(), "Available Commands:") {
		t.Fatalf("expected cobra help: %q", stdout.String())
	}
}

func TestRootVersionFlag(t *testing.T) {
	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{"-v"}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(stdout.String(), "0.1.0") {
		t.Fatalf("expected version: %q", stdout.String())
	}
}

func TestShowCommand(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "encoded.gguf")
	testutil.WriteEncodedGGUF(t, path)

	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{"show", path}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	got := stdout.String()
	if !strings.Contains(got, "Model") {
		t.Fatalf("expected model summary: %q", got)
	}
	if !strings.Contains(got, "architecture") {
		t.Fatalf("expected architecture in show output: %q", got)
	}
}

func TestRmCommand(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "temp.gguf")
	testutil.WriteEncodedGGUF(t, path)

	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{"rm", path}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("expected file removed")
	}
}

func TestCpCommand(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.gguf")
	dstDir := filepath.Join(dir, "models")
	testutil.WriteEncodedGGUF(t, src)

	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{"cp", src, filepath.Join(dstDir, "copy.gguf")}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dstDir, "copy.gguf")); err != nil {
		t.Fatalf("copy missing: %v", err)
	}
}

func TestInspectCommand(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "encoded.gguf")
	testutil.WriteEncodedGGUF(t, path)

	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{"inspect", path}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(stdout.String(), "Tensors in") {
		t.Fatalf("unexpected inspect output: %q", stdout.String())
	}
}

func TestParseGenFlagsBackendAndTopK(t *testing.T) {
	_, opts, _, rest, err := parseGenFlags("run", []string{
		"--backend", "cuda",
		"--top-k", "40",
		"--ctx-size", "4096",
		"--draft-tokens", "8",
		"model.gguf", "hi",
	})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if opts.Backend != "cuda" || opts.TopK != 40 || opts.CtxSize != 4096 || opts.DraftTokens != 8 {
		t.Fatalf("opts=%+v", opts)
	}
	if len(rest) != 1 || rest[0] != "model.gguf" {
		t.Fatalf("rest=%v prompt=%q", rest, opts.Prompt)
	}
	if opts.Prompt != "hi" {
		t.Fatalf("prompt=%q", opts.Prompt)
	}
}

func TestValidateBenchEngine(t *testing.T) {
	if err := validateBenchEngine("inference"); err != nil {
		t.Fatal(err)
	}
	if err := validateBenchEngine("dflash"); err != nil {
		t.Fatal(err)
	}
	if err := validateBenchEngine("other"); err == nil {
		t.Fatal("expected error")
	}
}

func TestServeCommandHonorsContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	time.AfterFunc(150*time.Millisecond, cancel)

	dir := t.TempDir()
	err := Run(ctx, []string{"serve", "--host", "127.0.0.1", "--port", "0", "--models-dir", dir}, &bytes.Buffer{}, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
}

func TestParseBenchEngine(t *testing.T) {
	engine, rest, err := parseBenchEngine([]string{"--engine", "dflash", "--iterations", "2", "m.gguf"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if engine != "dflash" {
		t.Fatalf("engine=%q", engine)
	}
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	iter := fs.Int("iterations", 3, "")
	if err := fs.Parse(rest); err != nil {
		t.Fatal(err)
	}
	if *iter != 2 || fs.Arg(0) != "m.gguf" {
		t.Fatalf("rest parse: iter=%d arg=%s", *iter, fs.Arg(0))
	}
}
