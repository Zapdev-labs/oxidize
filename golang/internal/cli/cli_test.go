package cli

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestLegacyPromptFlag(t *testing.T) {
	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{"--prompt", "ping"}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if stdout.String() == "" {
		t.Fatal("expected output")
	}
}

func TestListCommand(t *testing.T) {
	dir := t.TempDir()
	copyFixture(t, filepath.Join(dir, "valid-v3.gguf"))

	var stdout bytes.Buffer
	if err := Run(context.Background(), []string{"list", "--models-dir", dir}, &stdout, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if got := stdout.String(); got == "" {
		t.Fatal("expected list output")
	}
}

func TestRunCommandRejectsMissingModel(t *testing.T) {
	var stderr bytes.Buffer
	err := Run(context.Background(), []string{"run"}, &bytes.Buffer{}, &stderr)
	if err == nil {
		t.Fatal("expected missing model error")
	}
	if got := stderr.String(); got != "oxidize run requires a model name or local .gguf path\n" {
		t.Fatalf("unexpected stderr: %q", got)
	}
}

func TestServeCommandHonorsContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	time.AfterFunc(150*time.Millisecond, cancel)

	err := Run(ctx, []string{"serve", "--host", "127.0.0.1", "--port", "0"}, &bytes.Buffer{}, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
}
