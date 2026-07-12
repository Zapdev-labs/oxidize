package cli

import (
	"bytes"
	"context"
	"io"
	"strings"
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
)

func Test_ChatREPL_updates_settings_and_reports_invalid_values(t *testing.T) {
	// Given
	cfg := generate.DefaultRunConfig()
	cfg.ModelPath = "/models/demo.gguf"
	input := strings.NewReader("/settings\n/set temperature 0.7\n/set top-p nope\n/settings\n/bye\n")
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	// When
	err := chatREPLWithIO(context.Background(), cfg, input, &stdout, &stderr, false, nil)

	// Then
	if err != nil {
		t.Fatalf("chatREPLWithIO() error = %v", err)
	}
	output := stdout.String()
	if !strings.Contains(output, "temperature=0.80") || !strings.Contains(output, "temperature=0.70") {
		t.Fatalf("settings output missing before/after values:\n%s", output)
	}
	if !strings.Contains(stderr.String(), "top-p must be a number") {
		t.Fatalf("stderr missing invalid-value error: %s", stderr.String())
	}
}

func Test_ChatREPL_handles_commands_without_generating(t *testing.T) {
	// Given
	cfg := generate.DefaultRunConfig()
	cfg.ModelPath = "demo.gguf"
	input := strings.NewReader("/help\n/clear\n/unknown\n/bye\n")
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	generated := 0

	// When
	err := chatREPLWithIO(context.Background(), cfg, input, &stdout, &stderr, false,
		func(context.Context, generate.RunConfig, io.Writer) error {
			generated++
			return nil
		})

	// Then
	if err != nil {
		t.Fatalf("chatREPLWithIO() error = %v", err)
	}
	if generated != 0 {
		t.Fatalf("generated %d times for slash commands", generated)
	}
	if !strings.Contains(stdout.String(), "/set temperature") || !strings.Contains(stdout.String(), "clear") {
		t.Fatalf("command output incomplete:\n%s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "unknown command") {
		t.Fatalf("stderr missing unknown-command error: %s", stderr.String())
	}
}
