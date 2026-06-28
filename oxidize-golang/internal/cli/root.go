package cli

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/internal/buildinfo"
	"github.com/spf13/cobra"
)

func Run(ctx context.Context, args []string, stdout io.Writer, stderr io.Writer) error {
	if shouldUseLegacy(args) {
		return runLegacy(ctx, args, stdout, stderr)
	}

	root := newRootCommand(stdout, stderr)
	root.SetArgs(args)
	return root.ExecuteContext(ctx)
}

func shouldUseLegacy(args []string) bool {
	for _, a := range args {
		switch a {
		case "--prompt", "-prompt":
			return true
		}
		if strings.HasPrefix(a, "--prompt=") {
			return true
		}
	}
	return false
}

func newRootCommand(stdout, stderr io.Writer) *cobra.Command {
	cobra.EnableCommandSorting = false

	root := &cobra.Command{
		Use:   "oxidize",
		Short: "Local-first LLM inference",
		Long: `Oxidize runs GGUF models locally with CPU and GPU backends.

Download models from Hugging Face, chat interactively, or serve an
OpenAI-compatible API — similar to ollama but for raw GGUF files.`,
		Example: `  oxidize run qwen2.5-0.5b "hello"
  oxidize pull Qwen/Qwen2.5-0.5B-Instruct-GGUF
  oxidize list
  oxidize show ./models/model.gguf
  oxidize serve`,
		SilenceUsage:  true,
		SilenceErrors: true,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if showVersion, _ := cmd.Flags().GetBool("version"); showVersion {
				_, err := fmt.Fprintf(cmd.OutOrStdout(), "%s %s\n", buildinfo.Name, buildinfo.Version)
				return err
			}
			return cmd.Help()
		},
	}

	root.SetOut(stdout)
	root.SetErr(stderr)
	root.CompletionOptions.DisableDefaultCmd = false
	root.Flags().BoolP("version", "v", false, "Show version information")

	registerCommands(root)
	return root
}
