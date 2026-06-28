package cli

import (
	"fmt"

	"github.com/Zapdev-labs/oxidize/golang/internal/buildinfo"
	"github.com/spf13/cobra"
)

func registerCommands(root *cobra.Command) {
	root.AddCommand(
		newRunCommand(),
		newPullCommand(),
		newShowCommand(),
		newListCommand(),
		newServeCommand(),
		newChatCommand(),
		newBenchCommand(),
		newInspectCommand(),
		newConvertCommand(),
		newCopyCommand(),
		newRmCommand(),
		newGPUClusterCommand(),
		newVersionCommand(),
	)
}

func newRunCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "run MODEL [PROMPT]",
		Short: "Run a model",
		Long: `Run a model with an optional prompt. Without a prompt, starts an
interactive session (like ollama run). Pipe stdin for non-interactive use.`,
		Example: `  oxidize run ./models/model.gguf "hello"
  oxidize run qwen2.5-0.5b
  echo "summarize this" | oxidize run model.gguf`,
		DisableFlagParsing: true,
		Args:               cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runCommand(cmd.Context(), args, cmd.OutOrStdout(), cmd.ErrOrStderr())
		},
	}
}

func newPullCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "pull MODEL",
		Short: "Pull a model from Hugging Face",
		Long:  "Download a GGUF file from a Hugging Face repo into the local cache.",
		Example: `  oxidize pull Qwen/Qwen2.5-0.5B-Instruct-GGUF
  oxidize pull org/model --file weights-q4_k_m.gguf --models-dir ./models`,
		DisableFlagParsing: true,
		Args:               cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return pullCommand(args, cmd.OutOrStdout(), cmd.ErrOrStderr())
		},
	}
}

func newShowCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "show MODEL",
		Short:   "Show model information",
		Aliases: []string{"info"},
		Example: `  oxidize show ./models/model.gguf
  oxidize show qwen2.5-0.5b --verbose`,
		DisableFlagParsing: true,
		Args:               cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return showCommand(args, cmd.OutOrStdout())
		},
	}
}

func newChatCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "chat MODEL",
		Short: "Interactive chat REPL",
		Long:  "Start a multi-turn chat session. Alias for run MODEL without a prompt.",
		Example: `  oxidize chat ./models/model.gguf
  oxidize chat qwen2.5-0.5b`,
		DisableFlagParsing: true,
		Args:               cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return chatCommand(cmd.Context(), args, cmd.OutOrStdout(), cmd.ErrOrStderr())
		},
	}
}

func newBenchCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "bench MODEL",
		Short: "Decode throughput benchmark",
		Example: `  oxidize bench ./models/model.gguf
  oxidize bench model.gguf --iterations 5 --engine dflash`,
		DisableFlagParsing: true,
		Args:               cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return benchCommand(cmd.Context(), args, cmd.OutOrStdout())
		},
	}
}

func newInspectCommand() *cobra.Command {
	return &cobra.Command{
		Use:    "inspect MODEL",
		Short:  "Print raw GGUF metadata and tensors",
		Hidden: true,
		DisableFlagParsing: true,
		Args:               cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return inspectCommand(args, cmd.OutOrStdout())
		},
	}
}

func newListCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List models",
		Long:    "List GGUF models in the models directory (default: ./models).",
		Example: `  oxidize list
  oxidize list qwen`,
		DisableFlagParsing: true,
		Args:               cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return listCommand(args, cmd.OutOrStdout())
		},
	}
}

func newServeCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start the API server",
		Long:    "Start the OpenAI-compatible HTTP API (like ollama serve).",
		Example: `  oxidize serve
  oxidize serve ./models/model.gguf --host 0.0.0.0 --port 11434`,
		DisableFlagParsing: true,
		Args:               cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return serveCommand(cmd.Context(), args)
		},
	}
}

func newConvertCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "convert",
		Short: "Convert SafeTensors to GGUF",
		Example: `  oxidize convert --input ./model.safetensors --output ./model.gguf`,
		DisableFlagParsing: true,
		Args:               cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return convertCommand(args, cmd.OutOrStdout())
		},
	}
}

func newCopyCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "cp SOURCE DESTINATION",
		Short:   "Copy a model",
		Aliases: []string{"copy"},
		Example: `  oxidize cp ~/.cache/oxidize/hf/demo/model.gguf ./models/demo.gguf
  oxidize cp ./old.gguf new-name`,
		DisableFlagParsing: true,
		Args:               cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cpCommand(args, cmd.OutOrStdout())
		},
	}
}

func newRmCommand() *cobra.Command {
	return &cobra.Command{
		Use:     "rm MODEL [MODEL...]",
		Short:   "Remove a model",
		Aliases: []string{"remove", "delete"},
		Example: `  oxidize rm qwen2.5-0.5b
  oxidize rm old-model another-model`,
		DisableFlagParsing: true,
		Args:               cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return rmCommand(args, cmd.OutOrStdout())
		},
	}
}

func newGPUClusterCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:    "gpu-cluster",
		Short:  "GPU cluster config generation and detection",
		Hidden: true,
	}

	cmd.AddCommand(
		&cobra.Command{
			Use:  "profiles",
			Args: cobra.NoArgs,
			RunE: func(c *cobra.Command, _ []string) error {
				return gpuClusterCommand([]string{"profiles"}, c.OutOrStdout(), c.ErrOrStderr())
			},
		},
		&cobra.Command{
			Use:  "detect",
			Args: cobra.NoArgs,
			RunE: func(c *cobra.Command, _ []string) error {
				return gpuClusterCommand([]string{"detect"}, c.OutOrStdout(), c.ErrOrStderr())
			},
		},
		&cobra.Command{
			Use:                "generate",
			DisableFlagParsing: true,
			Args:               cobra.ArbitraryArgs,
			RunE: func(c *cobra.Command, args []string) error {
				all := append([]string{"generate"}, args...)
				return gpuClusterCommand(all, c.OutOrStdout(), c.ErrOrStderr())
			},
		},
	)

	return cmd
}

func newVersionCommand() *cobra.Command {
	var short bool
	cmd := &cobra.Command{
		Use:   "version",
		Short: "Show version information",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if short {
				_, err := fmt.Fprintln(cmd.OutOrStdout(), buildinfo.Version)
				return err
			}
			_, err := fmt.Fprintf(cmd.OutOrStdout(), "%s %s (%s)\n",
				buildinfo.Name, buildinfo.Version, buildinfo.ModulePath)
			return err
		},
	}
	cmd.Flags().BoolVar(&short, "short", false, "print only the version number")
	return cmd
}
