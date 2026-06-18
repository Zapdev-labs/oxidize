package cli

import (
	"flag"
	"fmt"
)

type runOptions = genOptions

func parseRunFlags(name string, args []string) (*flag.FlagSet, runOptions, flagVisits, []string, error) {
	return parseGenFlags(name, args)
}

func requireModelArg(rest []string) (string, error) {
	if len(rest) == 0 {
		return "", fmt.Errorf("requires a model name or local .gguf path")
	}
	return rest[0], nil
}
