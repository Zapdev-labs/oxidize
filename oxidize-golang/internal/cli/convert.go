package cli

import (
	"flag"
	"fmt"
	"io"

	"github.com/Zapdev-labs/oxidize/golang/core/convert"
)

func convertCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("convert", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	input := fs.String("input", "", "input SafeTensors file or directory")
	output := fs.String("output", "", "output GGUF path")
	arch := fs.String("arch", "", "architecture override")
	config := fs.String("config", "", "config.json path")
	noMap := fs.Bool("no-map-hf-names", false, "skip HF tensor name mapping")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *input == "" || *output == "" {
		_, _ = fmt.Fprintln(stdout, "usage: oxidize convert --input in.safetensors --output out.gguf")
		return fmt.Errorf("convert: --input and --output are required")
	}
	cfg := convert.Config{
		InputPath:       *input,
		OutputPath:      *output,
		ArchOverride:    *arch,
		MapHFTensorName: !*noMap,
		ConfigPath:      *config,
	}
	if err := convert.ConvertSafeTensorsToGGUF(cfg); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "wrote %s\n", *output)
	return nil
}
