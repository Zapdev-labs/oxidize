package cli

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
	"github.com/Zapdev-labs/oxidize/golang/internal/gguf"
)

func inspectCommand(args []string, stdout io.Writer) error {
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		_, _ = fmt.Fprintln(stdout, "Usage: oxidize inspect <model> [--file NAME]")
		return nil
	}
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	hfFile := fs.String("file", "", "GGUF filename for Hugging Face repos")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		return fmt.Errorf("oxidize inspect requires a .gguf path")
	}
	path, err := resolveModelPathWithHF(fs.Arg(0), *hfFile)
	if err != nil {
		return err
	}
	file, err := gguf.LoadFile(path)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(stdout, "Metadata in %s:\n", path); err != nil {
		return err
	}
	keys := make([]string, 0, len(file.Metadata))
	for key := range file.Metadata {
		keys = append(keys, key)
	}
	sortStrings(keys)
	for _, key := range keys {
		if _, err := fmt.Fprintf(stdout, "  %s = %s\n", key, formatMetadata(file.Metadata[key])); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(stdout, "\nTensors in %s:\n", path); err != nil {
		return err
	}
	for _, tensor := range file.TensorInfos {
		qtype := quantization.FromGGMLType(tensor.GGMLType)
		qsize, err := gguf.TensorStorageBytes(tensor)
		if err != nil {
			qsize = 0
		}
		if _, err := fmt.Fprintf(
			stdout,
			"  %s dims=%v type=%v offset=%d qsize=%d\n",
			tensor.Name,
			tensor.Dimensions,
			qtype,
			tensor.RelativeOffset,
			qsize,
		); err != nil {
			return err
		}
	}
	stat, err := os.Stat(path)
	if err == nil {
		_, _ = fmt.Fprintf(stdout, "\nfile_size=%d bytes\n", stat.Size())
	}
	return nil
}

func formatMetadata(v gguf.MetadataValue) string {
	if v.String != "" {
		return fmt.Sprintf("%q", v.String)
	}
	if len(v.Array) > 0 {
		parts := make([]string, 0, len(v.Array))
		for _, item := range v.Array {
			parts = append(parts, formatMetadata(item))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	}
	if v.Bool {
		return "true"
	}
	if v.Float64 != 0 || v.Type == gguf.MetadataFloat32 || v.Type == gguf.MetadataFloat64 {
		return fmt.Sprintf("%g", v.Float64)
	}
	if v.Int64 != 0 || v.Type == gguf.MetadataInt8 || v.Type == gguf.MetadataInt16 || v.Type == gguf.MetadataInt32 || v.Type == gguf.MetadataInt64 {
		return fmt.Sprintf("%d", v.Int64)
	}
	return fmt.Sprintf("%d", v.Uint64)
}

func sortStrings(values []string) {
	for i := 1; i < len(values); i++ {
		j := i
		for j > 0 && values[j-1] > values[j] {
			values[j-1], values[j] = values[j], values[j-1]
			j--
		}
	}
}
