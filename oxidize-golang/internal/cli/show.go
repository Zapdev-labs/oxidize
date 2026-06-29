package cli

import (
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/internal/gguf"
)

func showCommand(args []string, stdout io.Writer) error {
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		_, _ = fmt.Fprintln(stdout, `Usage: oxidize show MODEL [options]

Show model metadata in a readable summary (like ollama show).

Options:
  --file NAME    GGUF filename for Hugging Face repos
  --verbose      include tensor count and file path`)
		return nil
	}

	fs := flag.NewFlagSet("show", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	hfFile := fs.String("file", "", "GGUF filename for Hugging Face repos")
	verbose := fs.Bool("verbose", false, "show extra details")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		return fmt.Errorf("oxidize show requires a model name or .gguf path")
	}

	path, err := resolveModelPathWithHF(fs.Arg(0), *hfFile)
	if err != nil {
		return err
	}
	if err := validateGGUFPath(path); err != nil {
		return err
	}

	file, err := gguf.LoadFile(path)
	if err != nil {
		return err
	}

	meta := file.Metadata
	name := fs.Arg(0)
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		name = strings.TrimSuffix(filepathBase(path), ".gguf")
	}

	rows := [][]string{
		{"", "name", name},
		{"", "file", path},
		{"", "format", "gguf"},
	}
	if arch := metaString(meta, "general.architecture"); arch != "" {
		rows = append(rows, []string{"", "architecture", arch})
	}
	if quant := metaString(meta, "general.file_type"); quant != "" {
		rows = append(rows, []string{"", "quantization", quant})
	}
	if params := metaString(meta, "general.parameter_count"); params != "" {
		rows = append(rows, []string{"", "parameters", params})
	} else if size := metaString(meta, "general.size_label"); size != "" {
		rows = append(rows, []string{"", "parameters", size})
	}
	if ctx := metaString(meta, "llama.context_length"); ctx != "" {
		rows = append(rows, []string{"", "context length", ctx})
	} else if ctx := metaString(meta, "qwen2.context_length"); ctx != "" {
		rows = append(rows, []string{"", "context length", ctx})
	}
	if emb := metaString(meta, "general.embedding_length"); emb != "" {
		rows = append(rows, []string{"", "embedding length", emb})
	}
	if layers := metaString(meta, "llama.block_count"); layers != "" {
		rows = append(rows, []string{"", "layers", layers})
	}

	if st, statErr := os.Stat(path); statErr == nil {
		rows = append(rows, []string{"", "size", humanBytes(st.Size())})
		rows = append(rows, []string{"", "modified", humanTime(st.ModTime(), "")})
	}
	rows = append(rows, []string{"", "id", modelDigest(path)})

	if *verbose {
		rows = append(rows, []string{"", "tensors", fmt.Sprintf("%d", len(file.TensorInfos))})
		rows = append(rows, []string{"", "gguf version", fmt.Sprintf("%d", file.Version)})
	}

	_, _ = fmt.Fprintln(stdout, " Model")
	renderTable(stdout, []string{"", "", ""}, rows)
	return nil
}

func metaString(meta map[string]gguf.MetadataValue, key string) string {
	v, ok := meta[key]
	if !ok {
		return ""
	}
	return formatMetadata(v)
}

func modelDigest(path string) string {
	h := sha256.Sum256([]byte(path))
	return hex.EncodeToString(h[:])[:12]
}

func filepathBase(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' || path[i] == '\\' {
			return path[i+1:]
		}
	}
	return path
}
