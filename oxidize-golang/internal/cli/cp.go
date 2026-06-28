package cli

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func cpCommand(args []string, stdout io.Writer) error {
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		_, _ = fmt.Fprintln(stdout, `Usage: oxidize cp SOURCE DESTINATION

Copy a GGUF model file. DESTINATION can be a path or a name in ./models.

Examples:
  oxidize cp ~/.cache/oxidize/hf/org_model/model.gguf ./models/my-model.gguf
  oxidize cp ./old.gguf new-name`)
		return nil
	}

	fs := flag.NewFlagSet("cp", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	modelsDir := fs.String("models-dir", "", "models directory for short names")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("oxidize cp requires SOURCE and DESTINATION")
	}

	src, err := resolveModelPathWithHF(fs.Arg(0), "")
	if err != nil {
		return err
	}
	if err := validateGGUFPath(src); err != nil {
		return err
	}

	dest := strings.TrimSpace(fs.Arg(1))
	if !strings.HasSuffix(strings.ToLower(dest), ".gguf") {
		dir := resolveModelsDir(*modelsDir)
		if dir == "" {
			return fmt.Errorf("destination %q is not a .gguf path and no models directory found", dest)
		}
		dest = filepath.Join(dir, dest+".gguf")
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	if err := copyFile(src, dest); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "copied %q to %q\n", fs.Arg(0), dest)
	return nil
}

func copyFile(src, dest string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()
	if _, err := io.Copy(out, in); err != nil {
		_ = os.Remove(dest)
		return err
	}
	return out.Close()
}
