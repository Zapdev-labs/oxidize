// oxidize-c-tui: interactive chat TUI wrapping the oxidize-c engine.
// Stdlib only. Spawns `oxidize-c --chat` once (model stays loaded, KV cache
// carries across turns) and streams its output with ANSI colors.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	cReset = "\x1b[0m"
	cBold  = "\x1b[1m"
	cDim   = "\x1b[2m"
	cCyan  = "\x1b[36m"
	cGreen = "\x1b[32m"
)

func main() {
	model := flag.String("model", "", "path to .gguf model (required)")
	bin := flag.String("bin", "", "path to oxidize-c binary (default: next to this binary, then PATH)")
	system := flag.String("system", "", "system prompt")
	maxTokens := flag.Int("max-tokens", 512, "max tokens per reply")
	temp := flag.Float64("temp", 0.7, "temperature (0 = greedy)")
	topK := flag.Int("top-k", 40, "top-k")
	topP := flag.Float64("top-p", 0.95, "top-p")
	minP := flag.Float64("min-p", 0, "min-p (0 = off)")
	repeatPenalty := flag.Float64("repeat-penalty", 1.1, "repetition penalty (<=1 = off)")
	ctx := flag.Int("ctx", 4096, "context length")
	threads := flag.Int("threads", 0, "threads (0 = auto)")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "usage: oxidize-c-tui --model path.gguf [flags]")
		flag.PrintDefaults()
		os.Exit(1)
	}

	engine := *bin
	if engine == "" {
		if exe, err := os.Executable(); err == nil {
			cand := filepath.Join(filepath.Dir(exe), "..", "oxidize-c")
			if _, err := os.Stat(cand); err == nil {
				engine = cand
			}
		}
		if engine == "" {
			engine = "oxidize-c"
		}
	}

	args := []string{
		"--model", *model, "--chat",
		"--max-tokens", fmt.Sprint(*maxTokens),
		"--temp", fmt.Sprint(*temp),
		"--top-k", fmt.Sprint(*topK),
		"--top-p", fmt.Sprint(*topP),
		"--ctx", fmt.Sprint(*ctx),
		"--threads", fmt.Sprint(*threads),
	}
	if *minP > 0 {
		args = append(args, "--min-p", fmt.Sprint(*minP))
	}
	if *repeatPenalty > 1 {
		args = append(args, "--repeat-penalty", fmt.Sprint(*repeatPenalty))
	}
	if *system != "" {
		args = append(args, "--system", *system)
	}

	cmd := exec.Command(engine, args...)
	engineIn, err := cmd.StdinPipe()
	if err != nil {
		fatal(err)
	}
	engineOut, err := cmd.StdoutPipe()
	if err != nil {
		fatal(err)
	}
	engineErr, err := cmd.StderrPipe()
	if err != nil {
		fatal(err)
	}
	if err := cmd.Start(); err != nil {
		fatal(fmt.Errorf("starting %s: %w", engine, err))
	}

	// Engine stderr: stats lines "[...]" dimmed, everything else as-is.
	go func() {
		sc := bufio.NewScanner(engineErr)
		for sc.Scan() {
			line := sc.Text()
			if strings.HasPrefix(line, "[") {
				fmt.Fprintf(os.Stderr, "%s%s%s\n", cDim, line, cReset)
			} else {
				fmt.Fprintln(os.Stderr, line)
			}
		}
	}()

	name := strings.TrimSuffix(filepath.Base(*model), ".gguf")
	fmt.Printf("%s%s>>> oxidize-c · %s%s\n", cBold, cCyan, name, cReset)
	fmt.Printf("%sType a message. /bye to exit.%s\n", cDim, cReset)

	stdin := bufio.NewScanner(os.Stdin)
	out := bufio.NewReader(engineOut)
	for {
		fmt.Printf("\n%s%s❯%s ", cBold, cGreen, cReset)
		if !stdin.Scan() {
			fmt.Println()
			break
		}
		line := strings.TrimSpace(stdin.Text())
		if line == "" {
			continue
		}
		if line == "/bye" || line == "/exit" || line == "exit" || line == "quit" {
			break
		}
		if line == "/help" {
			fmt.Println("  /bye   exit")
			fmt.Println("  /help  this help")
			continue
		}
		if _, err := io.WriteString(engineIn, line+"\n"); err != nil {
			fatal(fmt.Errorf("engine died: %w", err))
		}
		// Stream reply until the 0x1e record separator.
		for {
			b, err := out.ReadByte()
			if err != nil {
				fmt.Println()
				fatal(fmt.Errorf("engine closed: %w", err))
			}
			if b == 0x1e {
				out.ReadByte() // trailing newline
				break
			}
			os.Stdout.Write([]byte{b})
		}
	}
	engineIn.Close()
	cmd.Wait()
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "oxidize-c-tui: %v\n", err)
	os.Exit(1)
}
