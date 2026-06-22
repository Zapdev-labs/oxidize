package cli

import (
	"fmt"
	"io"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
	"github.com/charmbracelet/lipgloss"
)

var (
	styleTitle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("86")).
			MarginBottom(1)

	styleSubtitle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("244"))

	styleCmd = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("212"))

	styleDim = lipgloss.NewStyle().
			Foreground(lipgloss.Color("240"))

	styleSuccess = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("42"))

	styleHeader = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("99")).
			BorderStyle(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("99")).
			Padding(0, 2)
)

func printBanner(w io.Writer) {
	banner := styleTitle.Render("oxidize") + " " +
		styleSubtitle.Render("local-first LLM inference")
	_, _ = fmt.Fprintln(w, banner)
	_, _ = fmt.Fprintln(w)
}

func printCommandRow(w io.Writer, name, desc string) {
	_, _ = fmt.Fprintf(w, "  %s  %s\n",
		styleCmd.Render(padRight(name, 22)),
		styleDim.Render(desc),
	)
}

func padRight(s string, width int) string {
	if len(s) >= width {
		return s
	}
	return s + strings.Repeat(" ", width-len(s))
}

func printSection(w io.Writer, title string) {
	_, _ = fmt.Fprintln(w, styleSubtitle.Render(title))
}

func kernelLabel() string {
	if quantization.OxkHasAVX2() {
		return "OXK (C++ AVX2)"
	}
	return "Go (scalar)"
}

func printBenchHeader(w io.Writer, model, engine string, iterations, maxTokens int) {
	_, _ = fmt.Fprintln(w, styleHeader.Render("Benchmark"))
	_, _ = fmt.Fprintf(w, "  model       %s\n", styleCmd.Render(model))
	_, _ = fmt.Fprintf(w, "  engine      %s\n", engine)
	_, _ = fmt.Fprintf(w, "  iterations  %d\n", iterations)
	_, _ = fmt.Fprintf(w, "  max_tokens  %d\n", maxTokens)
	_, _ = fmt.Fprintf(w, "  kernels     %s\n", styleSuccess.Render(kernelLabel()))
	_, _ = fmt.Fprintln(w)
}

func printBenchRound(w io.Writer, round, tokens int, elapsed, speed float64) {
	_, _ = fmt.Fprintf(w, "  round %d  %s  %s  %s\n",
		round,
		styleDim.Render(fmt.Sprintf("%d tokens", tokens)),
		styleDim.Render(fmt.Sprintf("%.2fs", elapsed)),
		styleSuccess.Render(fmt.Sprintf("%.1f tok/s", speed)),
	)
}

func printBenchAverage(w io.Writer, avg float64, totalTokens int) {
	_, _ = fmt.Fprintln(w)
	_, _ = fmt.Fprintf(w, "  %s  %s over %d tokens\n",
		styleTitle.Render("average"),
		styleSuccess.Render(fmt.Sprintf("%.2f tok/s", avg)),
		totalTokens,
	)
}
