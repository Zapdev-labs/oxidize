package cli

import (
	"context"
	"io"

	"github.com/Zapdev-labs/oxidize/golang/core/model"
	"github.com/Zapdev-labs/oxidize/golang/core/tokenizer"
	"github.com/Zapdev-labs/oxidize/golang/internal/generate"
)

func loadInference(path string, loader generate.LoaderConfig) (*model.InferenceModel, generate.CachedModel, error) {
	return generate.InferenceFromCache(path, loader)
}

func loadTokenizerForCLI(modelPath, tokenizerPath string) (tokenizer.Tokenizer, error) {
	if stringsTrim(tokenizerPath) != "" {
		return tokenizer.LoadFromGGUFFile(tokenizerPath)
	}
	tok, err := tokenizer.LoadFromGGUFFile(modelPath)
	if err != nil {
		return tokenizer.NewBpeTokenizer(nil, nil, tokenizer.SpecialTokens{BOS: 1, EOS: 2}), nil
	}
	return tok, nil
}

func encodeOpts() tokenizer.EncodeOptions { return tokenizer.EncodeOptions{} }

func stringsTrim(s string) string {
	for len(s) > 0 && (s[0] == ' ' || s[0] == '\t') {
		s = s[1:]
	}
	for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\t') {
		s = s[:len(s)-1]
	}
	return s
}

func generateRun(ctx context.Context, cfg generate.RunConfig, stdout, stderr io.Writer) error {
	return generate.RunFromGGUF(ctx, cfg, stdout)
}
