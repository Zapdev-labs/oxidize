package model

import (
	"context"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
)

// HasMTPWeights reports whether a GGUF file contains MTP/nextn tensors.
func HasMTPWeights(path string) bool {
	mapped, err := ggufcore.LoadMapped(path)
	if err != nil {
		return false
	}
	for _, t := range mapped.Parsed.TensorInfos {
		n := strings.ToLower(t.Name)
		if strings.Contains(n, "nextn") || strings.Contains(n, "mtp") {
			return true
		}
	}
	return false
}

// MtpGenerationStream uses in-GGUF MTP heads for multi-token draft steps.
type MtpGenerationStream struct {
	model   Model
	session *Session
	config  GenerationConfig
	done    bool
	prompt  []Token
}

// NewMtpGenerationStream constructs an MTP-backed generation stream.
func NewMtpGenerationStream(model Model, session *Session, config GenerationConfig) *MtpGenerationStream {
	return &MtpGenerationStream{model: model, session: session, config: config}
}

// Seed sets the prompt tokens.
func (s *MtpGenerationStream) Seed(prompt []Token) {
	s.prompt = append([]Token(nil), prompt...)
}

// Next generates the next token (MTP-aware path uses the same forward as baseline today).
func (s *MtpGenerationStream) Next(ctx context.Context) (Token, bool, error) {
	if s.done {
		return 0, true, errGenerationFinished
	}
	if err := ctx.Err(); err != nil {
		return 0, true, &GenerationError{Message: err.Error()}
	}
	contextTokens := append([]Token(nil), s.prompt...)
	logits, err := s.model.Forward(contextTokens, s.session)
	if err != nil {
		return 0, true, &GenerationError{Message: err.Error()}
	}
	token, err := Sample(logits, s.config.Sampling, nil)
	if err != nil {
		return 0, true, err
	}
	if token == s.config.StopToken {
		s.done = true
		return token, true, nil
	}
	s.prompt = append(s.prompt, token)
	if len(s.prompt) >= s.config.MaxNewTokens {
		s.done = true
	}
	return token, s.done, nil
}
