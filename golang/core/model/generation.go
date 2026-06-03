package model

import (
	"context"
	"errors"
	"sync"
)

// GenerationConfig mirrors GenerationConfig.
type GenerationConfig struct {
	MaxNewTokens     int
	StopToken        Token
	StopSequences    [][]Token
	PrefillBatchSize int
	Sampling         SamplingConfig
	SuppressedTokens []Token
}

// DefaultGenerationConfig returns sensible defaults.
func DefaultGenerationConfig() GenerationConfig {
	return GenerationConfig{
		MaxNewTokens:     128,
		StopToken:        2,
		StopSequences:    nil,
		PrefillBatchSize: 1,
		Sampling:         DefaultSamplingConfig(),
		SuppressedTokens: nil,
	}
}

// GenerationError mirrors GenerationError.
type GenerationError struct{ Message string }

func (e *GenerationError) Error() string { return "generation: " + e.Message }

// IsModelError reports whether the error wraps a model error.
func (e *GenerationError) IsModelError() bool { return false }

// SpeculativeGenerationConfig mirrors SpeculativeGenerationConfig.
type SpeculativeGenerationConfig struct {
	Generation         GenerationConfig
	DraftTokensPerStep int
}

// DefaultSpeculativeGenerationConfig returns sensible defaults.
func DefaultSpeculativeGenerationConfig() SpeculativeGenerationConfig {
	return SpeculativeGenerationConfig{
		Generation:         DefaultGenerationConfig(),
		DraftTokensPerStep: 4,
	}
}

// GenerationStream is a small in-process representation of the async stream
// that the Rust crate returns from `Model::generate`. It blocks on the
// supplied model + session for each token.
type GenerationStream struct {
	mu      sync.Mutex
	model   Model
	session *Session
	config  GenerationConfig
	done    bool
	tokens  []Token
}

// NewGenerationStream constructs a stream bound to the given model and
// session.
func NewGenerationStream(model Model, session *Session, config GenerationConfig) *GenerationStream {
	return &GenerationStream{model: model, session: session, config: config}
}

// Seed prepends tokens to the stream's internal state.
func (s *GenerationStream) Seed(prompt []Token) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.tokens) == 0 {
		s.tokens = append(s.tokens, prompt...)
	}
}

// Next produces the next token. When generation is finished, returns
// (token=0, done=true, nil). Errors are wrapped in GenerationError.
func (s *GenerationStream) Next(ctx context.Context) (Token, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		return 0, true, nil
	}
	if s.config.MaxNewTokens > 0 && len(s.tokens) >= s.config.MaxNewTokens {
		s.done = true
		return 0, true, nil
	}
	if err := ctx.Err(); err != nil {
		return 0, true, &GenerationError{Message: err.Error()}
	}
	logits, err := s.model.Forward(s.tokens, s.session)
	if err != nil {
		return 0, true, &GenerationError{Message: err.Error()}
	}
	if len(logits) == 0 {
		return 0, true, &GenerationError{Message: "empty logits"}
	}
	tok, err := Sample(logits, s.config.Sampling, nil)
	if err != nil {
		return 0, true, &GenerationError{Message: err.Error()}
	}
	s.tokens = append(s.tokens, tok)
	if tok == s.config.StopToken {
		s.done = true
		return tok, true, nil
	}
	for _, seq := range s.config.StopSequences {
		if len(seq) == 0 {
			continue
		}
		if len(s.tokens) >= len(seq) {
			ok := true
			for i, x := range seq {
				if s.tokens[len(s.tokens)-len(seq)+i] != x {
					ok = false
					break
				}
			}
			if ok {
				s.done = true
				return tok, true, nil
			}
		}
	}
	return tok, false, nil
}

// SpeculativeGenerationStream mirrors SpeculativeGenerationStream<'a, T: Model>.
type SpeculativeGenerationStream struct {
	mu      sync.Mutex
	draft   Model
	target  Model
	session *Session
	config  SpeculativeGenerationConfig
	done    bool
}

// NewSpeculativeGenerationStream constructs a speculative stream using a
// draft + target model.
func NewSpeculativeGenerationStream(draft, target Model, session *Session, config SpeculativeGenerationConfig) *SpeculativeGenerationStream {
	return &SpeculativeGenerationStream{draft: draft, target: target, session: session, config: config}
}

// Next returns the next accepted draft token plus a done flag.
func (s *SpeculativeGenerationStream) Next(ctx context.Context) (Token, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		return 0, true, nil
	}
	if err := ctx.Err(); err != nil {
		return 0, true, &GenerationError{Message: err.Error()}
	}
	tokens, accepted, err := SpeculativeDecode(s.draft, s.target, s.config.DraftTokensPerStep, s.session)
	if err != nil {
		return 0, true, &GenerationError{Message: err.Error()}
	}
	if accepted == 0 {
		s.done = true
		return 0, true, nil
	}
	// `accepted == 0` is unreachable here (handled above), so the stream is
	// not finished after returning an accepted token.
	return tokens[accepted-1], false, nil
}

// errGenerationFinished is returned when a stream has been fully consumed.
var errGenerationFinished = errors.New("generation finished")

// IsFinished reports whether err is errGenerationFinished.
func IsFinished(err error) bool { return errors.Is(err, errGenerationFinished) }
