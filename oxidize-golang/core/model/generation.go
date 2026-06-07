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

type generationState int

const (
	generationPrefill generationState = iota
	generationDecode
	generationDone
)

// GenerationStream is a small in-process representation of the async stream
// that the Rust crate returns from `Model::generate`. It blocks on the
// supplied model + session for each token.
type GenerationStream struct {
	mu      sync.Mutex
	model   Model
	session *Session
	config  GenerationConfig
	state   generationState
	done    bool
	prompt  []Token
	generated int
	lastToken Token
	hasLast   bool
	recent    []Token
	maxStop   int
}

// NewGenerationStream constructs a stream bound to the given model and
// session.
func NewGenerationStream(model Model, session *Session, config GenerationConfig) *GenerationStream {
	maxStop := 0
	for _, seq := range config.StopSequences {
		if len(seq) > maxStop {
			maxStop = len(seq)
		}
	}
	return &GenerationStream{
		model:   model,
		session: session,
		config:  config,
		state:   generationPrefill,
		recent:  make([]Token, 0, maxStop),
		maxStop: maxStop,
	}
}

// Seed prepends tokens to the stream's internal state.
func (s *GenerationStream) Seed(prompt []Token) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.prompt) == 0 && len(prompt) > 0 {
		s.prompt = append(s.prompt, prompt...)
	}
}

func (s *GenerationStream) forwardPrefill() (Logits, error) {
	batch := s.config.PrefillBatchSize
	if batch <= 0 {
		batch = 1
	}
	var logits Logits
	for i := 0; i < len(s.prompt); i += batch {
		end := i + batch
		if end > len(s.prompt) {
			end = len(s.prompt)
		}
		chunk := s.prompt[i:end]
		out, err := s.model.Forward(chunk, s.session)
		if err != nil {
			return nil, err
		}
		logits = out
	}
	return logits, nil
}

func (s *GenerationStream) suppressTokens(logits Logits) {
	for _, tok := range s.config.SuppressedTokens {
		if int(tok) < len(logits) {
			logits[tok] = negInfF32
		}
	}
}

func (s *GenerationStream) finishAfterToken(tok Token) bool {
	if tok == s.config.StopToken {
		return true
	}
	if s.maxStop == 0 {
		return false
	}
	s.recent = append(s.recent, tok)
	if len(s.recent) > s.maxStop {
		s.recent = s.recent[len(s.recent)-s.maxStop:]
	}
	for _, seq := range s.config.StopSequences {
		if len(seq) == 0 || len(s.recent) < len(seq) {
			continue
		}
		ok := true
		for i, x := range seq {
			if s.recent[len(s.recent)-len(seq)+i] != x {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

// Next produces the next token. When generation is finished, returns
// (token=0, done=true, nil). Errors are wrapped in GenerationError.
func (s *GenerationStream) Next(ctx context.Context) (Token, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done || s.state == generationDone {
		return 0, true, nil
	}
	if s.config.MaxNewTokens > 0 && s.generated >= s.config.MaxNewTokens {
		s.done = true
		s.state = generationDone
		return 0, true, nil
	}
	if err := ctx.Err(); err != nil {
		return 0, true, &GenerationError{Message: err.Error()}
	}

	var logits Logits
	var err error
	switch s.state {
	case generationPrefill:
		s.state = generationDecode
		logits, err = s.forwardPrefill()
	case generationDecode:
		if !s.hasLast {
			s.done = true
			s.state = generationDone
			return 0, true, nil
		}
		logits, err = s.model.Forward([]Token{s.lastToken}, s.session)
	default:
		return 0, true, nil
	}
	if err != nil {
		return 0, true, &GenerationError{Message: err.Error()}
	}
	if len(logits) == 0 {
		return 0, true, &GenerationError{Message: "empty logits"}
	}
	s.suppressTokens(logits)
	tok, err := Sample(logits, s.config.Sampling, nil)
	if err != nil {
		return 0, true, &GenerationError{Message: err.Error()}
	}
	s.generated++
	s.lastToken = tok
	s.hasLast = true
	if s.finishAfterToken(tok) {
		s.done = true
		s.state = generationDone
		return tok, true, nil
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

// Seed runs a prefill on the target model for the prompt tokens.
func (s *SpeculativeGenerationStream) Seed(prompt []Token) {
	if len(prompt) == 0 {
		return
	}
	_, _ = s.target.Forward(prompt, s.session)
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
	return tokens[accepted-1], false, nil
}

// errGenerationFinished is returned when a stream has been fully consumed.
var errGenerationFinished = errors.New("generation finished")

// IsFinished reports whether err is errGenerationFinished.
func IsFinished(err error) bool { return errors.Is(err, errGenerationFinished) }
