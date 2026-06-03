package model

import (
	"fmt"
	"math"
)

// SpeculativeConfig mirrors SpeculativeConfig.
type SpeculativeConfig struct {
	DraftTokensPerStep int
	MaxNewTokens       int
	Sampling           SamplingConfig
	StopToken          *Token
	StrictMode         bool
	MinAcceptanceRate  float32
}

// DefaultSpeculativeConfig returns the balanced default.
func DefaultSpeculativeConfig() SpeculativeConfig {
	return SpeculativeConfig{
		DraftTokensPerStep: 4,
		MaxNewTokens:       128,
		Sampling:           DefaultSamplingConfig(),
		StopToken:          nil,
		StrictMode:         false,
		MinAcceptanceRate:  0.3,
	}
}

// Conservative returns a config tuned for high accuracy (low draft count).
func Conservative() SpeculativeConfig {
	c := DefaultSpeculativeConfig()
	c.DraftTokensPerStep = 2
	c.StrictMode = true
	return c
}

// Aggressive returns a config tuned for speed (high draft count, lower
// acceptance threshold).
func Aggressive() SpeculativeConfig {
	c := DefaultSpeculativeConfig()
	c.DraftTokensPerStep = 8
	c.MinAcceptanceRate = 0.3
	return c
}

// SpeculativeStats mirrors SpeculativeStats.
type SpeculativeStats struct {
	TotalDraftTokens     int
	TotalAcceptedTokens  int
	TotalRejectedTokens  int
	DraftForwardPasses   int
	TargetForwardPasses  int
	FallbackTokens       int
	Accepted             int
	Rejected             int
	Total                int
}

// AcceptanceRate returns the ratio of accepted drafts to total.
func (s SpeculativeStats) AcceptanceRate() float32 {
	if s.Total == 0 {
		return 0
	}
	return float32(s.Accepted) / float32(s.Total)
}

// SpeculativeDecoder mirrors SpeculativeDecoder<'a, T: Model>.
type SpeculativeDecoder struct {
	Draft     Model
	Target    Model
	Session   *Session
	Config    SpeculativeConfig
	Stats     SpeculativeStats
}

// NewSpeculativeDecoder constructs a decoder with the given draft + target
// models.
func NewSpeculativeDecoder(draft, target Model, session *Session, config SpeculativeConfig) *SpeculativeDecoder {
	return &SpeculativeDecoder{Draft: draft, Target: target, Session: session, Config: config}
}

// Step runs one speculative decode step and returns the accepted tokens.
func (d *SpeculativeDecoder) Step() ([]Token, error) {
	if d.Config.DraftTokensPerStep <= 0 {
		return nil, fmt.Errorf("draft tokens per step must be > 0")
	}
	tokens, accepted, err := SpeculativeDecode(d.Draft, d.Target, d.Config.DraftTokensPerStep, d.Session)
	if err != nil {
		return nil, err
	}
	d.Stats.Total += d.Config.DraftTokensPerStep
	if accepted < d.Config.DraftTokensPerStep {
		d.Stats.Rejected++
	} else {
		d.Stats.Accepted += d.Config.DraftTokensPerStep
	}
	return tokens, nil
}

// SpeculativeError mirrors SpeculativeError.
type SpeculativeError struct{ Message string }

func (e *SpeculativeError) Error() string { return "speculative: " + e.Message }

// LoadDraftModelForSpeculative is a placeholder helper. In the Rust crate it
// resolves a smaller draft model from the same path; in the Go port we
// return a stub model with the same vocab/context as the target.
func LoadDraftModelForSpeculative(target Model) Model {
	switch m := target.(type) {
	case *LlamaModel:
		out := *m
		out.Config.LayerCount = m.Config.LayerCount / 2
		if out.Config.LayerCount == 0 {
			out.Config.LayerCount = 1
		}
		return &out
	case *InferenceModel:
		out := *m
		out.Config.LayerCount = m.Config.LayerCount / 2
		if out.Config.LayerCount == 0 {
			out.Config.LayerCount = 1
		}
		return &out
	}
	return target
}

// SpeculativeConfigBuilder mirrors SpeculativeConfigBuilder.
type SpeculativeConfigBuilder struct {
	cfg SpeculativeConfig
}

// NewSpeculativeConfigBuilder constructs a builder.
func NewSpeculativeConfigBuilder() *SpeculativeConfigBuilder { return &SpeculativeConfigBuilder{cfg: DefaultSpeculativeConfig()} }

// WithDraftTokens sets the draft tokens per step.
func (b *SpeculativeConfigBuilder) WithDraftTokens(n int) *SpeculativeConfigBuilder { b.cfg.DraftTokensPerStep = n; return b }

// WithMaxNewTokens sets the maximum new tokens.
func (b *SpeculativeConfigBuilder) WithMaxNewTokens(n int) *SpeculativeConfigBuilder { b.cfg.MaxNewTokens = n; return b }

// WithStrict sets strict mode.
func (b *SpeculativeConfigBuilder) WithStrict(s bool) *SpeculativeConfigBuilder { b.cfg.StrictMode = s; return b }

// Build returns the final config.
func (b *SpeculativeConfigBuilder) Build() SpeculativeConfig { return b.cfg }

// silence unused import warnings
var _ = math.Abs
