package model

import (
	"math"
)

// DFlashConfig mirrors DFlashConfig.
type DFlashConfig struct {
	DraftLayers        int
	VerifyLayers       int
	MaxAcceptanceRate  float32
	MinDraftConfidence float32
	Temperature        float32
}

// DefaultDFlashConfig returns balanced defaults.
func DefaultDFlashConfig() DFlashConfig {
	return DFlashConfig{
		DraftLayers:        1,
		VerifyLayers:       4,
		MaxAcceptanceRate:  0.95,
		MinDraftConfidence: 0.1,
		Temperature:        1.0,
	}
}

// DFlashStats mirrors DFlashStats.
type DFlashStats struct {
	DraftsGenerated  int
	DraftsAccepted   int
	SpeedupEstimate  float32
	AvgDraftLatency  float32
}

// AcceptanceRate returns the ratio of accepted drafts to total.
func (s DFlashStats) AcceptanceRate() float32 {
	if s.DraftsGenerated == 0 {
		return 0
	}
	return float32(s.DraftsAccepted) / float32(s.DraftsGenerated)
}

// DFlashDraftModel mirrors DFlashDraftModel.
type DFlashDraftModel struct {
	Target  Model
	Config  DFlashConfig
	Stats   DFlashStats
	Cache   []Token
}

// NewDFlashDraftModel constructs a draft model wrapping a target model.
func NewDFlashDraftModel(target Model, config DFlashConfig) *DFlashDraftModel {
	if config.DraftLayers <= 0 {
		config.DraftLayers = 1
	}
	if config.VerifyLayers <= 0 {
		config.VerifyLayers = 4
	}
	return &DFlashDraftModel{Target: target, Config: config}
}

// Forward implements Model by delegating to the target and applying DFlash
// sampling heuristics.
func (d *DFlashDraftModel) Forward(tokens []Token, session *Session) (Logits, error) {
	return d.Target.Forward(tokens, session)
}

// Reset clears the draft cache.
func (d *DFlashDraftModel) Reset() { d.Cache = d.Cache[:0] }

// GenerateDraft produces a draft of size `steps` tokens.
func (d *DFlashDraftModel) GenerateDraft(prompt []Token, steps int) ([]Token, error) {
	if steps <= 0 {
		steps = 1
	}
	out := append([]Token(nil), prompt...)
	sess := NewSession()
	for i := 0; i < steps; i++ {
		logits, err := d.Target.Forward(out, sess)
		if err != nil {
			return nil, err
		}
		tok, err := Sample(logits, SamplingConfig{
			Temperature: d.Config.Temperature,
			TopP:        1.0,
		}, nil)
		if err != nil {
			return nil, err
		}
		out = append(out, tok)
		d.Cache = append(d.Cache, tok)
	}
	d.Stats.DraftsGenerated += steps
	return out[len(prompt):], nil
}

// VerifyDrafts runs a verification pass and accepts drafts with a confidence
// above the configured minimum.
func (d *DFlashDraftModel) VerifyDrafts(drafts []Token) (accepted []Token, err error) {
	if len(drafts) == 0 {
		return nil, nil
	}
	sess := NewSession()
	logits, err := d.Target.Forward(drafts, sess)
	if err != nil {
		return nil, err
	}
	probs := softmax(logits)
	_ = probs
	for _, t := range drafts {
		prob := float32(0)
		if int(t) < len(probs) {
			prob = probs[t]
		}
		if prob >= d.Config.MinDraftConfidence {
			accepted = append(accepted, t)
		}
	}
	d.Stats.DraftsAccepted += len(accepted)
	return accepted, nil
}

// DraftConfidence computes a per-token confidence for the draft sequence.
func DraftConfidence(target Model, drafts []Token) ([]float32, error) {
	if len(drafts) == 0 {
		return nil, nil
	}
	sess := NewSession()
	logits, err := target.Forward(drafts, sess)
	if err != nil {
		return nil, err
	}
	probs := softmax(logits)
	out := make([]float32, len(drafts))
	for i, t := range drafts {
		if int(t) < len(probs) {
			out[i] = probs[t]
		}
	}
	return out, nil
}

// DFlashEngine mirrors DFlashEngine<'a, T: Model>.
type DFlashEngine struct {
	Target Model
	Draft  *DFlashDraftModel
	Config DFlashConfig
	Stats  DFlashStats
}

// NewDFlashEngine constructs a draft + target engine.
func NewDFlashEngine(target Model, draft *DFlashDraftModel, config DFlashConfig) *DFlashEngine {
	return &DFlashEngine{Target: target, Draft: draft, Config: config}
}

// Step runs a single DFlash decode iteration.
func (e *DFlashEngine) Step(prompt []Token) (accepted []Token, err error) {
	drafts, err := e.Draft.GenerateDraft(prompt, e.Config.DraftLayers)
	if err != nil {
		return nil, err
	}
	accepted, err = e.Draft.VerifyDrafts(drafts)
	if err != nil {
		return nil, err
	}
	ratio := e.Stats.AcceptanceRate()
	e.Stats.SpeedupEstimate = float32(1 + math.Ceil(float64(e.Config.DraftLayers)*float64(ratio)))
	return accepted, nil
}

// MaxAcceptanceRate returns the configured maximum acceptance rate.
func (e *DFlashEngine) MaxAcceptanceRate() float32 { return e.Config.MaxAcceptanceRate }

// IsStable returns true when the engine is achieving a stable acceptance
// rate within bounds of the configured maximum.
func (e *DFlashEngine) IsStable() bool {
	return e.Stats.AcceptanceRate() <= e.Config.MaxAcceptanceRate
}
