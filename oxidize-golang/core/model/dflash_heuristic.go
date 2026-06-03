package model

import (
	"math"
)

// DFlashDecodeOpts tunes the heuristic draft wrapper (target-delegating stub).
type DFlashDecodeOpts struct {
	DraftSteps         int
	MaxAcceptanceRate  float32
	MinDraftConfidence float32
	Temperature        float32
}

// DefaultDFlashDecodeOpts returns balanced speculative-decode defaults.
func DefaultDFlashDecodeOpts() DFlashDecodeOpts {
	return DFlashDecodeOpts{
		DraftSteps:         4,
		MaxAcceptanceRate:  0.95,
		MinDraftConfidence: 0.1,
		Temperature:        1.0,
	}
}

// DFlashStats tracks heuristic draft verification metrics.
type DFlashStats struct {
	DraftsGenerated int
	DraftsAccepted  int
	SpeedupEstimate float32
	AvgDraftLatency float32
}

// AcceptanceRate returns the ratio of accepted drafts to total.
func (s DFlashStats) AcceptanceRate() float32 {
	if s.DraftsGenerated == 0 {
		return 0
	}
	return float32(s.DraftsAccepted) / float32(s.DraftsGenerated)
}

// HeuristicDFlashDraft wraps a target model for lightweight draft heuristics
// when a full DFlashDraftModel is not loaded.
type HeuristicDFlashDraft struct {
	Target Model
	Config DFlashConfig
	Decode DFlashDecodeOpts
	Stats  DFlashStats
	Cache  []Token
}

// NewHeuristicDFlashDraft constructs a heuristic draft around a target model.
func NewHeuristicDFlashDraft(target Model, config DFlashConfig) *HeuristicDFlashDraft {
	return NewHeuristicDFlashDraftWithOpts(target, config, DefaultDFlashDecodeOpts())
}

// NewHeuristicDFlashDraftWithOpts constructs a heuristic draft with decode tuning.
func NewHeuristicDFlashDraftWithOpts(target Model, config DFlashConfig, decode DFlashDecodeOpts) *HeuristicDFlashDraft {
	if decode.DraftSteps <= 0 {
		if config.BlockSize > 0 {
			decode.DraftSteps = config.BlockSize
		} else {
			decode.DraftSteps = 4
		}
	}
	return &HeuristicDFlashDraft{Target: target, Config: config, Decode: decode}
}

// Forward implements Model by delegating to the target.
func (d *HeuristicDFlashDraft) Forward(tokens []Token, session *Session) (Logits, error) {
	return d.Target.Forward(tokens, session)
}

// VocabSize delegates to the target model.
func (d *HeuristicDFlashDraft) VocabSize() int { return d.Target.VocabSize() }

// ContextSize delegates to the target model.
func (d *HeuristicDFlashDraft) ContextSize() int { return d.Target.ContextSize() }

// LayerCount delegates to the target model.
func (d *HeuristicDFlashDraft) LayerCount() int { return d.Target.LayerCount() }

// Reset clears the draft cache.
func (d *HeuristicDFlashDraft) Reset() { d.Cache = d.Cache[:0] }

// GenerateDraft produces a draft of size steps tokens.
func (d *HeuristicDFlashDraft) GenerateDraft(prompt []Token, steps int) ([]Token, error) {
	if steps <= 0 {
		steps = d.Decode.DraftSteps
	}
	out := append([]Token(nil), prompt...)
	sess := NewSession()
	for i := 0; i < steps; i++ {
		logits, err := d.Target.Forward(out, sess)
		if err != nil {
			return nil, err
		}
		tok, err := Sample(logits, SamplingConfig{
			Temperature: d.Decode.Temperature,
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

// VerifyDrafts runs a verification pass and accepts drafts above MinDraftConfidence.
func (d *HeuristicDFlashDraft) VerifyDrafts(drafts []Token) (accepted []Token, err error) {
	if len(drafts) == 0 {
		return nil, nil
	}
	sess := NewSession()
	logits, err := d.Target.Forward(drafts, sess)
	if err != nil {
		return nil, err
	}
	probs := softmax(logits)
	for _, t := range drafts {
		prob := float32(0)
		if int(t) < len(probs) {
			prob = probs[t]
		}
		if prob >= d.Decode.MinDraftConfidence {
			accepted = append(accepted, t)
		}
	}
	d.Stats.DraftsAccepted += len(accepted)
	return accepted, nil
}

// DraftConfidence computes per-token confidence for the draft sequence.
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

// DFlashEngine orchestrates heuristic draft + target speculative steps.
type DFlashEngine struct {
	Target Model
	Draft  *HeuristicDFlashDraft
	Config DFlashConfig
	Decode DFlashDecodeOpts
	Stats  DFlashStats
}

// NewDFlashEngine constructs a draft + target engine.
func NewDFlashEngine(target Model, draft *HeuristicDFlashDraft, config DFlashConfig) *DFlashEngine {
	decode := DefaultDFlashDecodeOpts()
	if draft != nil {
		decode = draft.Decode
	}
	return &DFlashEngine{Target: target, Draft: draft, Config: config, Decode: decode}
}

// Step runs a single DFlash decode iteration.
func (e *DFlashEngine) Step(prompt []Token) (accepted []Token, err error) {
	drafts, err := e.Draft.GenerateDraft(prompt, e.Decode.DraftSteps)
	if err != nil {
		return nil, err
	}
	accepted, err = e.Draft.VerifyDrafts(drafts)
	if err != nil {
		return nil, err
	}
	ratio := e.Stats.AcceptanceRate()
	e.Stats.SpeedupEstimate = float32(1 + math.Ceil(float64(e.Decode.DraftSteps)*float64(ratio)))
	return accepted, nil
}

// MaxAcceptanceRate returns the configured maximum acceptance rate.
func (e *DFlashEngine) MaxAcceptanceRate() float32 { return e.Decode.MaxAcceptanceRate }

// IsStable returns true when acceptance rate is within configured bounds.
func (e *DFlashEngine) IsStable() bool {
	return e.Stats.AcceptanceRate() <= e.Decode.MaxAcceptanceRate
}
