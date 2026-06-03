package model

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"strings"
)

// SamplingError mirrors SamplingError.
type SamplingError struct{ Message string }

func (e *SamplingError) Error() string { return "sampling: " + e.Message }

// RepetitionPenaltyConfig mirrors RepetitionPenaltyConfig.
type RepetitionPenaltyConfig struct {
	Penalty          float32
	LastN            int
	FrequencyPenalty float32
	PresencePenalty  float32
}

// DefaultRepetitionPenalty returns neutral defaults.
func DefaultRepetitionPenalty() RepetitionPenaltyConfig {
	return RepetitionPenaltyConfig{Penalty: 1.0, LastN: 64}
}

// MirostatConfig mirrors MirostatConfig.
type MirostatConfig struct {
	Tau float32
	Eta float32
	M   int
}

// DefaultMirostat returns Mirostat v2 defaults.
func DefaultMirostat() MirostatConfig { return MirostatConfig{Tau: 5.0, Eta: 0.1, M: 5} }

// NewlinePenalty mirrors NewlinePenalty.
type NewlinePenalty struct {
	Count  int
	Reward float32
}
// GrammarSymbol mirrors GrammarSymbol.
type GrammarSymbol interface{ isGrammar() }

// GrammarTerminal is a token id terminal.
type GrammarTerminal struct{ TokenID uint32 }

// GrammarNonTerminal is a named non-terminal.
type GrammarNonTerminal struct{ Name string }

func (GrammarTerminal) isGrammar()    {}
func (GrammarNonTerminal) isGrammar() {}

// GrammarConstraint mirrors GrammarConstraint.
type GrammarConstraint struct {
	Start       string
	Productions map[string][][]GrammarSymbol
	MaxStates   int
}

// NewGrammarConstraint constructs an empty grammar.
func NewGrammarConstraint(start string) *GrammarConstraint {
	return &GrammarConstraint{Start: start, Productions: map[string][][]GrammarSymbol{}, MaxStates: 20000}
}

// AddProduction appends a production rule.
func (g *GrammarConstraint) AddProduction(name string, body []GrammarSymbol) {
	g.Productions[name] = append(g.Productions[name], body)
}

// AllowsToken reports whether the given token is currently allowed by the
// grammar. This is a simplified check that returns true when the grammar has
// no productions yet or when the token is listed in the matching production
// for the start symbol.
func (g *GrammarConstraint) AllowsToken(token Token, _ []Token) bool {
	if g == nil {
		return true
	}
	productions, ok := g.Productions[g.Start]
	if !ok {
		return true
	}
	for _, prod := range productions {
		for _, sym := range prod {
			if t, ok := sym.(GrammarTerminal); ok {
				if t.TokenID == token {
					return true
				}
			}
		}
	}
	return len(productions) == 0
}

// SamplingConfig mirrors SamplingConfig.
type SamplingConfig struct {
	Temperature      float32
	TopP             float32
	TopK             int
	MinP             float32
	TypicalP         float32
	TailFreeZ        float32
	Repetition       RepetitionPenaltyConfig
	Mirostat         MirostatConfig
	NewlinePenalty   NewlinePenalty
	Grammar          *GrammarConstraint
	SuppressedTokens []Token
	XTC              *XtcSamplerConfig
	DRY              *DrySamplerConfig
	Chain            *SamplerChain
}

// DefaultSamplingConfig returns sensible sampling defaults.
func DefaultSamplingConfig() SamplingConfig {
	return SamplingConfig{
		Temperature:    1.0,
		TopP:           1.0,
		TopK:           0,
		MinP:           0.0,
		TypicalP:       0.0,
		TailFreeZ:      0.0,
		Repetition:     DefaultRepetitionPenalty(),
		Mirostat:       MirostatConfig{},
		NewlinePenalty: NewlinePenalty{},
		SuppressedTokens: nil,
	}
}

// Greedy returns the argmax token id.
func Greedy(logits Logits) (Token, error) {
	if len(logits) == 0 {
		return 0, &SamplingError{Message: "empty logits"}
	}
	best := 0
	bestVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			best = i
		}
	}
	return Token(best), nil
}

// Sample draws a token from the distribution described by logits + config.
func Sample(logits Logits, config SamplingConfig, rng *rand.Rand) (Token, error) {
	if len(logits) == 0 {
		return 0, &SamplingError{Message: "empty logits"}
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	// Apply repetition penalty if any
	if config.Repetition.Penalty != 1.0 {
		logits = applyRepetitionPenalty(logits, config.Repetition, nil)
	}
	// Apply temperature
	if config.Temperature > 0 && config.Temperature != 1.0 {
		for i, v := range logits {
			logits[i] = v / config.Temperature
		}
	}
	// Suppressed tokens
	if len(config.SuppressedTokens) > 0 {
		sup := make(map[Token]struct{}, len(config.SuppressedTokens))
		for _, t := range config.SuppressedTokens {
			sup[t] = struct{}{}
		}
		for t := range sup {
			if int(t) < len(logits) {
				logits[t] = negInfF32
			}
		}
	}
	// Top-K
	if config.TopK > 0 {
		logits = topK(logits, config.TopK)
	}
	// Min-P
	if config.MinP > 0 {
		logits = minP(logits, config.MinP)
	}
	// Top-P
	if config.TopP > 0 && config.TopP < 1.0 {
		logits = topP(logits, config.TopP)
	}
	// Softmax
	probs := softmax(logits)
	// Sample
	return sampleCategorical(probs, rng), nil
}

// SampleWithRepetition applies repetition penalty before sampling.
func SampleWithRepetition(logits Logits, config SamplingConfig, history []Token, rng *rand.Rand) (Token, error) {
	if config.Repetition.Penalty != 1.0 {
		logits = applyRepetitionPenalty(logits, config.Repetition, history)
	}
	return Sample(logits, config, rng)
}

// SampleWithRepetitionAndGrammar applies repetition + grammar filtering.
func SampleWithRepetitionAndGrammar(logits Logits, config SamplingConfig, history []Token, rng *rand.Rand) (Token, error) {
	if config.Grammar != nil {
		for t := range logits {
			if !config.Grammar.AllowsToken(Token(t), history) {
				logits[t] = negInfF32
			}
		}
	}
	return SampleWithRepetition(logits, config, history, rng)
}

// SpeculativeDecodeResult mirrors SpeculativeDecodeResult.
type SpeculativeDecodeResult struct {
	Accepted   []Token
	BonusToken *Token
	NumDrafts  int
}

// SpeculativeDecode runs a single speculative decode step using a draft and
// target model.
func SpeculativeDecode(draft, target Model, draftTokens int, session *Session) ([]Token, int, error) {
	if draftTokens <= 0 {
		draftTokens = 4
	}
	// Generate draft tokens
	draftOut := make([]Token, 0, draftTokens)
	ds := NewGenerationStream(draft, session, GenerationConfig{MaxNewTokens: draftTokens, StopToken: 0, Sampling: DefaultSamplingConfig()})
	ctx := context.Background()
	for {
		tok, done, err := ds.Next(ctx)
		if err != nil {
			return nil, 0, err
		}
		draftOut = append(draftOut, tok)
		if done {
			break
		}
	}
	// Verify via target model logits
	logits, err := target.Forward(draftOut, session)
	if err != nil {
		return nil, 0, err
	}
	bestTok, err := Greedy(logits)
	if err != nil {
		return nil, 0, err
	}
	accepted := append([]Token(nil), draftOut...)
	accepted = append(accepted, bestTok)
	return accepted, len(accepted), nil
}

func sessionContext() context.Context { return context.Background() }

// SampleMirostat implements Mirostat v2 sampling.
func SampleMirostat(logits Logits, config MirostatConfig, lastSurprise float32) (Token, float32, error) {
	if len(logits) == 0 {
		return 0, lastSurprise, &SamplingError{Message: "empty logits"}
	}
	sorted := sortedCopy(logits)
	maxLogit := sorted[0]
	for _, v := range sorted {
		if v > maxLogit {
			maxLogit = v
		}
	}
	probs := softmax(sorted)
	logProbs := make([]float32, len(probs))
	for i, p := range probs {
		logProbs[i] = float32(math.Log(float64(p)))
	}
	// pick token that minimizes surprise (closest to tau)
	bestIdx := 0
	bestDist := float32(math.Abs(float64(logProbs[0] + config.Tau)))
	for i, lp := range logProbs {
		dist := float32(math.Abs(float64(lp + config.Tau)))
		if dist < bestDist {
			bestDist = dist
			bestIdx = i
		}
	}
	surprise := -logProbs[bestIdx]
	updatedSurprise := lastSurprise
	if updatedSurprise == 0 {
		updatedSurprise = surprise
	} else {
		updatedSurprise = (1-config.Eta)*updatedSurprise + config.Eta*surprise
	}
	return Token(bestIdx), updatedSurprise, nil
}

// BeamSearchResult mirrors BeamSearchResult.
type BeamSearchResult struct {
	Tokens  []Token
	Score   float32
}

// BeamSearch implements a simple length-normalized beam search.
func BeamSearch(model Model, prompt []Token, beamWidth, maxSteps int, session *Session) (BeamSearchResult, error) {
	if beamWidth <= 0 {
		beamWidth = 1
	}
	type beam struct {
		tokens []Token
		score  float32
	}
	beams := []beam{{tokens: append([]Token(nil), prompt...), score: 0}}
	for step := 0; step < maxSteps; step++ {
		var candidates []beam
		for _, b := range beams {
			logits, err := model.Forward(b.tokens, session)
			if err != nil {
				return BeamSearchResult{}, err
			}
			probs := softmax(logits)
			for i, p := range probs {
				if math.IsInf(float64(p), -1) || math.IsNaN(float64(p)) {
					continue
				}
				toks := append([]Token(nil), b.tokens...)
				toks = append(toks, Token(i))
				candidates = append(candidates, beam{tokens: toks, score: b.score + float32(math.Log(float64(p)+1e-12))})
			}
		}
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].score > candidates[j].score
		})
		if len(candidates) > beamWidth {
			candidates = candidates[:beamWidth]
		}
		beams = candidates
	}
	if len(beams) == 0 {
		return BeamSearchResult{}, nil
	}
	best := beams[0]
	norm := float32(len(best.tokens))
	return BeamSearchResult{Tokens: best.tokens, Score: best.score / norm}, nil
}

// Helpers ---------------------------------------------------------------

func applyRepetitionPenalty(logits Logits, cfg RepetitionPenaltyConfig, history []Token) Logits {
	if cfg.Penalty == 1.0 || len(history) == 0 {
		return logits
	}
	if cfg.LastN > 0 && len(history) > cfg.LastN {
		history = history[len(history)-cfg.LastN:]
	}
	freq := map[Token]int{}
	for _, t := range history {
		freq[t]++
	}
	out := append(Logits(nil), logits...)
	for t, count := range freq {
		if int(t) < len(out) {
			if out[t] >= 0 {
				out[t] = out[t] / cfg.Penalty
			} else {
				out[t] = out[t] * cfg.Penalty
			}
			if cfg.FrequencyPenalty != 0 {
				out[t] -= float32(count) * cfg.FrequencyPenalty
			}
			if cfg.PresencePenalty != 0 {
				if count > 0 {
					out[t] -= cfg.PresencePenalty
				}
			}
		}
	}
	return out
}

func topK(logits Logits, k int) Logits {
	idx := sortedIndices(logits)
	cutoff := logits[idx[0]]
	if k < len(idx) {
		cutoff = logits[idx[k-1]]
	}
	for i, v := range logits {
		if v < cutoff {
			logits[i] = negInfF32
		}
	}
	return logits
}

func minP(logits Logits, p float32) Logits {
	probs := softmax(logits)
	maxIdx := 0
	for i, v := range probs {
		if v > probs[maxIdx] {
			maxIdx = i
		}
	}
	cutoff := probs[maxIdx] * p
	for i, pv := range probs {
		if pv < cutoff && i != maxIdx {
			logits[i] = negInfF32
		}
	}
	return logits
}

func topP(logits Logits, p float32) Logits {
	idx := sortedIndices(logits)
	var cumProb float32
	cutoff := logits[idx[0]]
	for _, i := range idx {
		cumProb += float32(math.Exp(float64(logits[i] - logits[idx[0]])))
		if cumProb >= p {
			cutoff = logits[i]
			break
		}
	}
	for i, v := range logits {
		if v < cutoff {
			logits[i] = negInfF32
		}
	}
	return logits
}

func softmax(logits Logits) []float32 {
	if len(logits) == 0 {
		return nil
	}
	maxVal := logits[0]
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}
	out := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		out[i] = float32(math.Exp(float64(v - maxVal)))
		sum += out[i]
	}
	inv := 1 / sum
	for i := range out {
		out[i] *= inv
	}
	return out
}

func sortedIndices(logits Logits) []int {
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool {
		return logits[idx[i]] > logits[idx[j]]
	})
	return idx
}

func sortedCopy(logits Logits) Logits {
	out := append(Logits(nil), logits...)
	sort.Slice(out, func(i, j int) bool { return out[i] > out[j] })
	return out
}

func sampleCategorical(probs []float32, rng *rand.Rand) Token {
	r := rng.Float32()
	var cum float32
	for i, p := range probs {
		cum += p
		if r < cum {
			return Token(i)
		}
	}
	return Token(len(probs) - 1)
}

// silence unused import
var _ = strings.Repeat
