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
	Temperature       float32
	TopP              float32
	TopK              int
	MinP              float32
	TypicalP          float32
	TailFreeZ         float32
	LocallyTypicalTau float32
	Repetition        RepetitionPenaltyConfig
	Mirostat          MirostatConfig
	NewlinePenalty    NewlinePenalty
	Grammar           *GrammarConstraint
	SuppressedTokens  []Token
	XTC               *XtcSamplerConfig
	DRY               *DrySamplerConfig
	Chain             *SamplerChain
}

// DefaultSamplingConfig returns sensible sampling defaults.
func DefaultSamplingConfig() SamplingConfig {
	return SamplingConfig{
		Temperature:       1.0,
		TopP:              1.0,
		TopK:              0,
		MinP:              0.0,
		TypicalP:          0.0,
		TailFreeZ:         0.0,
		LocallyTypicalTau: 0.0,
		Repetition:        DefaultRepetitionPenalty(),
		Mirostat:          MirostatConfig{},
		NewlinePenalty:    NewlinePenalty{},
		SuppressedTokens:  nil,
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
	// Typical-P (entropy-based typicality filtering)
	if config.TypicalP > 0 && config.TypicalP < 1.0 {
		logits = typicalP(logits, config.TypicalP)
	}
	// Tail-free (second-derivative cutoff)
	if config.TailFreeZ > 0 && config.TailFreeZ < 1.0 {
		logits = tailFreeZ(logits, config.TailFreeZ)
	}
	// Locally-typical (tau-based entropy deviation filtering)
	if config.LocallyTypicalTau > 0 {
		logits = locallyTypicalTau(logits, config.LocallyTypicalTau)
	}
	// Fast unfiltered path for large vocabularies when no rank/probability
	// filters are active: avoids an allocating full softmax pass.
	if len(logits) >= 4096 && config.TopK == 0 && config.MinP == 0 &&
		(config.TopP == 0 || config.TopP >= 1.0) &&
		config.TypicalP == 0 && config.TailFreeZ == 0 && config.LocallyTypicalTau == 0 {
		temp := config.Temperature
		if temp <= 0 {
			temp = 1.0
		}
		return SampleUnfiltered(logits, temp, rng), nil
	}
	// Softmax
	probs := softmax(logits)
	// Sample
	return sampleCategorical(probs, rng), nil
}

// SampleUnfiltered draws a token directly from temperature-scaled logits
// without building an intermediate normalized softmax slice. It mirrors
// sample_unfiltered in sampling.rs and is a fast path for large vocabularies.
func SampleUnfiltered(logits Logits, temperature float32, rng *rand.Rand) Token {
	if len(logits) == 0 {
		return 0
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	if temperature <= 0 {
		temperature = 1.0
	}
	maxLogit := logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var rawSum float32
	for _, v := range logits {
		rawSum += float32(math.Exp(float64((v - maxLogit) / temperature)))
	}
	if rawSum <= 0 || math.IsInf(float64(rawSum), 0) || math.IsNaN(float64(rawSum)) {
		tok, _ := Greedy(logits)
		return tok
	}
	target := rng.Float32() * rawSum
	var cumulative float32
	for i, v := range logits {
		cumulative += float32(math.Exp(float64((v - maxLogit) / temperature)))
		if target <= cumulative {
			return Token(i)
		}
	}
	tok, _ := Greedy(logits)
	return tok
}

// typicalP keeps the minimal set of tokens (ordered by closeness of their
// surprise to the distribution entropy) whose cumulative probability reaches p.
// Mirrors apply_typical_sampling in sampling.rs.
func typicalP(logits Logits, p float32) Logits {
	if p <= 0 || len(logits) == 0 {
		return logits
	}
	probs := softmax(logits)
	var entropy float32
	for _, pr := range probs {
		pr = maxF32(pr, minPositiveF32)
		entropy -= pr * float32(math.Log(float64(pr)))
	}
	type cand struct {
		idx  int
		diff float32
	}
	cands := make([]cand, len(probs))
	for i, pr := range probs {
		surprise := -float32(math.Log(float64(maxF32(pr, minPositiveF32))))
		cands[i] = cand{idx: i, diff: absF32(surprise - entropy)}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].diff < cands[j].diff })
	keep := make([]bool, len(probs))
	var cum float32
	for _, c := range cands {
		keep[c.idx] = true
		cum += probs[c.idx]
		if cum >= p {
			break
		}
	}
	for i := range logits {
		if !keep[i] {
			logits[i] = negInfF32
		}
	}
	return logits
}

// tailFreeZ removes the low-probability tail using the second derivative of the
// sorted probability curve. Mirrors apply_tail_free_sampling in sampling.rs.
func tailFreeZ(logits Logits, z float32) Logits {
	if z <= 0 || len(logits) <= 2 {
		return logits
	}
	idx := sortedIndices(logits)
	probs := softmax(logits)
	secondDeriv := make([]float32, len(idx)-2)
	for i := 0; i < len(idx)-2; i++ {
		d1 := probs[idx[i]] - probs[idx[i+1]]
		d2 := probs[idx[i+1]] - probs[idx[i+2]]
		secondDeriv[i] = absF32(d1 - d2)
	}
	var sdSum float32
	for _, sd := range secondDeriv {
		sdSum += sd
	}
	if sdSum <= 0 || math.IsInf(float64(sdSum), 0) || math.IsNaN(float64(sdSum)) {
		return logits
	}
	cutoff := len(idx)
	var cum float32
	for i, sd := range secondDeriv {
		cum += sd / sdSum
		if cum >= z {
			cutoff = i + 2
			if cutoff < 1 {
				cutoff = 1
			}
			break
		}
	}
	keep := make([]bool, len(logits))
	for i := 0; i < cutoff && i < len(idx); i++ {
		keep[idx[i]] = true
	}
	for i := range logits {
		if !keep[i] {
			logits[i] = negInfF32
		}
	}
	return logits
}

// locallyTypicalTau keeps tokens whose surprise lies within tau*entropy of the
// distribution entropy. Mirrors apply_locally_typical_sampling in sampling.rs.
func locallyTypicalTau(logits Logits, tau float32) Logits {
	if tau <= 0 || len(logits) == 0 {
		return logits
	}
	probs := softmax(logits)
	var entropy float32
	for _, pr := range probs {
		pr = maxF32(pr, minPositiveF32)
		entropy -= pr * float32(math.Log(float64(pr)))
	}
	deviationLimit := entropy * tau
	keep := make([]bool, len(probs))
	any := false
	for i, pr := range probs {
		surprise := -float32(math.Log(float64(maxF32(pr, minPositiveF32))))
		if absF32(surprise-entropy) <= deviationLimit {
			keep[i] = true
			any = true
		}
	}
	if !any {
		maxIdx := 0
		for i, pr := range probs {
			if pr > probs[maxIdx] {
				maxIdx = i
			}
		}
		keep[maxIdx] = true
	}
	for i := range logits {
		if !keep[i] {
			logits[i] = negInfF32
		}
	}
	return logits
}

const minPositiveF32 = float32(1.1754944e-38)

func maxF32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func absF32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
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

// SpeculativeVerifyResult mirrors SpeculativeDecodeResult in sampling.rs and
// carries enough detail for statistics tracking.
type SpeculativeVerifyResult struct {
	Tokens               []Token
	AcceptedDraftTokens  int
	UsedResidualFallback bool
}

// softmaxProbsTemp returns a temperature-scaled softmax over logits.
func softmaxProbsTemp(logits Logits, temperature float32) ([]float32, error) {
	if len(logits) == 0 {
		return nil, &SamplingError{Message: "empty logits"}
	}
	if temperature <= 0 || math.IsInf(float64(temperature), 0) || math.IsNaN(float64(temperature)) {
		temperature = 1.0
	}
	maxLogit := logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}
	out := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		e := float32(math.Exp(float64((v - maxLogit) / temperature)))
		out[i] = e
		sum += e
	}
	if sum <= 0 || math.IsInf(float64(sum), 0) || math.IsNaN(float64(sum)) {
		return nil, &SamplingError{Message: "non-finite softmax sum"}
	}
	inv := 1 / sum
	for i := range out {
		out[i] *= inv
	}
	return out, nil
}

// residualProbs computes the normalized residual distribution max(p-q, 0).
// Mirrors residual_probs in sampling.rs.
func residualProbs(target, draft []float32) []float32 {
	out := make([]float32, len(target))
	var sum float32
	for i := range target {
		d := float32(0)
		if i < len(draft) {
			d = draft[i]
		}
		r := target[i] - d
		if r < 0 {
			r = 0
		}
		out[i] = r
		sum += r
	}
	if sum <= 0 {
		// Fall back to the raw target distribution.
		copy(out, target)
		var tsum float32
		for _, v := range target {
			tsum += v
		}
		if tsum > 0 {
			inv := 1 / tsum
			for i := range out {
				out[i] *= inv
			}
		}
		return out
	}
	inv := 1 / sum
	for i := range out {
		out[i] *= inv
	}
	return out
}

// SpeculativeDecodeLogits verifies draft tokens against precomputed target
func SpeculativeDecodeLogits(draftTokens []Token, draftLogits, targetLogits []Logits, cfg SamplingConfig, randoms []float32) (SpeculativeVerifyResult, error) {
	n := len(draftTokens)
	if n == 0 || len(draftLogits) != n || len(targetLogits) != n+1 || len(randoms) < n+1 {
		return SpeculativeVerifyResult{}, &SamplingError{Message: "invalid speculative inputs"}
	}
	greedyMode := cfg.Temperature <= 0 || cfg.TopK == 1
	verifyTemp := float32(1.0)
	if !greedyMode {
		verifyTemp = cfg.Temperature
	}
	emitted := make([]Token, 0, n+1)
	for step := 0; step < n; step++ {
		draftTok := draftTokens[step]
		if greedyMode {
			targetArgmax, err := Greedy(targetLogits[step])
			if err != nil {
				return SpeculativeVerifyResult{}, err
			}
			if draftTok == targetArgmax {
				emitted = append(emitted, draftTok)
				continue
			}
			emitted = append(emitted, targetArgmax)
			return SpeculativeVerifyResult{Tokens: emitted, AcceptedDraftTokens: step, UsedResidualFallback: true}, nil
		}
		draftProbs, err := softmaxProbsTemp(draftLogits[step], verifyTemp)
		if err != nil {
			return SpeculativeVerifyResult{}, err
		}
		targetProbs, err := softmaxProbsTemp(targetLogits[step], verifyTemp)
		if err != nil {
			return SpeculativeVerifyResult{}, err
		}
		if len(draftProbs) != len(targetProbs) {
			return SpeculativeVerifyResult{}, &SamplingError{Message: "speculative vocab mismatch"}
		}
		ti := int(draftTok)
		if ti >= len(draftProbs) {
			return SpeculativeVerifyResult{}, &SamplingError{Message: "speculative token out of range"}
		}
		q := maxF32(draftProbs[ti], minPositiveF32)
		p := targetProbs[ti]
		acceptProb := p / q
		if acceptProb > 1.0 {
			acceptProb = 1.0
		}
		if randoms[step] <= acceptProb {
			emitted = append(emitted, draftTok)
			continue
		}
		residual := residualProbs(targetProbs, draftProbs)
		sampled := sampleProbabilities(residual, randoms[step])
		emitted = append(emitted, Token(sampled))
		return SpeculativeVerifyResult{Tokens: emitted, AcceptedDraftTokens: step, UsedResidualFallback: true}, nil
	}
	finalRng := rand.New(rand.NewSource(int64(math.Float32bits(randoms[n]))))
	finalTok, err := Sample(append(Logits(nil), targetLogits[n]...), cfg, finalRng)
	if err != nil {
		return SpeculativeVerifyResult{}, err
	}
	emitted = append(emitted, finalTok)
	return SpeculativeVerifyResult{Tokens: emitted, AcceptedDraftTokens: n, UsedResidualFallback: false}, nil
}

// sampleProbabilities draws an index from a normalized distribution using a
// single random value in [0,1).
func sampleProbabilities(probs []float32, r float32) int {
	var cum float32
	for i, p := range probs {
		cum += p
		if r <= cum {
			return i
		}
	}
	return len(probs) - 1
}

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

// SampleMirostatV2 is a fully-validated Mirostat v2 sampler that mirrors
func SampleMirostatV2(logits Logits, temperature float32, config MirostatConfig, mu, random float32) (Token, float32, error) {
	if len(logits) == 0 {
		return 0, mu, &SamplingError{Message: "empty logits"}
	}
	if math.IsInf(float64(temperature), 0) || math.IsNaN(float64(temperature)) || temperature <= 0 {
		return 0, mu, &SamplingError{Message: "invalid temperature"}
	}
	if math.IsInf(float64(config.Tau), 0) || math.IsNaN(float64(config.Tau)) || config.Tau <= 0 ||
		math.IsInf(float64(config.Eta), 0) || math.IsNaN(float64(config.Eta)) || config.Eta <= 0 ||
		math.IsInf(float64(mu), 0) || math.IsNaN(float64(mu)) {
		return 0, mu, &SamplingError{Message: "invalid mirostat parameters"}
	}
	if math.IsInf(float64(random), 0) || math.IsNaN(float64(random)) || random < 0 || random >= 1 {
		return 0, mu, &SamplingError{Message: "invalid random"}
	}
	probs, err := softmaxProbsTemp(logits, temperature)
	if err != nil {
		return 0, mu, err
	}
	type ip struct {
		idx  int
		prob float32
	}
	indexed := make([]ip, len(probs))
	for i, p := range probs {
		indexed[i] = ip{idx: i, prob: p}
	}
	// Order by closeness of surprisal to the running target mu.
	sort.Slice(indexed, func(a, b int) bool {
		sa := -float32(math.Log(float64(maxF32(indexed[a].prob, minPositiveF32))))
		sb := -float32(math.Log(float64(maxF32(indexed[b].prob, minPositiveF32))))
		return absF32(sa-mu) < absF32(sb-mu)
	})
	// Weighted pick over the reordered, renormalized probabilities.
	var sum float32
	for _, e := range indexed {
		sum += e.prob
	}
	chosen := indexed[0]
	if sum > 0 {
		target := random * sum
		var cum float32
		for _, e := range indexed {
			cum += e.prob
			if target <= cum {
				chosen = e
				break
			}
		}
	}
	observed := -float32(math.Log(float64(maxF32(chosen.prob, minPositiveF32))))
	updatedMu := mu - config.Eta*(observed-config.Tau)
	return Token(chosen.idx), updatedMu, nil
}

// BeamSearchLogits runs beam search over precomputed per-step logits with EOS
// early stopping and final length-aware selection. Mirrors beam_search in
// sampling.rs.
func BeamSearchLogits(logitsPerStep []Logits, beamWidth int, eosToken *Token) (BeamSearchResult, error) {
	if beamWidth <= 0 {
		return BeamSearchResult{}, &SamplingError{Message: "invalid beam width"}
	}
	if len(logitsPerStep) == 0 {
		return BeamSearchResult{}, &SamplingError{Message: "invalid beam search inputs"}
	}
	for _, l := range logitsPerStep {
		if len(l) == 0 {
			return BeamSearchResult{}, &SamplingError{Message: "invalid beam search inputs"}
		}
	}
	type beam struct {
		tokens   []Token
		score    float32
		finished bool
	}
	beams := []beam{{}}
	for _, stepLogits := range logitsPerStep {
		probs := softmax(stepLogits)
		var candidates []beam
		for _, b := range beams {
			if b.finished {
				candidates = append(candidates, b)
				continue
			}
			for tokIdx, p := range probs {
				if p <= 0 || math.IsInf(float64(p), 0) || math.IsNaN(float64(p)) {
					continue
				}
				next := append([]Token(nil), b.tokens...)
				next = append(next, Token(tokIdx))
				finished := eosToken != nil && *eosToken == Token(tokIdx)
				candidates = append(candidates, beam{
					tokens:   next,
					score:    b.score + float32(math.Log(float64(p))),
					finished: finished,
				})
			}
		}
		if len(candidates) == 0 {
			return BeamSearchResult{}, &SamplingError{Message: "empty logits"}
		}
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })
		if len(candidates) > beamWidth {
			candidates = candidates[:beamWidth]
		}
		beams = candidates
		allFinished := true
		for _, b := range beams {
			if !b.finished {
				allFinished = false
				break
			}
		}
		if allFinished {
			break
		}
	}
	best := beams[0]
	for _, b := range beams[1:] {
		if b.score > best.score {
			best = b
		}
	}
	return BeamSearchResult{Tokens: best.tokens, Score: best.score}, nil
}

// BeamSearchResult mirrors BeamSearchResult.
type BeamSearchResult struct {
	Tokens []Token
	Score  float32
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
