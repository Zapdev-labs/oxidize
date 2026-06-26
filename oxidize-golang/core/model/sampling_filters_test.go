package model

import (
	"math"
	"math/rand"
	"testing"
)

func countFinite(l Logits) int {
	n := 0
	for _, v := range l {
		if !math.IsInf(float64(v), -1) {
			n++
		}
	}
	return n
}

func TestTypicalPReducesVocab(t *testing.T) {
	logits := Logits{4.0, 3.5, 1.0, 0.5, 0.2, 0.1, -1.0, -2.0}
	out := typicalP(append(Logits(nil), logits...), 0.5)
	kept := countFinite(out)
	if kept == 0 {
		t.Fatal("typicalP removed all tokens")
	}
	if kept == len(logits) {
		t.Fatal("typicalP kept every token at p=0.5")
	}
}

func TestTailFreeZReducesVocab(t *testing.T) {
	logits := Logits{5.0, 4.0, 1.0, 0.9, 0.8, 0.1, -3.0}
	out := tailFreeZ(append(Logits(nil), logits...), 0.9)
	kept := countFinite(out)
	if kept == 0 {
		t.Fatal("tailFreeZ removed all tokens")
	}
	// Top token must always survive.
	if math.IsInf(float64(out[0]), -1) {
		t.Fatal("tailFreeZ dropped the top token")
	}
}

func TestLocallyTypicalTauKeepsAtLeastOne(t *testing.T) {
	logits := Logits{10.0, -10.0, -10.0, -10.0}
	out := locallyTypicalTau(append(Logits(nil), logits...), 0.01)
	if countFinite(out) == 0 {
		t.Fatal("locallyTypicalTau must keep at least one token")
	}
}

func TestSampleUnfilteredDeterministicArgmax(t *testing.T) {
	// A near-one-hot distribution should overwhelmingly select the max index.
	logits := make(Logits, 8)
	for i := range logits {
		logits[i] = -50
	}
	logits[3] = 50
	rng := rand.New(rand.NewSource(7))
	got := SampleUnfiltered(logits, 1.0, rng)
	if got != 3 {
		t.Fatalf("SampleUnfiltered = %d, want 3", got)
	}
}

func TestSampleUnfilteredFastPathInSample(t *testing.T) {
	// Large vocab with no rank filters triggers the fast path; verify it still
	// returns a valid token id.
	logits := make(Logits, 5000)
	for i := range logits {
		logits[i] = -30
	}
	logits[1234] = 30
	cfg := DefaultSamplingConfig()
	rng := rand.New(rand.NewSource(1))
	tok, err := Sample(append(Logits(nil), logits...), cfg, rng)
	if err != nil {
		t.Fatal(err)
	}
	if int(tok) >= len(logits) {
		t.Fatalf("token %d out of range", tok)
	}
	if tok != 1234 {
		t.Fatalf("expected dominant token 1234, got %d", tok)
	}
}

func TestSpeculativeDecodeLogitsGreedyAcceptAll(t *testing.T) {
	// Target argmax matches every draft token -> all accepted + bonus.
	draft := []Token{1, 2}
	mk := func(arg int) Logits {
		l := make(Logits, 4)
		for i := range l {
			l[i] = 0
		}
		l[arg] = 10
		return l
	}
	draftLogits := []Logits{mk(1), mk(2)}
	targetLogits := []Logits{mk(1), mk(2), mk(3)}
	cfg := DefaultSamplingConfig()
	cfg.Temperature = 0 // greedy
	randoms := []float32{0.1, 0.1, 0.1}
	res, err := SpeculativeDecodeLogits(draft, draftLogits, targetLogits, cfg, randoms)
	if err != nil {
		t.Fatal(err)
	}
	if res.AcceptedDraftTokens != 2 {
		t.Fatalf("accepted = %d, want 2", res.AcceptedDraftTokens)
	}
	if res.UsedResidualFallback {
		t.Fatal("greedy full-accept should not use residual fallback")
	}
	if len(res.Tokens) != 3 || res.Tokens[2] != 3 {
		t.Fatalf("expected bonus token 3, got %v", res.Tokens)
	}
}

func TestSpeculativeDecodeLogitsGreedyReject(t *testing.T) {
	draft := []Token{1, 2}
	mk := func(arg int) Logits {
		l := make(Logits, 4)
		l[arg] = 10
		return l
	}
	draftLogits := []Logits{mk(1), mk(2)}
	// Second target position disagrees (argmax 0, not draft 2).
	targetLogits := []Logits{mk(1), mk(0), mk(3)}
	cfg := DefaultSamplingConfig()
	cfg.Temperature = 0
	randoms := []float32{0.1, 0.1, 0.1}
	res, err := SpeculativeDecodeLogits(draft, draftLogits, targetLogits, cfg, randoms)
	if err != nil {
		t.Fatal(err)
	}
	if res.AcceptedDraftTokens != 1 {
		t.Fatalf("accepted = %d, want 1", res.AcceptedDraftTokens)
	}
	if !res.UsedResidualFallback {
		t.Fatal("rejection should set residual fallback")
	}
	if res.Tokens[len(res.Tokens)-1] != 0 {
		t.Fatalf("corrected token = %d, want 0", res.Tokens[len(res.Tokens)-1])
	}
}

func TestSpeculativeDecodeLogitsInvalid(t *testing.T) {
	cfg := DefaultSamplingConfig()
	if _, err := SpeculativeDecodeLogits(nil, nil, nil, cfg, nil); err == nil {
		t.Fatal("expected error on empty inputs")
	}
}

func TestSampleMirostatV2Validation(t *testing.T) {
	logits := Logits{1, 2, 3, 4}
	cfg := DefaultMirostat()
	// Invalid temperature.
	if _, _, err := SampleMirostatV2(logits, 0, cfg, 5.0, 0.5); err == nil {
		t.Fatal("expected error on zero temperature")
	}
	// Invalid random.
	if _, _, err := SampleMirostatV2(logits, 1.0, cfg, 5.0, 1.5); err == nil {
		t.Fatal("expected error on out-of-range random")
	}
	// Invalid mu.
	if _, _, err := SampleMirostatV2(logits, 1.0, cfg, float32(math.Inf(1)), 0.5); err == nil {
		t.Fatal("expected error on non-finite mu")
	}
	// Valid call returns updated mu.
	tok, mu, err := SampleMirostatV2(logits, 1.0, cfg, 5.0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if int(tok) >= len(logits) {
		t.Fatalf("token %d out of range", tok)
	}
	if math.IsNaN(float64(mu)) || math.IsInf(float64(mu), 0) {
		t.Fatalf("updated mu not finite: %v", mu)
	}
}

func TestBeamSearchLogitsEOSStops(t *testing.T) {
	mk := func(arg int) Logits {
		l := make(Logits, 3)
		l[arg] = 10
		return l
	}
	eos := Token(2)
	// Step 0 strongly prefers token 2 (EOS).
	steps := []Logits{mk(2), mk(1), mk(0)}
	res, err := BeamSearchLogits(steps, 2, &eos)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Tokens) == 0 {
		t.Fatal("beam search returned no tokens")
	}
	// Best beam should have finished at EOS on the first step.
	if res.Tokens[0] != 2 {
		t.Fatalf("first token = %d, want 2 (eos)", res.Tokens[0])
	}
}

func TestBeamSearchLogitsInvalid(t *testing.T) {
	if _, err := BeamSearchLogits(nil, 1, nil); err == nil {
		t.Fatal("expected error on empty input")
	}
	if _, err := BeamSearchLogits([]Logits{{1, 2}}, 0, nil); err == nil {
		t.Fatal("expected error on zero beam width")
	}
}

func TestSpeculativeStatsRecordStep(t *testing.T) {
	var s SpeculativeStats
	s.RecordStep(SpeculativeVerifyResult{AcceptedDraftTokens: 3, UsedResidualFallback: false}, 4)
	s.RecordStep(SpeculativeVerifyResult{AcceptedDraftTokens: 1, UsedResidualFallback: true}, 4)
	if s.TotalDraftTokens != 8 {
		t.Fatalf("total drafts = %d, want 8", s.TotalDraftTokens)
	}
	if s.TotalAcceptedTokens != 4 {
		t.Fatalf("accepted = %d, want 4", s.TotalAcceptedTokens)
	}
	if s.FallbackTokens != 1 {
		t.Fatalf("fallback = %d, want 1", s.FallbackTokens)
	}
	if got := s.DraftAcceptanceRate(); got != 0.5 {
		t.Fatalf("acceptance rate = %v, want 0.5", got)
	}
	if s.TargetForwardPasses != 2 {
		t.Fatalf("target passes = %d, want 2", s.TargetForwardPasses)
	}
	if got := s.TokensPerTargetForward(); got <= 0 {
		t.Fatalf("tokens/forward = %v, want > 0", got)
	}
}
