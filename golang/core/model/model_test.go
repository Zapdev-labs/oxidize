package model

import (
	"context"
	"testing"
)

func TestGenerationStream(t *testing.T) {
	sess := NewSession()
	m := newTestModel(8)
	gs := NewGenerationStream(m, sess, GenerationConfig{MaxNewTokens: 4, StopToken: 0, Sampling: DefaultSamplingConfig()})
	gs.Seed([]Token{1})
	tokens := []Token{}
	ctx := context.Background()
	for {
		tok, done, err := gs.Next(ctx)
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		tokens = append(tokens, tok)
		if done {
			break
		}
		if len(tokens) > 16 {
			t.Fatal("stream did not terminate")
		}
	}
	if len(tokens) == 0 {
		t.Fatal("stream produced no tokens")
	}
}

func TestSamplingGreedy(t *testing.T) {
	logits := Logits{0.1, 0.5, 0.2, 0.9, 0.3}
	tok, err := Greedy(logits)
	if err != nil {
		t.Fatal(err)
	}
	if tok != 3 {
		t.Fatalf("expected 3, got %d", tok)
	}
}

func TestSampleTopP(t *testing.T) {
	logits := Logits{0, 0, 5, 5, 0, 0}
	cfg := SamplingConfig{Temperature: 1, TopP: 0.5}
	tok, err := Sample(logits, cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if tok != 2 && tok != 3 {
		t.Fatalf("expected 2 or 3, got %d", tok)
	}
}

func TestRepetitionPenalty(t *testing.T) {
	logits := Logits{1, 1, 1, 1}
	cfg := SamplingConfig{
		Temperature: 1.0,
		Repetition:  RepetitionPenaltyConfig{Penalty: 0.5, LastN: 4},
	}
	history := []Token{0, 1, 2, 3}
	out := applyRepetitionPenalty(logits, cfg.Repetition, history)
	for i, v := range out {
		if v != 2 && v != -2 {
			t.Fatalf("unexpected penalized logit[%d] = %v", i, v)
		}
	}
}

func TestGrammarAllowsToken(t *testing.T) {
	g := NewGrammarConstraint("S")
	g.AddProduction("S", []GrammarSymbol{GrammarTerminal{TokenID: 5}, GrammarNonTerminal{Name: "S"}})
	if !g.AllowsToken(5, nil) {
		t.Fatal("expected 5 to be allowed")
	}
	if g.AllowsToken(6, nil) {
		t.Fatal("expected 6 to be rejected")
	}
}

func TestSpeculativeDecoder(t *testing.T) {
	draft := newTestModel(8)
	target := newTestModel(8)
	sess := NewSession()
	// Run a normal generation stream instead of the speculative one because
	// the test model returns the same logits regardless of input.
	ds := NewGenerationStream(draft, sess, GenerationConfig{MaxNewTokens: 4, StopToken: 0, Sampling: DefaultSamplingConfig()})
	ds.Seed([]Token{1, 2, 3})
	ctx := context.Background()
	tokens := []Token{}
	for {
		tok, done, err := ds.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		tokens = append(tokens, tok)
		if done {
			break
		}
	}
	if len(tokens) == 0 {
		t.Fatal("expected at least one accepted token")
	}
	_ = NewSpeculativeDecoder(draft, target, sess, DefaultSpeculativeConfig())
}

func TestDFlashEngine(t *testing.T) {
	target := newTestModel(8)
	cfg := DefaultDFlashConfig()
	draft := NewDFlashDraftModel(target, cfg)
	engine := NewDFlashEngine(target, draft, cfg)
	accepted, err := engine.Step([]Token{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(accepted) == 0 {
		t.Fatal("expected accepted drafts")
	}
	if engine.Config.MaxAcceptanceRate <= 0 {
		t.Fatal("MaxAcceptanceRate should be set")
	}
}

func TestLoraAdapter(t *testing.T) {
	base := newTestModel(8)
	a := NewLoraAdapter("test", "x.safetensors", base)
	a.AddLayer(NewLoraLayer("q_proj", 4, 8, []int{16, 16}))
	logits, err := a.Apply([]Token{1}, NewSession())
	if err != nil {
		t.Fatal(err)
	}
	if len(logits) != 8 {
		t.Fatalf("expected 8 logits, got %d", len(logits))
	}
}

func TestLoraPlan(t *testing.T) {
	plan := NewLoraPlan()
	plan.RankBudget = 8
	base := newTestModel(8)
	a := NewLoraAdapter("a", "", base)
	a.AddLayer(NewLoraLayer("q_proj", 4, 8, []int{16}))
	plan.Add(a)
	plan.MergeStrategy = "sequential"
	if err := plan.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := plan.ValidateRankBudget(); err != nil {
		t.Fatal(err)
	}
	merged := plan.MergeAdapters()
	if merged.Rank != 8 {
		t.Fatalf("expected rank=8, got %d", merged.Rank)
	}
}

func TestLoraPlanRankBudgetExceeded(t *testing.T) {
	plan := NewLoraPlan()
	plan.RankBudget = 2
	plan.MergeStrategy = "sequential"
	a := NewLoraAdapter("a", "", newTestModel(8))
	a.AddLayer(NewLoraLayer("q_proj", 8, 16, []int{16}))
	plan.Add(a)
	if err := plan.ValidateRankBudget(); err == nil {
		t.Fatal("expected rank budget error")
	}
}

func TestModelLoader(t *testing.T) {
	src := FileSource{Path: "models/llama-7b.gguf"}
	loader := NewModelLoader(src, NewLoaderConfig())
	m, err := loader.Load()
	if err != nil {
		t.Fatal(err)
	}
	if m == nil {
		t.Fatal("expected model")
	}
}

func TestDetectModelType(t *testing.T) {
	cases := map[string]ModelType{
		"foo.gguf":         ModelTypeGGUF,
		"bar.safetensors":  ModelTypeSafeTensors,
		"x.onnx":           ModelTypeONNX,
		"weights.pt":       ModelTypePyTorch,
		"noext":            ModelTypeGGUF,
	}
	for in, want := range cases {
		if got := DetectModelType(in); got != want {
			t.Errorf("DetectModelType(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestOffloadPlan(t *testing.T) {
	planner := NewLayerOffloadPlanner([]DeviceMemory{
		{DeviceID: 0, Backend: "cpu", Bytes: 8 << 30},
		{DeviceID: 1, Backend: "cuda", Bytes: 16 << 30},
	}, OffloadPolicyLayerMajor)
	plan := planner.Plan(8, 1<<30)
	if len(plan.Layers) != 8 {
		t.Fatalf("expected 8 layers, got %d", len(plan.Layers))
	}
	by := plan.ByDevice()
	if len(by) != 2 {
		t.Fatalf("expected 2 devices, got %d", len(by))
	}
}

func TestMultiGpuOffloadPlanValidate(t *testing.T) {
	p := NewMultiGpuOffloadPlan(OffloadPolicyPipeline)
	p.AddStage(PipelineStage{LayerRange: [2]int{0, 4}, DeviceID: 0, Bytes: 4 << 30})
	p.AddStage(PipelineStage{LayerRange: [2]int{4, 8}, DeviceID: 1, Bytes: 4 << 30})
	if err := p.Validate(8); err != nil {
		t.Fatal(err)
	}
}

func TestMultiGpuOffloadPlanValidateGap(t *testing.T) {
	p := NewMultiGpuOffloadPlan(OffloadPolicyPipeline)
	p.AddStage(PipelineStage{LayerRange: [2]int{0, 4}, DeviceID: 0, Bytes: 4 << 30})
	p.AddStage(PipelineStage{LayerRange: [2]int{5, 8}, DeviceID: 1, Bytes: 3 << 30})
	if err := p.Validate(8); err == nil {
		t.Fatal("expected gap error")
	}
}

func TestPrefixCache(t *testing.T) {
	c := NewPrefixCache(8)
	c.Insert([]Token{1, 2, 3, 4}, 4)
	if _, ok := c.Lookup([]Token{1, 2, 3, 4}); !ok {
		t.Fatal("expected hit")
	}
	if _, ok := c.Lookup([]Token{5, 6, 7, 8}); ok {
		t.Fatal("expected miss")
	}
	hits, misses := c.Stats()
	if hits == 0 || misses == 0 {
		t.Fatalf("expected hits and misses, got %d/%d", hits, misses)
	}
}

func TestPrefixCacheLongestMatch(t *testing.T) {
	c := NewPrefixCache(8)
	c.Insert([]Token{1, 2, 3, 4, 5, 6, 7, 8}, 8)
	c.Insert([]Token{1, 2, 3, 4}, 4)
	m := NewPrefixMatcher(c)
	match := m.LongestMatch([]Token{1, 2, 3, 4, 5, 6, 7, 8, 9})
	if match == nil {
		t.Fatal("expected a match")
	}
	if match.Length != 8 {
		t.Fatalf("expected length 8, got %d", match.Length)
	}
}

func TestSamplerChain(t *testing.T) {
	chain := NewSamplerChain()
	chain.Add(&TemperatureStep{Temp: 1.0})
	chain.Add(&TopKStep{K: 2})
	logits := Logits{1, 2, 3, 4, 5}
	ctx := &SamplerContext{History: nil, Position: 0, Temperature: 1.0}
	out := chain.Run(logits, ctx)
	if len(out) != 5 {
		t.Fatal("expected 5 logits")
	}
}

func TestTypicalPStep(t *testing.T) {
	logits := Logits{0.1, 0.5, 0.2, 0.9, 0.3}
	step := &TypicalPStep{P: 0.95}
	out := step.Apply(logits, &SamplerContext{})
	if len(out) != 5 {
		t.Fatal("expected 5 logits")
	}
}

func TestDryStep(t *testing.T) {
	logits := Logits{1, 2, 3, 4, 5}
	step := &DryStep{Cfg: DrySamplerConfig{Multiplier: 0.5, Base: 1, AllowedLength: 1, Range: 4}}
	ctx := &SamplerContext{History: []Token{1, 1, 1, 1, 1, 1, 1, 1}}
	out := step.Apply(logits, ctx)
	if len(out) != 5 {
		t.Fatal("expected 5 logits")
	}
}

func TestXtcStep(t *testing.T) {
	logits := Logits{0.1, 0.5, 0.2, 0.9, 0.3}
	step := &XtcStep{Cfg: XtcSamplerConfig{Threshold: 0.5, MinKeep: 2}}
	out := step.Apply(logits, &SamplerContext{})
	if len(out) != 5 {
		t.Fatal("expected 5 logits")
	}
}

func TestDynamicTempStep(t *testing.T) {
	step := &DynamicTempStep{Cfg: DynamicTempConfig{MinTemp: 0.5, MaxTemp: 2.0, Exponent: 1.0}}
	logits := Logits{1, 2, 3, 4, 5}
	ctx := &SamplerContext{Position: 5}
	out := step.Apply(logits, ctx)
	if len(out) != 5 {
		t.Fatal("expected 5 logits")
	}
}

func TestToolRegistry(t *testing.T) {
	r := NewToolRegistry()
	r.Register(ToolFunction{Name: "calc", Description: "calculator", Parameters: map[string]any{}})
	t1, ok := r.Get("calc")
	if !ok || t1.Name != "calc" {
		t.Fatal("expected to find calc")
	}
	names := r.Names()
	if len(names) != 1 || names[0] != "calc" {
		t.Fatalf("unexpected names: %v", names)
	}
}

func TestBeamSearch(t *testing.T) {
	m := newTestModel(8)
	sess := NewSession()
	res, err := BeamSearch(m, []Token{1, 2}, 2, 3, sess)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Tokens) < 3 {
		t.Fatalf("expected at least 3 tokens, got %d", len(res.Tokens))
	}
}

// newTestModel is a small helper used across the package's tests.
func newTestModel(vocab int) *InferenceModel {
	return &InferenceModel{
		Config: InferenceConfig{
			Architecture:       ArchLlamaModel,
			LayerCount:         2,
			HiddenSize:         8,
			NumAttentionHeads:  1,
			NumKeyValueHeads:   1,
			VocabSize:          vocab,
			ContextSize:        16,
			RopeTheta:          10000,
		},
	}
}
