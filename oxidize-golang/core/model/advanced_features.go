package model

import (
	"math"
	"sort"
)

// negInfF32 is a portable way to represent -infinity in float32 contexts.
var negInfF32 = float32(math.Inf(-1))

// XtcSamplerConfig mirrors XtcSamplerConfig.
type XtcSamplerConfig struct {
	Threshold float32
	MinKeep   int
}

// DrySamplerConfig mirrors DrySamplerConfig.
type DrySamplerConfig struct {
	Multiplier    float32
	Base          float32
	AllowedLength int
	Range         int
}

// DynamicTempConfig mirrors DynamicTempConfig.
type DynamicTempConfig struct {
	MinTemp  float32
	MaxTemp  float32
	Exponent float32
}

// SamplerChain mirrors SamplerChain.
type SamplerChain struct {
	Steps []SamplerStep
}

// SamplerStep is a single filter in a sampler chain.
type SamplerStep interface {
	Apply(logits Logits, ctx *SamplerContext) Logits
}

// SamplerContext carries state across steps.
type SamplerContext struct {
	History     []Token
	RNGSeed     int64
	Position    int
	Temperature float32
	Custom      map[string]any
}

// NewSamplerChain constructs an empty chain.
func NewSamplerChain() *SamplerChain { return &SamplerChain{} }

// Add appends a step.
func (c *SamplerChain) Add(step SamplerStep) { c.Steps = append(c.Steps, step) }

// Run runs all steps in order.
func (c *SamplerChain) Run(logits Logits, ctx *SamplerContext) Logits {
	for _, step := range c.Steps {
		logits = step.Apply(logits, ctx)
	}
	return logits
}

// TemperatureStep scales logits by 1/temperature.
type TemperatureStep struct{ Temp float32 }

// Apply implements SamplerStep.
func (s *TemperatureStep) Apply(logits Logits, _ *SamplerContext) Logits {
	if s.Temp == 0 || s.Temp == 1.0 {
		return logits
	}
	out := make(Logits, len(logits))
	inv := 1.0 / float32(s.Temp)
	for i, v := range logits {
		out[i] = v * inv
	}
	return out
}

// TopKStep keeps the top K tokens.
type TopKStep struct{ K int }

// Apply implements SamplerStep.
func (s *TopKStep) Apply(logits Logits, _ *SamplerContext) Logits { return topK(logits, s.K) }

// TopPStep keeps tokens until cumulative probability exceeds P.
type TopPStep struct{ P float32 }

// Apply implements SamplerStep.
func (s *TopPStep) Apply(logits Logits, _ *SamplerContext) Logits { return topP(logits, s.P) }

// MinPStep keeps tokens with probability above MinP * max(prob).
type MinPStep struct{ P float32 }

// Apply implements SamplerStep.
func (s *MinPStep) Apply(logits Logits, _ *SamplerContext) Logits { return minP(logits, s.P) }

// TypicalPStep implements typical-p sampling.
type TypicalPStep struct{ P float32 }

// Apply implements SamplerStep.
func (s *TypicalPStep) Apply(logits Logits, _ *SamplerContext) Logits {
	if s.P <= 0 || len(logits) == 0 {
		return logits
	}
	probs := softmax(logits)
	entropy := float32(0)
	for _, p := range probs {
		if p > 0 {
			entropy -= p * float32(math.Log(float64(p)))
		}
	}
	type cand struct {
		idx int
		diff float32
	}
	cands := make([]cand, len(probs))
	for i, p := range probs {
		cands[i] = cand{idx: i, diff: float32(math.Abs(float64(-float32(math.Log(float64(p)+1e-12)) - entropy)))}
	}
	_ = cands
	sort.Slice(cands, func(i, j int) bool { return cands[i].diff < cands[j].diff })
	var cumProb float32
	keep := make([]bool, len(probs))
	for _, c := range cands {
		keep[c.idx] = true
		cumProb += probs[c.idx]
		if cumProb >= s.P {
			break
		}
	}
	for i, k := range keep {
		if !k {
			logits[i] = negInfF32
		}
	}
	return logits
}

// TailFreeStep implements tail-free sampling.
type TailFreeStep struct{ Z float32 }

// Apply implements SamplerStep.
func (s *TailFreeStep) Apply(logits Logits, _ *SamplerContext) Logits {
	if s.Z <= 0 || len(logits) < 3 {
		return logits
	}
	idx := sortedIndices(logits)
	probs := softmax(logits)
	secondDerivs := make([]float32, len(idx)-2)
	for i := 1; i < len(idx)-1; i++ {
		secondDerivs[i-1] = float32(math.Abs(float64(probs[idx[i+1]] - 2*probs[idx[i]] + probs[idx[i-1]])))
	}
	var total float32
	for _, sd := range secondDerivs {
		total += sd
	}
	keep := make([]bool, len(idx))
	cumZ := float32(0)
	for i, sd := range secondDerivs {
		if total > 0 {
			cumZ += sd / total
		}
		if cumZ < s.Z {
			keep[i+1] = true
		} else {
			break
		}
	}
	keep[0] = true
	out := make(Logits, len(logits))
	for i := range out {
		out[i] = negInfF32
	}
	for _, k := range idx {
		if keep[lowerIndex(idx, k)] {
			out[k] = logits[k]
		}
	}
	return out
}

func lowerIndex(idx []int, target int) int {
	for i, v := range idx {
		if v == target {
			return i
		}
	}
	return 0
}

// DryStep implements DRY (Don't Repeat Yourself) sampling.
type DryStep struct{ Cfg DrySamplerConfig }

// Apply implements SamplerStep.
func (s *DryStep) Apply(logits Logits, ctx *SamplerContext) Logits {
	if s.Cfg.Multiplier == 0 || ctx == nil || len(ctx.History) < 2 {
		return logits
	}
	rangeLen := s.Cfg.Range
	if rangeLen <= 0 {
		rangeLen = len(ctx.History)
	}
	if rangeLen > len(ctx.History) {
		rangeLen = len(ctx.History)
	}
	out := append(Logits(nil), logits...)
	allowed := s.Cfg.AllowedLength
	if allowed <= 0 {
		allowed = 1
	}
	for n := allowed; n < rangeLen; n++ {
		if len(ctx.History) < n*2 {
			continue
		}
		a := ctx.History[len(ctx.History)-n:]
		b := ctx.History[len(ctx.History)-2*n : len(ctx.History)-n]
		equal := true
		for i := range a {
			if a[i] != b[i] {
				equal = false
				break
			}
		}
		if !equal {
			continue
		}
		for _, t := range a {
			if int(t) < len(out) {
				out[t] -= s.Cfg.Multiplier * s.Cfg.Base
			}
		}
	}
	return out
}

// XtcStep implements XTC sampling.
type XtcStep struct{ Cfg XtcSamplerConfig }

// Apply implements SamplerStep.
func (s *XtcStep) Apply(logits Logits, _ *SamplerContext) Logits {
	if s.Cfg.Threshold <= 0 || len(logits) == 0 {
		return logits
	}
	idx := sortedIndices(logits)
	probs := softmax(logits)
	removed := 0
	for _, i := range idx {
		if len(probs)-removed <= s.Cfg.MinKeep {
			break
		}
		if probs[i] > s.Cfg.Threshold {
			logits[i] = negInfF32
			removed++
		}
	}
	return logits
}

// DynamicTempStep implements dynamic temperature.
type DynamicTempStep struct{ Cfg DynamicTempConfig }

// Apply implements SamplerStep.
func (s *DynamicTempStep) Apply(logits Logits, ctx *SamplerContext) Logits {
	if s.Cfg.MinTemp == 0 && s.Cfg.MaxTemp == 0 {
		return logits
	}
	if ctx == nil {
		return logits
	}
	if s.Cfg.Exponent == 0 {
		ctx.Temperature = 1.0
		return logits
	}
	ctx.Temperature = s.Cfg.MinTemp + (s.Cfg.MaxTemp-s.Cfg.MinTemp)*float32(math.Pow(float64(ctx.Position), float64(s.Cfg.Exponent)))
	return s.applyWithTemp(logits, ctx.Temperature)
}

func (s *DynamicTempStep) applyWithTemp(logits Logits, temp float32) Logits {
	if temp == 0 || temp == 1.0 {
		return logits
	}
	out := make(Logits, len(logits))
	inv := 1.0 / temp
	for i, v := range logits {
		out[i] = v * inv
	}
	return out
}

// NewlinePenaltyStep implements a newline penalty.
type NewlinePenaltyStep struct{ Cfg NewlinePenalty }

// Apply implements SamplerStep.
func (s *NewlinePenaltyStep) Apply(logits Logits, _ *SamplerContext) Logits {
	if s.Cfg.Count <= 0 {
		return logits
	}
	out := append(Logits(nil), logits...)
	if int(Token('\n')) < len(out) {
		out[Token('\n')] += s.Cfg.Reward * float32(s.Cfg.Count)
	}
	return out
}

// Mirror Mirostat-only default chain
var DefaultMirostatConfig = MirostatConfig{Tau: 5.0, Eta: 0.1, M: 5}

// ToolFunction mirrors ToolFunction.
type ToolFunction struct {
	Name        string
	Description string
	Parameters  map[string]any
}

// ToolCall mirrors ToolCall.
type ToolCall struct {
	Name      string
	Arguments map[string]any
}

// ToolFormat describes how tool calls are encoded.
type ToolFormat string

// Recognised tool formats.
const (
	ToolFormatJSON  ToolFormat = "json"
	ToolFormatYAML  ToolFormat = "yaml"
	ToolFormatNative ToolFormat = "native"
)

// ToolRegistry mirrors ToolRegistry.
type ToolRegistry struct {
	tools map[string]ToolFunction
}

// NewToolRegistry constructs an empty registry.
func NewToolRegistry() *ToolRegistry { return &ToolRegistry{tools: map[string]ToolFunction{}} }

// Register adds a tool function.
func (r *ToolRegistry) Register(t ToolFunction) { r.tools[t.Name] = t }

// Get returns the named tool.
func (r *ToolRegistry) Get(name string) (ToolFunction, bool) {
	t, ok := r.tools[name]
	return t, ok
}

// Names returns the registered tool names.
func (r *ToolRegistry) Names() []string {
	out := make([]string, 0, len(r.tools))
	for k := range r.tools {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// EngineConfig mirrors EngineConfig.
type EngineConfig struct {
	Sampling  SamplingConfig
	Template  string
	StopTokens []Token
	Tools     []ToolFunction
}

// NewEngineConfig returns an engine config with defaults.
func NewEngineConfig() EngineConfig { return EngineConfig{Sampling: DefaultSamplingConfig()} }
