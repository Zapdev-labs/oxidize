package model

import (
	"errors"
	"math"
)

// LoraLayer mirrors LoraLayer.
type LoraLayer struct {
	Name      string
	Rank      int
	Alpha     float32
	Scale     float32
	BaseShape []int
	UpLoaded  bool
	DownLoaded bool
}

// NewLoraLayer constructs a layer placeholder.
func NewLoraLayer(name string, rank int, alpha float32, baseShape []int) LoraLayer {
	scale := float32(1.0)
	if alpha > 0 && rank > 0 {
		scale = alpha / float32(rank)
	}
	return LoraLayer{Name: name, Rank: rank, Alpha: alpha, Scale: scale, BaseShape: baseShape}
}

// LoraAdapter mirrors LoraAdapter.
type LoraAdapter struct {
	Name     string
	Path     string
	Layers   map[string]LoraLayer
	Rank     int
	Alpha    float32
	BaseModel Model
}

// NewLoraAdapter constructs an empty adapter for a given base model.
func NewLoraAdapter(name, path string, base Model) *LoraAdapter {
	return &LoraAdapter{Name: name, Path: path, Layers: map[string]LoraLayer{}, BaseModel: base}
}

// AddLayer registers a LoRA layer with the adapter.
func (a *LoraAdapter) AddLayer(layer LoraLayer) {
	a.Layers[layer.Name] = layer
	if layer.Rank > 0 && a.Rank == 0 {
		a.Rank = layer.Rank
	}
	if layer.Alpha > 0 && a.Alpha == 0 {
		a.Alpha = layer.Alpha
	}
}

// Apply runs the adapter on top of the base model's forward pass.
func (a *LoraAdapter) Apply(tokens []Token, session *Session) (Logits, error) {
	if a.BaseModel == nil {
		return nil, errors.New("lora: base model is nil")
	}
	logits, err := a.BaseModel.Forward(tokens, session)
	if err != nil {
		return nil, err
	}
	if a.Rank > 0 && a.Alpha > 0 {
		scale := a.Alpha / float32(a.Rank)
		for i := range logits {
			logits[i] *= scale
		}
	}
	return logits, nil
}

// Forward implements Model by delegating to Apply.
func (a *LoraAdapter) Forward(tokens []Token, session *Session) (Logits, error) { return a.Apply(tokens, session) }

// LoraError mirrors LoraError.
type LoraError struct{ Message string }

func (e *LoraError) Error() string { return "lora: " + e.Message }

// LoraPlan mirrors LoraPlan.
type LoraPlan struct {
	Adapters      []*LoraAdapter
	MergeStrategy string // "sequential", "parallel", "tying"
	RankBudget    int
	EstimatedGain float32
}

// NewLoraPlan returns an empty plan.
func NewLoraPlan() *LoraPlan { return &LoraPlan{MergeStrategy: "sequential"} }

// Add attaches an adapter to the plan.
func (p *LoraPlan) Add(a *LoraAdapter) { p.Adapters = append(p.Adapters, a) }

// Validate returns nil if the plan is internally consistent.
func (p *LoraPlan) Validate() error {
	if p.MergeStrategy == "" {
		return &LoraError{Message: "merge strategy is required"}
	}
	if len(p.Adapters) == 0 {
		return &LoraError{Message: "no adapters in plan"}
	}
	switch p.MergeStrategy {
	case "sequential", "parallel", "tying":
	default:
		return &LoraError{Message: "unknown merge strategy: " + p.MergeStrategy}
	}
	return nil
}

// MergeAdapters produces a single consolidated LoraAdapter.
func (p *LoraPlan) MergeAdapters() *LoraAdapter {
	merged := NewLoraAdapter("merged", "", nil)
	merged.Rank = p.RankBudget
	for _, a := range p.Adapters {
		for _, l := range a.Layers {
			merged.AddLayer(l)
		}
	}
	merged.Alpha = float32(merged.Rank) * p.EstimatedGain
	return merged
}

// EstimateMemory estimates the memory cost in bytes for the plan.
func (p *LoraPlan) EstimateMemory() int64 {
	var total int64
	for _, a := range p.Adapters {
		for _, l := range a.Layers {
			r := int64(l.Rank)
			if r == 0 {
				r = 1
			}
			dim := 1
			for _, d := range l.BaseShape {
				dim *= d
			}
			total += int64(r) * int64(dim) * 4 // 2 fp32 matrices
		}
	}
	return total
}

// RankBudgetExceededError mirrors RankBudgetExceededError.
type RankBudgetExceededError struct{ Requested, Available int }

func (e *RankBudgetExceededError) Error() string {
	return "lora: rank budget exceeded"
}

// ValidateRankBudget ensures the requested rank does not exceed the budget.
func (p *LoraPlan) ValidateRankBudget() error {
	for _, a := range p.Adapters {
		if a.Rank > p.RankBudget {
			return &RankBudgetExceededError{Requested: a.Rank, Available: p.RankBudget}
		}
	}
	return nil
}

// LoraScalingConfig mirrors LoraScalingConfig.
type LoraScalingConfig struct {
	Auto    bool
	Manual  float32
	MaxGain float32
}

// ScaleFactor returns the LoRA scaling factor.
func (c LoraScalingConfig) ScaleFactor(rank int) float32 {
	if c.Auto {
		if rank == 0 {
			return 0
		}
		return float32(math.Min(float64(c.MaxGain), 1.0))
	}
	return c.Manual
}
