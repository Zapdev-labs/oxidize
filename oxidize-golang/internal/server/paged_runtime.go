package server

import (
	"context"
	"fmt"
	"sync"

	"github.com/Zapdev-labs/oxidize/golang/core/model"
	"github.com/Zapdev-labs/oxidize/golang/core/paged"
)

// PagedRuntime couples the vLLM-style scheduler with batched decode steps on one model.
type PagedRuntime struct {
	mu        sync.Mutex
	sched     *paged.Scheduler
	model     model.Model
	sessions  map[int]*model.Session
	prompts   map[int][]int
	maxTok    map[int]int
	generated map[int]int
}

// NewPagedRuntime loads a GGUF model and starts a paged scheduler.
func NewPagedRuntime(modelPath string, schedCfg paged.SchedulerConfig) (*PagedRuntime, error) {
	m, err := model.LoadGGUFModelFromPath(modelPath, model.NewLoaderConfig())
	if err != nil {
		return nil, fmt.Errorf("paged runtime: %w", err)
	}
	return &PagedRuntime{
		sched:     paged.NewScheduler(schedCfg),
		model:     m,
		sessions:  make(map[int]*model.Session),
		prompts:   make(map[int][]int),
		maxTok:    make(map[int]int),
		generated: make(map[int]int),
	}, nil
}

// Enqueue adds a prompt to the scheduler queue.
func (r *PagedRuntime) Enqueue(promptTokens []int, maxNewTokens int) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	req, err := r.sched.AddRequest(promptTokens, maxNewTokens)
	if err != nil {
		return 0, err
	}
	id := req.ID
	r.prompts[id] = append([]int(nil), promptTokens...)
	r.maxTok[id] = maxNewTokens
	r.generated[id] = 0
	r.sessions[id] = model.NewSession()
	return id, nil
}

// Step runs one scheduler step and decodes one token per active request.
func (r *PagedRuntime) Step(ctx context.Context) (map[int]model.Token, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	active, err := r.sched.Step()
	if err != nil {
		return nil, err
	}
	out := make(map[int]model.Token, len(active))
	for _, req := range active {
		id := req.ID
		sess := r.sessions[id]
		tokens := r.prompts[id]
		if r.generated[id] == 0 && len(tokens) > 1 {
			if _, err := r.model.Forward(intTokensToModel(tokens[:len(tokens)-1]), sess); err != nil {
				return nil, err
			}
		}
		last := tokens[len(tokens)-1]
		logits, err := r.model.Forward([]model.Token{model.Token(last)}, sess)
		if err != nil {
			return nil, err
		}
		next, err := model.Greedy(logits)
		if err != nil {
			return nil, err
		}
		out[id] = next
		r.prompts[id] = append(tokens, int(next))
		r.generated[id]++
		if r.generated[id] >= r.maxTok[id] || next == 2 {
			r.sched.Finish(id)
			delete(r.sessions, id)
			delete(r.prompts, id)
			delete(r.maxTok, id)
			delete(r.generated, id)
		}
	}
	return out, nil
}

// Stats returns scheduler queue/active/free block counts.
func (r *PagedRuntime) Stats() (queue, active, free int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.sched.Stats()
}

func intTokensToModel(tokens []int) []model.Token {
	out := make([]model.Token, len(tokens))
	for i, t := range tokens {
		out[i] = model.Token(t)
	}
	return out
}
