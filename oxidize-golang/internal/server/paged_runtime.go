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

// PagedRuntimeV2 couples the budgeted three-phase SchedulerV2 with per-sequence
// sessions. It enforces a token budget per step, supports prefill chunking and
// prefix caching, and builds an InputBatch each step so multiple sequences can
// be processed together. The model forward is still driven per-sequence (one
// session each) because the pure-Go model backend does not yet expose a fused
// multi-sequence kernel, but scheduling, batching metadata, and block
// management mirror the Rust scheduler.
type PagedRuntimeV2 struct {
	mu       sync.Mutex
	sched    *paged.SchedulerV2
	model    model.Model
	sessions map[int]*model.Session
	// prefilled tracks how many prompt tokens have been pushed through the
	// model session for each sequence (decode resumes from there).
	modelPrefilled map[int]int
}

// NewPagedRuntimeV2 loads a GGUF model and starts a budgeted v2 scheduler.
func NewPagedRuntimeV2(modelPath string, cfg paged.SchedulerV2Config, totalBlocks, blockSize int) (*PagedRuntimeV2, error) {
	m, err := model.LoadGGUFModelFromPath(modelPath, model.NewLoaderConfig())
	if err != nil {
		return nil, fmt.Errorf("paged runtime v2: %w", err)
	}
	return &PagedRuntimeV2{
		sched:          paged.NewSchedulerV2(cfg, totalBlocks, blockSize),
		model:          m,
		sessions:       make(map[int]*model.Session),
		modelPrefilled: make(map[int]int),
	}, nil
}

// Enqueue adds a prompt to the v2 scheduler and returns its sequence id.
func (r *PagedRuntimeV2) Enqueue(promptTokens []int, maxNewTokens, stopToken int, hasStop bool) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	id := r.sched.AddRequest(promptTokens, maxNewTokens, stopToken, hasStop)
	r.sessions[id] = model.NewSession()
	r.modelPrefilled[id] = 0
	return id
}

// Step runs one scheduler step, processes the resulting InputBatch through the
// model, samples tokens, and postprocesses (finish detection + block reclaim).
// It returns the sampled tokens keyed by sequence id.
func (r *PagedRuntimeV2) Step(ctx context.Context) (map[int]model.Token, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	res, err := r.sched.Step()
	if err != nil {
		return nil, err
	}
	if len(res.ScheduledSeqIDs) == 0 {
		return map[int]model.Token{}, nil
	}
	batch := r.sched.BuildInputBatch(res)

	sampled := make(map[int]int, batch.BatchSize)
	out := make(map[int]model.Token, batch.BatchSize)
	for i, id := range batch.SeqIDs {
		sess := r.sessions[id]
		if sess == nil {
			sess = model.NewSession()
			r.sessions[id] = sess
		}
		toks := intTokensToModel(batch.TokenIDs[i])
		if len(toks) == 0 {
			continue
		}
		logits, err := r.model.Forward(toks, sess)
		if err != nil {
			return nil, err
		}
		// Only sequences finishing their prefill (or decoding) produce a
		// sampled token this step. For a prefill chunk that does not yet reach
		// the end of the prompt, we keep accumulating context and skip
		// sampling so we do not emit mid-prompt tokens.
		if batch.IsPrefill[i] {
			seq, ok := r.sched.GetSequence(id)
			if !ok || seq.RemainingPrefillTokens() > 0 {
				continue
			}
		}
		next, err := model.Greedy(logits)
		if err != nil {
			return nil, err
		}
		sampled[id] = int(next)
		out[id] = next
	}

	if err := r.sched.PostprocessStep(sampled); err != nil {
		return nil, err
	}
	// Reap fully finished sequences' sessions.
	for id := range out {
		if seq, ok := r.sched.GetSequence(id); ok && seq.Status() == paged.StatusFinished {
			delete(r.sessions, id)
			delete(r.modelPrefilled, id)
		}
	}
	return out, nil
}

// Stats returns waiting/running counts and free block count.
func (r *PagedRuntimeV2) Stats() (waiting, running, free int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.sched.WaitingCount(), r.sched.RunningCount(), r.sched.Pool().FreeCount()
}
