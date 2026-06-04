package generate

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/Zapdev-labs/oxidize/golang/core/model"
	"github.com/Zapdev-labs/oxidize/golang/core/paged"
	"github.com/Zapdev-labs/oxidize/golang/core/tokenizer"
)

type pagedGenerateRuntime struct {
	sched     *paged.Scheduler
	mdl       model.Model
	sessions  map[int]*model.Session
	prompts   map[int][]int
	maxTok    map[int]int
	generated map[int]int
}

func newPagedGenerateRuntime(modelPath string, schedCfg paged.SchedulerConfig) (*pagedGenerateRuntime, error) {
	m, err := LoadModelFromPath(modelPath, LoaderConfig{AllowFallback: true})
	if err != nil {
		return nil, err
	}
	return &pagedGenerateRuntime{
		sched:     paged.NewScheduler(schedCfg),
		mdl:       m.Model,
		sessions:  make(map[int]*model.Session),
		prompts:   make(map[int][]int),
		maxTok:    make(map[int]int),
		generated: make(map[int]int),
	}, nil
}

func (r *pagedGenerateRuntime) enqueue(promptTokens []int, maxNew int) (int, error) {
	req, err := r.sched.AddRequest(promptTokens, maxNew)
	if err != nil {
		return 0, err
	}
	id := req.ID
	r.prompts[id] = append([]int(nil), promptTokens...)
	r.maxTok[id] = maxNew
	r.generated[id] = 0
	r.sessions[id] = model.NewSession()
	return id, nil
}

func (r *pagedGenerateRuntime) step(ctx context.Context) (map[int]model.Token, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
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
			batch := make([]model.Token, len(tokens)-1)
			for i, t := range tokens[:len(tokens)-1] {
				batch[i] = model.Token(t)
			}
			if _, err := r.mdl.Forward(batch, sess); err != nil {
				return nil, err
			}
		}
		last := tokens[len(tokens)-1]
		logits, err := r.mdl.Forward([]model.Token{model.Token(last)}, sess)
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

// RunPagedFromGGUF uses the vLLM-style paged scheduler for generation.
func RunPagedFromGGUF(ctx context.Context, cfg RunConfig, stdout io.Writer) error {
	if strings.TrimSpace(cfg.ModelPath) == "" {
		return fmt.Errorf("generate: empty model path")
	}
	if strings.TrimSpace(cfg.Prompt) == "" {
		return nil
	}

	tok, err := loadTokenizer(cfg.ModelPath, cfg.TokenizerModel)
	if err != nil {
		return err
	}
	promptTokens, err := tok.Encode(cfg.Prompt, tokenizer.EncodeOptions{})
	if err != nil {
		return fmt.Errorf("encode: %w", err)
	}
	if len(promptTokens) == 0 {
		promptTokens = []model.Token{1}
	}
	intPrompt := make([]int, len(promptTokens))
	for i, t := range promptTokens {
		intPrompt[i] = int(t)
	}

	runtime, err := newPagedGenerateRuntime(cfg.ModelPath, paged.DefaultSchedulerConfig())
	if err != nil {
		return err
	}
	reqID, err := runtime.enqueue(intPrompt, cfg.MaxNewTokens)
	if err != nil {
		return err
	}

	start := time.Now()
	generated := 0
	for generated < cfg.MaxNewTokens {
		tokens, err := runtime.step(ctx)
		if err != nil {
			return err
		}
		tokOut, ok := tokens[reqID]
		if !ok {
			if len(tokens) == 0 {
				break
			}
			continue
		}
		piece, err := tok.Decode([]model.Token{tokOut})
		if err != nil {
			piece = fmt.Sprintf("<%d>", tokOut)
		}
		if _, err := io.WriteString(stdout, piece); err != nil {
			return err
		}
		generated++
		if tokOut == cfg.StopToken || tokOut == 2 {
			break
		}
	}
	elapsed := time.Since(start).Seconds()
	if elapsed > 0 && generated > 0 {
		_, _ = fmt.Fprintf(stdout, "\ngeneration stats: tokens=%d speed=%.2f tok/s (paged)\n", generated, float64(generated)/elapsed)
	}
	return nil
}
