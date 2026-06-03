// Package util hosts shared cross-cutting helpers: web worker message
// contracts, benchmark suite types, and small utility functions.
package util

import (
	"encoding/json"
	"math"
	"sort"
)

// WebWorkerRequest mirrors WebWorkerRequest.
type WebWorkerRequest struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

// WebWorkerResponse mirrors WebWorkerResponse.
type WebWorkerResponse struct {
	Type    string          `json:"type"`
	OK      bool            `json:"ok"`
	Payload json.RawMessage `json:"payload,omitempty"`
	Error   string          `json:"error,omitempty"`
}

// GenerateRequest mirrors GenerateRequest.
type GenerateRequest struct {
	Prompt        string `json:"prompt"`
	MaxNewTokens  int    `json:"max_new_tokens"`
	Temperature   float32 `json:"temperature"`
	TopP          float32 `json:"top_p"`
	TopK          int    `json:"top_k"`
	Stream        bool   `json:"stream"`
	StopToken     int    `json:"stop_token"`
	Seed          int64  `json:"seed"`
}

// GenerateResponse mirrors GenerateResponse.
type GenerateResponse struct {
	Text         string `json:"text"`
	Tokens       []int  `json:"tokens"`
	FinishReason string `json:"finish_reason"`
}

// TokenEvent mirrors TokenEvent.
type TokenEvent struct {
	Index int    `json:"index"`
	Token int    `json:"token"`
	Text  string `json:"text"`
}

// EncodeWebWorkerRequest serialises a request.
func EncodeWebWorkerRequest(r WebWorkerRequest) ([]byte, error) { return json.Marshal(r) }

// DecodeWebWorkerRequest deserialises a request.
func DecodeWebWorkerRequest(data []byte) (WebWorkerRequest, error) {
	var r WebWorkerRequest
	if err := json.Unmarshal(data, &r); err != nil {
		return r, err
	}
	return r, nil
}

// EncodeWebWorkerResponse serialises a response.
func EncodeWebWorkerResponse(r WebWorkerResponse) ([]byte, error) { return json.Marshal(r) }

// DecodeWebWorkerResponse deserialises a response.
func DecodeWebWorkerResponse(data []byte) (WebWorkerResponse, error) {
	var r WebWorkerResponse
	if err := json.Unmarshal(data, &r); err != nil {
		return r, err
	}
	return r, nil
}

// BenchmarkCase mirrors BenchmarkCase.
type BenchmarkCase struct {
	Name     string
	Prompt   string
	MaxTokens int
	Tags     []string
}

// PerplexityDatasetCase mirrors PerplexityDatasetCase.
type PerplexityDatasetCase struct {
	Name  string
	Texts []string
}

// Result mirrors BenchmarkResult.
type Result struct {
	Name         string
	TokensPerSec float32
	LatencyMs    float32
	MemoryMB     int64
}

// Summary computes mean + p50/p95 over a slice of results.
type Summary struct {
	Count       int
	Mean        float32
	Median      float32
	P95         float32
	Min         float32
	Max         float32
}

// Summarise returns mean + percentiles for a numeric field of the result set.
func Summarise(results []Result) Summary {
	if len(results) == 0 {
		return Summary{}
	}
	tps := make([]float32, len(results))
	for i, r := range results {
		tps[i] = r.TokensPerSec
	}
	sort.Slice(tps, func(i, j int) bool { return tps[i] < tps[j] })
	var sum float32
	for _, v := range tps {
		sum += v
	}
	minVal := tps[0]
	maxVal := tps[len(tps)-1]
	median := tps[len(tps)/2]
	p95idx := int(math.Ceil(0.95*float64(len(tps)))) - 1
	if p95idx < 0 {
		p95idx = 0
	}
	if p95idx >= len(tps) {
		p95idx = len(tps) - 1
	}
	return Summary{Count: len(tps), Mean: sum / float32(len(tps)), Median: median, P95: tps[p95idx], Min: minVal, Max: maxVal}
}

// FormatErrorMessage returns a JSON-friendly error string.
func FormatErrorMessage(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// PipelineStep mirrors PipelineStep.
type PipelineStep struct {
	Name    string
	Apply   func(in any) (any, error)
	Enabled bool
}

// Pipeline mirrors Pipeline.
type Pipeline struct {
	Steps []PipelineStep
}

// NewPipeline constructs an empty pipeline.
func NewPipeline() *Pipeline { return &Pipeline{} }

// Add appends a step.
func (p *Pipeline) Add(step PipelineStep) {
	if step.Enabled {
		p.Steps = append(p.Steps, step)
	}
}

// Run runs the pipeline sequentially.
func (p *Pipeline) Run(in any) (any, error) {
	cur := in
	for _, s := range p.Steps {
		out, err := s.Apply(cur)
		if err != nil {
			return nil, &PipelineError{Step: s.Name, Cause: err}
		}
		cur = out
	}
	return cur, nil
}

// PipelineError mirrors PipelineError.
type PipelineError struct {
	Step  string
	Cause error
}

func (e *PipelineError) Error() string { return "pipeline[" + e.Step + "]: " + e.Cause.Error() }
func (e *PipelineError) Unwrap() error { return e.Cause }
