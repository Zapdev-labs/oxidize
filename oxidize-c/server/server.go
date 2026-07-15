// OpenAI-compatible HTTP layer over a Generator (implemented by *Model). Kept
// stdlib-only (net/http + encoding/json). Endpoints: GET /healthz, GET
// /v1/models, POST /v1/chat/completions, POST /v1/completions — each streaming
// (SSE) or not. Concurrency: Generator.Generate serializes on the shared model,
// so overlapping requests queue rather than corrupt the KV cache.
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// Generator is the seam the HTTP layer drives; *Model is the real one, and tests
// substitute a stub so the JSON/SSE plumbing is exercised without a real model.
type Generator interface {
	Generate(prompt string, p SampleParams, sink func([]byte) bool) error
	ID() string
	Meta() Metadata
}

// Server wires the endpoints to a Generator.
type Server struct{ gen Generator }

func NewServer(gen Generator) *Server { return &Server{gen: gen} }

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.healthz)
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("POST /v1/chat/completions", s.chatCompletions)
	mux.HandleFunc("POST /v1/completions", s.completions)
	return mux
}

// ---- request/response types ------------------------------------------------

// flexString decodes an OpenAI field that is either a string or []string (used
// for `stop` and the completions `prompt`).
type flexString []string

func (f *flexString) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	if b[0] == '[' {
		var arr []string
		if err := json.Unmarshal(b, &arr); err != nil {
			return err
		}
		*f = arr
		return nil
	}
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*f = []string{s}
	return nil
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type sampleFields struct {
	Temperature      *float64 `json:"temperature"`
	TopP             *float64 `json:"top_p"`
	TopK             *int     `json:"top_k"`
	MinP             *float64 `json:"min_p"`
	FrequencyPenalty *float64 `json:"frequency_penalty"`
	PresencePenalty  *float64 `json:"presence_penalty"`
	RepeatPenalty    *float64 `json:"repeat_penalty"` // non-standard extension
	MaxTokens        *int     `json:"max_tokens"`
	Seed             *uint64  `json:"seed"`
	Stop             flexString `json:"stop"`
	Stream           bool     `json:"stream"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	sampleFields
}

type completionRequest struct {
	Model  string     `json:"model"`
	Prompt flexString `json:"prompt"`
	sampleFields
}

func (f sampleFields) params() SampleParams {
	p := SampleParams{Temperature: 1.0, TopP: 1.0, RepeatPenalty: 1.0, MaxTokens: 256}
	if f.Temperature != nil {
		p.Temperature = float32(*f.Temperature)
	}
	if f.TopP != nil {
		p.TopP = float32(*f.TopP)
	}
	if f.TopK != nil {
		p.TopK = *f.TopK
	}
	if f.MinP != nil {
		p.MinP = float32(*f.MinP)
	}
	if f.FrequencyPenalty != nil {
		p.FreqPenalty = float32(*f.FrequencyPenalty)
	}
	if f.PresencePenalty != nil {
		p.PresPenalty = float32(*f.PresencePenalty)
	}
	if f.RepeatPenalty != nil {
		p.RepeatPenalty = float32(*f.RepeatPenalty)
	}
	if f.MaxTokens != nil && *f.MaxTokens > 0 {
		p.MaxTokens = *f.MaxTokens
	}
	if f.Seed != nil {
		p.Seed, p.HasSeed = *f.Seed, true
	}
	return p
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ---- endpoints -------------------------------------------------------------

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) models(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id": s.gen.ID(), "object": "model", "created": startTime, "owned_by": "oxidize-c",
		}},
	})
}

func (s *Server) chatCompletions(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := decode(r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeErr(w, http.StatusBadRequest, "messages: at least one message required")
		return
	}
	prompt := flattenMessages(req.Messages)
	p := req.params()
	id := newID("chatcmpl-")
	model := s.gen.ID()

	if req.Stream {
		s.stream(w, r, id, model, "chat.completion.chunk", p, prompt, req.Stop, true)
		return
	}
	var sb strings.Builder
	finish, nTok, err := runGen(r.Context(), s.gen, prompt, p, req.Stop, func(b []byte) { sb.Write(b) })
	if err != nil && nTok == 0 {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "chat.completion", "created": time.Now().Unix(), "model": model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       chatMessage{Role: "assistant", Content: sb.String()},
			"finish_reason": finish,
		}},
		"usage": usage{CompletionTokens: nTok, TotalTokens: nTok},
	})
}

func (s *Server) completions(w http.ResponseWriter, r *http.Request) {
	var req completionRequest
	if err := decode(r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Prompt) == 0 {
		writeErr(w, http.StatusBadRequest, "prompt: required")
		return
	}
	prompt := strings.Join(req.Prompt, "\n") // ponytail: array prompts -> one joined completion
	p := req.params()
	id := newID("cmpl-")
	model := s.gen.ID()

	if req.Stream {
		s.stream(w, r, id, model, "text_completion", p, prompt, req.Stop, false)
		return
	}
	var sb strings.Builder
	finish, nTok, err := runGen(r.Context(), s.gen, prompt, p, req.Stop, func(b []byte) { sb.Write(b) })
	if err != nil && nTok == 0 {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "text_completion", "created": time.Now().Unix(), "model": model,
		"choices": []map[string]any{{"index": 0, "text": sb.String(), "finish_reason": finish}},
		"usage":   usage{CompletionTokens: nTok, TotalTokens: nTok},
	})
}

// stream serves an SSE response for either chat (chatMode) or completions.
func (s *Server) stream(w http.ResponseWriter, r *http.Request, id, model, object string,
	p SampleParams, prompt string, stop flexString, chatMode bool) {
	fl, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "streaming unsupported by this server")
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	created := time.Now().Unix()

	chunk := func(delta map[string]any, finish any) {
		choice := map[string]any{"index": 0, "finish_reason": finish}
		if chatMode {
			choice["delta"] = delta
		} else if c, ok := delta["content"]; ok {
			choice["text"] = c
		} else {
			choice["text"] = "" // text_completion requires a string; the terminal chunk's delta is empty (never JSON null)
		}
		writeSSE(w, fl, map[string]any{
			"id": id, "object": object, "created": created, "model": model,
			"choices": []map[string]any{choice},
		})
	}

	if chatMode {
		chunk(map[string]any{"role": "assistant"}, nil) // first chunk announces the role
	}
	finish, _, _ := runGen(r.Context(), s.gen, prompt, p, stop, func(b []byte) {
		chunk(map[string]any{"content": string(b)}, nil)
	})
	chunk(map[string]any{}, finish)
	io.WriteString(w, "data: [DONE]\n\n")
	fl.Flush()
}

// ---- generation driver -----------------------------------------------------

// runGen drives one generation, funnelling model output through an outputFilter
// (stop-sequence + UTF-8 boundary handling) into onEmit. It returns the OpenAI
// finish_reason and the number of tokens produced.
func runGen(ctx context.Context, gen Generator, prompt string, p SampleParams,
	stops flexString, onEmit func([]byte)) (finish string, nTok int, err error) {
	f := &outputFilter{stops: stops, maxLen: maxStopLen(stops)}
	stopHit := false
	err = gen.Generate(prompt, p, func(piece []byte) bool {
		if ctx != nil && ctx.Err() != nil {
			return true // client hung up
		}
		nTok++
		emit, stop := f.push(piece)
		if len(emit) > 0 {
			onEmit(emit)
		}
		if stop {
			stopHit = true
		}
		return stop
	})
	if tail := f.flush(); len(tail) > 0 {
		onEmit(tail)
	}
	switch {
	case stopHit:
		finish = "stop"
	case p.MaxTokens > 0 && nTok >= p.MaxTokens:
		finish = "length"
	default:
		finish = "stop" // EOS / EOT
	}
	if err != nil {
		log.Printf("generate: %v", err)
	}
	return finish, nTok, err
}

// flattenMessages renders OpenAI messages[] into a single prompt string. The C
// runtime then wraps the whole thing in the model's user-turn template.
// ponytail: single-turn faithful; multi-turn history is best-effort plain text,
// because the ABI templates one turn and cannot inject prior assistant text into
// the KV cache. Add a raw multi-turn path to the C ABI if that ceiling matters.
func flattenMessages(msgs []chatMessage) string {
	var b strings.Builder
	for _, m := range msgs {
		c := strings.TrimSpace(m.Content)
		if c == "" {
			continue
		}
		switch m.Role {
		case "system":
			b.WriteString(c)
			b.WriteString("\n\n")
		case "assistant":
			b.WriteString(c)
			b.WriteString("\n")
		default: // user, tool, ...
			b.WriteString(c)
			b.WriteString("\n")
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// ---- output filter: stop sequences + UTF-8 stream safety --------------------

type outputFilter struct {
	stops  []string
	maxLen int
	full   []byte
	sent   int
	done   bool
}

// push adds a raw model fragment and returns the bytes now safe to emit plus
// whether generation should stop. It holds back (a) any incomplete trailing
// UTF-8 sequence and (b) up to maxLen-1 bytes that might still complete a stop
// string split across fragments — so a stop is never partly emitted.
func (o *outputFilter) push(p []byte) (emit []byte, stop bool) {
	if o.done {
		return nil, true
	}
	o.full = append(o.full, p...)
	cut := -1
	for _, s := range o.stops {
		if s == "" {
			continue
		}
		if i := bytes.Index(o.full[o.sent:], []byte(s)); i >= 0 {
			if pos := o.sent + i; cut < 0 || pos < cut {
				cut = pos
			}
		}
	}
	var end int
	if cut >= 0 {
		end, o.done = cut, true
	} else {
		end = len(o.full)
		if o.maxLen > 1 {
			end -= o.maxLen - 1
		}
		if end < o.sent {
			end = o.sent
		}
		if k := trailingIncomplete(o.full[o.sent:end]); k > 0 {
			end -= k
		}
		if end < o.sent {
			end = o.sent
		}
	}
	emit = o.full[o.sent:end]
	o.sent = end
	return emit, o.done
}

// flush returns any remaining bytes at end of generation. Nothing follows a stop
// (we already cut there), so a stopped filter flushes empty.
func (o *outputFilter) flush() []byte {
	if o.done {
		return nil
	}
	e := o.full[o.sent:]
	o.sent = len(o.full)
	return e
}

func maxStopLen(stops []string) int {
	m := 0
	for _, s := range stops {
		if len(s) > m {
			m = len(s)
		}
	}
	return m
}

// trailingIncomplete returns how many trailing bytes of b form an incomplete
// (but potentially completable) UTF-8 sequence — bytes to hold back until the
// rest of the rune arrives. A standalone/complete boundary returns 0.
func trailingIncomplete(b []byte) int {
	n := len(b)
	for i := 1; i <= 3 && i <= n; i++ {
		c := b[n-i]
		if c < 0x80 {
			return 0 // ASCII: boundary right after it
		}
		if c >= 0xC0 { // lead byte of an i..need-byte sequence
			need := 2
			if c >= 0xF0 {
				need = 4
			} else if c >= 0xE0 {
				need = 3
			}
			if i < need {
				return i // not all continuation bytes have arrived
			}
			return 0
		}
		// else 0x80..0xBF continuation byte: keep scanning back for the lead
	}
	return 0
}

// ---- small helpers ---------------------------------------------------------

var startTime = time.Now().Unix()

func decode(r *http.Request, v any) error {
	dec := json.NewDecoder(io.LimitReader(r.Body, 8<<20))
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("invalid JSON body: %w", err)
	}
	return nil
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{
		"error": map[string]any{"message": msg, "type": "invalid_request_error"},
	})
}

func writeSSE(w http.ResponseWriter, fl http.Flusher, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
		return
	}
	fl.Flush()
}

func newID(prefix string) string { return prefix + rand.Text() }
