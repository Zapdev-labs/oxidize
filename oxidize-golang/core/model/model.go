// Package model mirrors oxidize_core::model. It contains the Model trait
// (Session, ModelError, Token, Logits, Boxed), all model implementations
// (LlamaModel, InferenceModel, LayerWiseModel, DFlashDraftModel), the
// generation and sampling pipelines, the speculative decoder, advanced
// samplers (XTC, DRY, dynamic temperature, grammar, tool calling), and
// helpers (lora, loader, offload, prefix cache).
package model

import (
	"errors"
	"fmt"
	"sync"
)

// Token is a vocabulary id.
type Token = uint32

// Logits is the raw output of a forward pass over the vocabulary.
type Logits []float32

// Session tracks the tokens that have been consumed by an autoregressive
// generation loop. Mirrors `Session` in the Rust crate.
type Session struct {
	mu             sync.Mutex
	consumedTokens int
}

// NewSession returns a fresh session.
func NewSession() *Session { return &Session{} }

// ConsumedTokens returns the number of tokens consumed by the session.
func (s *Session) ConsumedTokens() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.consumedTokens
}

// RecordTokens advances the session by `n` tokens.
func (s *Session) RecordTokens(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.consumedTokens += n
}

// RewindTo resets the session to a previous consumed token count.
func (s *Session) RewindTo(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if n < 0 {
		n = 0
	}
	if n < s.consumedTokens {
		s.consumedTokens = n
	}
}

// Error mirrors ModelError.
type Error struct{ Message string }

func (e *Error) Error() string { return "model: " + e.Message }

// NewErrorf constructs a formatted model error.
func NewErrorf(format string, args ...any) *Error {
	return &Error{Message: fmt.Sprintf(format, args...)}
}

// EmptyInputError mirrors the EmptyInput variant.
var EmptyInputError = &Error{Message: "empty input"}

// ContextExceededError mirrors ContextExceeded { context_size, requested_total_tokens }.
type ContextExceededError struct {
	ContextSize            int
	RequestedTotalTokens   int
}

func (e *ContextExceededError) Error() string {
	return fmt.Sprintf("model: context exceeded: %d > %d", e.RequestedTotalTokens, e.ContextSize)
}

// AsError converts a model error to *Error if possible.
func AsError(err error) (*Error, bool) {
	if err == nil {
		return nil, true
	}
	var me *Error
	if errors.As(err, &me) {
		return me, true
	}
	return nil, false
}

// Model is the trait implemented by every model backend (Inference, LayerWise,
// Llama, DFlash, MLX).
type Model interface {
	Forward(tokens []Token, session *Session) (Logits, error)
	VocabSize() int
	ContextSize() int
	LayerCount() int
}

// Boxed is a type-erased model handle.
type Boxed struct {
	M Model
}

// Forward delegates to the underlying model.
func (b *Boxed) Forward(tokens []Token, session *Session) (Logits, error) {
	return b.M.Forward(tokens, session)
}

// VocabSize returns the wrapped model's vocab size.
func (b *Boxed) VocabSize() int { return b.M.VocabSize() }

// ContextSize returns the wrapped model's context size.
func (b *Boxed) ContextSize() int { return b.M.ContextSize() }

// LayerCount returns the wrapped model's layer count.
func (b *Boxed) LayerCount() int { return b.M.LayerCount() }

// ForwardMany is a convenience that runs forward on a batch.
func ForwardMany(m Model, batch [][]Token, session *Session) ([]Logits, error) {
	out := make([]Logits, len(batch))
	for i, tokens := range batch {
		logits, err := m.Forward(tokens, session)
		if err != nil {
			return nil, err
		}
		out[i] = logits
	}
	return out, nil
}

// Rewindable is an optional capability for models that support rewinding the
// KV cache to a previous token count.
type Rewindable interface {
	RewindTo(session *Session, n int)
}

// Architecture mirrors ModelArchitecture.
type Architecture string

const (
	ArchLlamaModel    Architecture = "llama"
	ArchMistralModel  Architecture = "mistral"
	ArchMixtralModel  Architecture = "mixtral"
	ArchDeepSeekModel Architecture = "deepseek"
	ArchQwenModel     Architecture = "qwen"
	ArchGemmaModel    Architecture = "gemma"
	ArchPhiModel      Architecture = "phi"
	ArchFalconModel   Architecture = "falcon"
	ArchGpt2Model     Architecture = "gpt2"
	ArchGptJModel     Architecture = "gptj"
	ArchGptNeoXModel  Architecture = "gpt_neox"
	ArchMiniMaxModel  Architecture = "minimax"
)

// DefaultArchitecture is the architecture used when metadata is missing.
const DefaultArchitecture = ArchLlamaModel

// UsesAlibi returns true if the architecture uses ALiBi positional encoding.
func (a Architecture) UsesAlibi() bool {
	return a == ArchGptJModel || a == ArchGptNeoXModel
}

// UsesSlidingWindow returns true if the architecture uses sliding window
// attention (Mistral).
func (a Architecture) UsesSlidingWindow() bool {
	return a == ArchMistralModel || a == ArchMixtralModel
}

// UsesMoE returns true if the architecture is a mixture-of-experts.
func (a Architecture) UsesMoE() bool {
	return a == ArchMixtralModel || a == ArchDeepSeekModel
}

// UsesParallelAttnFFN returns true if the architecture parallelizes the
// attention and FFN computations.
func (a Architecture) UsesParallelAttnFFN() bool { return a == ArchFalconModel }

// UsesMLA returns true if the architecture uses multi-latent attention.
func (a Architecture) UsesMLA() bool { return a == ArchDeepSeekModel }

// String returns the canonical architecture name.
func (a Architecture) String() string { return string(a) }
