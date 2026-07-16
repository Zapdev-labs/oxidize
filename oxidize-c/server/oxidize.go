// cgo bindings to the oxidize-c stable C ABI (../src/oxidize.h). One OxModel is
// loaded read-only and shared across requests. Each OxSession owns its KV cache;
// Generate still serializes on genMu because forward scratch (x/q/k/...) is
// model-wide. The library is linked statically from ../liboxidize.a (build with
// `make lib` first).
package main

/*
#cgo CFLAGS: -I${SRCDIR}/../src
#cgo LDFLAGS: ${SRCDIR}/../liboxidize.a -lm -lpthread
#include <stdlib.h>
#include <stdint.h>
#include "oxidize.h"
#include "bridge.h"
*/
import "C"

import (
	"fmt"
	"runtime/cgo"
	"sync"
	"unsafe"
)

const errBufLen = 256

// Metadata is the --inspect-style summary of a loaded model (ox_metadata).
type Metadata struct {
	Arch     string
	ISA      string
	Vocab    int
	Ctx      int
	NTensors int
	NKV      int
}

// SampleParams map the OpenAI knobs onto ox_session_set_*.
type SampleParams struct {
	Temperature   float32
	TopP          float32
	TopK          int
	MinP          float32
	RepeatPenalty float32
	FreqPenalty   float32
	PresPenalty   float32
	Seed          uint64
	HasSeed       bool
	MaxTokens     int
}

// Model is one loaded GGUF (weights + tokenizer). Safe to share; Generate
// serializes internally.
type Model struct {
	c *C.OxModel
	// ponytail: genMu serializes scratch/compute on the shared model. Per-session
	// KV is owned in C now, so overlapping multi-turn conversations no longer
	// corrupt caches; this lock remains so two Generate calls do not race on
	// the model's forward scratch (x/q/k/...).
	genMu sync.Mutex
	id    string
	meta  Metadata
}

// OpenModel loads a GGUF via the C ABI. ctx==0 and threads<=0 mean model/CPU
// defaults. id names the model in /v1/models (defaults to the architecture).
func OpenModel(path, id string, ctx, threads int) (*Model, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	var opts C.OxModelOptions
	opts.struct_size = C.size_t(unsafe.Sizeof(opts))
	opts.ctx = C.size_t(ctx)
	opts.threads = C.int(threads)

	errbuf := make([]byte, errBufLen)
	var m *C.OxModel
	if C.ox_model_open(&m, cpath, &opts, cerr(errbuf), errBufLen) != 0 {
		return nil, fmt.Errorf("ox_model_open: %s", cstr(errbuf))
	}

	var md C.OxMetadata
	md.struct_size = C.size_t(unsafe.Sizeof(md))
	if C.ox_metadata(m, &md) != 0 {
		C.ox_model_close(m)
		return nil, fmt.Errorf("ox_metadata: struct_size/ABI mismatch")
	}
	meta := Metadata{
		Arch:     C.GoString(md.arch),
		ISA:      C.GoString(md.isa),
		Vocab:    int(md.vocab),
		Ctx:      int(md.ctx),
		NTensors: int(md.n_tensors),
		NKV:      int(md.n_kv),
	}
	if id == "" {
		id = meta.Arch
	}
	if id == "" {
		id = "oxidize-c"
	}
	return &Model{c: m, id: id, meta: meta}, nil
}

// Close releases the model. Do not call Generate afterwards.
func (m *Model) Close() {
	if m != nil && m.c != nil {
		C.ox_model_close(m.c)
		m.c = nil
	}
}

func (m *Model) ID() string     { return m.id }
func (m *Model) Meta() Metadata { return m.meta }

// Session is one multi-turn conversation with its own KV cache. Create with
// Model.NewSession; free with Close. Generate on a Session is serialized with
// other Generates on the same Model (shared scratch), but KV is not shared.
type Session struct {
	m *Model
	c *C.OxSession
}

// NewSession allocates a conversation against m. Caller must Close it.
func (m *Model) NewSession() (*Session, error) {
	if m == nil || m.c == nil {
		return nil, fmt.Errorf("ox_session_new: model is closed")
	}
	s := C.ox_session_new(m.c)
	if s == nil {
		return nil, fmt.Errorf("ox_session_new: out of memory")
	}
	return &Session{m: m, c: s}, nil
}

// Close frees the session and its KV. Safe on nil / double-close.
func (s *Session) Close() {
	if s != nil && s.c != nil {
		C.ox_session_free(s.c)
		s.c = nil
	}
}

// Reset clears KV position and the repeat-penalty window (ox_session_reset).
func (s *Session) Reset() {
	if s != nil && s.c != nil {
		C.ox_session_reset(s.c)
	}
}

func (s *Session) applySample(p SampleParams) {
	C.ox_session_set_temperature(s.c, C.float(p.Temperature))
	C.ox_session_set_top_p(s.c, C.float(p.TopP))
	C.ox_session_set_top_k(s.c, C.int(p.TopK))
	C.ox_session_set_min_p(s.c, C.float(p.MinP))
	C.ox_session_set_repeat_penalty(s.c, C.float(p.RepeatPenalty))
	C.ox_session_set_frequency_penalty(s.c, C.float(p.FreqPenalty))
	C.ox_session_set_presence_penalty(s.c, C.float(p.PresPenalty))
	if p.HasSeed {
		C.ox_session_set_seed(s.c, C.uint64_t(p.Seed))
	}
}

// Generate continues this conversation. Serialized against other Generates on
// the parent Model; this session's KV persists across calls until Reset/Close.
func (s *Session) Generate(prompt string, p SampleParams, sink func([]byte) bool) error {
	if s == nil || s.c == nil || s.m == nil {
		return fmt.Errorf("ox_generate: session is closed")
	}
	s.m.genMu.Lock()
	defer s.m.genMu.Unlock()

	s.applySample(p)
	maxTok := p.MaxTokens
	if maxTok <= 0 {
		maxTok = 256
	}
	h := cgo.NewHandle(sink)
	defer h.Delete()
	cprompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cprompt))
	errbuf := make([]byte, errBufLen)
	if C.oxGenerate(s.c, cprompt, C.int(maxTok), C.uintptr_t(h), cerr(errbuf), errBufLen) != 0 {
		return fmt.Errorf("ox_generate: %s", cstr(errbuf))
	}
	return nil
}

// Generate runs one completion for prompt on a fresh session (stateless).
// Prefer NewSession for multi-turn. sink returning true stops early.
func (m *Model) Generate(prompt string, p SampleParams, sink func([]byte) bool) error {
	s, err := m.NewSession()
	if err != nil {
		return err
	}
	defer s.Close()
	return s.Generate(prompt, p, sink)
}

//export oxGoSink
func oxGoSink(piece *C.char, length C.size_t, user C.uintptr_t) C.int {
	sink := cgo.Handle(user).Value().(func([]byte) bool)
	if sink(C.GoBytes(unsafe.Pointer(piece), C.int(length))) {
		return 1 // caller-requested stop (clean)
	}
	return 0
}

// Version returns the library version, e.g. "oxidize-c 0.1.0".
func Version() string { return C.GoString(C.ox_version()) }

// ISA returns the active kernel ISA, e.g. "avx2".
func ISA() string { return C.GoString(C.ox_isa()) }

func cerr(b []byte) *C.char { return (*C.char)(unsafe.Pointer(&b[0])) }

func cstr(b []byte) string {
	if i := indexByte(b, 0); i >= 0 {
		return string(b[:i])
	}
	return string(b)
}

func indexByte(b []byte, c byte) int {
	for i := range b {
		if b[i] == c {
			return i
		}
	}
	return -1
}
