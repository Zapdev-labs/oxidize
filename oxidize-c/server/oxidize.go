// cgo bindings to the oxidize-c stable C ABI (../src/oxidize.h). One OxModel is
// loaded read-only and shared across requests; generation serializes on a mutex
// because the C runtime's KV cache serves ONE forward pass at a time (documented
// ABI constraint). The library is linked statically from ../liboxidize.a, so the
// resulting binary has no runtime .so dependency (build it with `make lib` first).
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
	// ponytail: one global gen lock == the request queue. The ABI shares a single
	// KV cache across sessions, so generations must not overlap. Upgrade path:
	// per-session KV cache in the C runtime, then drop this lock.
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

// Generate runs one completion for prompt. The C runtime wraps prompt in the
// model's chat template. sink receives decoded UTF-8 fragments; returning true
// stops generation early. Serialized against other Generate calls on this model.
func (m *Model) Generate(prompt string, p SampleParams, sink func([]byte) bool) error {
	m.genMu.Lock()
	defer m.genMu.Unlock()

	s := C.ox_session_new(m.c)
	if s == nil {
		return fmt.Errorf("ox_session_new: out of memory")
	}
	defer C.ox_session_free(s)

	C.ox_session_set_temperature(s, C.float(p.Temperature))
	C.ox_session_set_top_p(s, C.float(p.TopP))
	C.ox_session_set_top_k(s, C.int(p.TopK))
	C.ox_session_set_min_p(s, C.float(p.MinP))
	C.ox_session_set_repeat_penalty(s, C.float(p.RepeatPenalty))
	C.ox_session_set_frequency_penalty(s, C.float(p.FreqPenalty))
	C.ox_session_set_presence_penalty(s, C.float(p.PresPenalty))
	if p.HasSeed {
		C.ox_session_set_seed(s, C.uint64_t(p.Seed))
	}

	maxTok := p.MaxTokens
	if maxTok <= 0 {
		maxTok = 256
	}

	// The handle lets the C callback find `sink` without passing a Go pointer
	// into C (which cgo forbids). It runs on this same goroutine, synchronously
	// inside ox_generate, so writing to an http.ResponseWriter from sink is safe.
	h := cgo.NewHandle(sink)
	defer h.Delete()

	cprompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cprompt))

	errbuf := make([]byte, errBufLen)
	if C.oxGenerate(s, cprompt, C.int(maxTok), C.uintptr_t(h), cerr(errbuf), errBufLen) != 0 {
		return fmt.Errorf("ox_generate: %s", cstr(errbuf))
	}
	return nil
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
