package quantization

/*
#include <stdint.h>
#include <stdlib.h>

extern void* oxidize_model_load(const char* path);
extern void  oxidize_model_free(void* handle);
extern uint32_t oxidize_model_vocab_size(void* handle);
extern void* oxidize_session_new(void);
extern void  oxidize_session_reset(void* session);
extern void  oxidize_session_free(void* session);
extern int   oxidize_model_forward(void* handle, void* session,
                 const uint32_t* tokens, size_t n_tokens,
                 float* logits_out, size_t vocab_size);
extern uint32_t oxidize_sample_argmax(const float* logits, size_t vocab_size);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// RustModel wraps liboxidize_ffi for full model load + forward via Rust.
type RustModel struct {
	handle    unsafe.Pointer
	session   unsafe.Pointer
	vocabSize int
	logits    []float32
}

// LoadRustModel loads a GGUF model via the Rust FFI.
// Returns an error if the shared library is unavailable or the model fails to load.
func LoadRustModel(path string) (*RustModel, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	handle := C.oxidize_model_load(cpath)
	if handle == nil {
		return nil, fmt.Errorf("oxidize_model_load failed for %s", path)
	}
	vs := int(C.oxidize_model_vocab_size(handle))
	if vs <= 0 {
		C.oxidize_model_free(handle)
		return nil, fmt.Errorf("oxidize_model_vocab_size returned %d", vs)
	}
	session := C.oxidize_session_new()
	return &RustModel{
		handle:    handle,
		session:   session,
		vocabSize: vs,
		logits:    make([]float32, vs),
	}, nil
}

// Forward runs one decode step. Returns the logits slice (reused across calls).
func (m *RustModel) Forward(tokens []uint32) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty tokens")
	}
	rc := C.oxidize_model_forward(
		m.handle,
		m.session,
		(*C.uint32_t)(unsafe.Pointer(&tokens[0])),
		C.size_t(len(tokens)),
		(*C.float)(unsafe.Pointer(&m.logits[0])),
		C.size_t(m.vocabSize),
	)
	if rc != 0 {
		return nil, fmt.Errorf("oxidize_model_forward returned %d", rc)
	}
	return m.logits, nil
}

// SampleArgmax returns the index of the highest logit.
func (m *RustModel) SampleArgmax() uint32 {
	return uint32(C.oxidize_sample_argmax(
		(*C.float)(unsafe.Pointer(&m.logits[0])),
		C.size_t(m.vocabSize),
	))
}

// ResetSession clears the KV cache so the model can start a new sequence.
func (m *RustModel) ResetSession() {
	C.oxidize_session_reset(m.session)
}

// VocabSize returns the vocabulary size reported by the loaded model.
func (m *RustModel) VocabSize() int { return m.vocabSize }

// Close frees the model and session handles.
func (m *RustModel) Close() {
	if m.session != nil {
		C.oxidize_session_free(m.session)
		m.session = nil
	}
	if m.handle != nil {
		C.oxidize_model_free(m.handle)
		m.handle = nil
	}
}
