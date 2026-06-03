package model

// LlamaArchitecture mirrors the legacy llama.rs Architecture enum.
type LlamaArchitecture string

const (
	LlamaArchLlama2   LlamaArchitecture = "llama2"
	LlamaArchLlama3   LlamaArchitecture = "llama3"
	LlamaArchMistral  LlamaArchitecture = "mistral"
	LlamaArchMixtral  LlamaArchitecture = "mixtral"
	LlamaArchQwen     LlamaArchitecture = "qwen"
	LlamaArchGemma    LlamaArchitecture = "gemma"
	LlamaArchPhi      LlamaArchitecture = "phi"
	LlamaArchFalcon   LlamaArchitecture = "falcon"
	LlamaArchGpt2     LlamaArchitecture = "gpt2"
	LlamaArchGptJ     LlamaArchitecture = "gptj"
	LlamaArchGptNeoX  LlamaArchitecture = "gpt_neox"
)

// LlamaConfig is the legacy config struct from llama.rs.
type LlamaConfig struct {
	Architecture LlamaArchitecture
	VocabSize    int
	ContextSize  int
	LayerCount   int
}

// LlamaModel is a stub model implementation that holds a config and reports
// the configured sizes; this mirrors the legacy llama.rs that pre-dates
// InferenceModel. In production code, prefer InferenceModel.
type LlamaModel struct {
	Config LlamaConfig
}

// NewLlama2 constructs a Llama2-flavored model.
func NewLlama2() *LlamaModel { return &LlamaModel{Config: LlamaConfig{Architecture: LlamaArchLlama2, VocabSize: 32000, ContextSize: 2048, LayerCount: 32}} }

// NewLlama3 constructs a Llama3-flavored model.
func NewLlama3() *LlamaModel { return &LlamaModel{Config: LlamaConfig{Architecture: LlamaArchLlama3, VocabSize: 128256, ContextSize: 8192, LayerCount: 32}} }

// NewMistral constructs a Mistral-flavored model.
func NewMistral() *LlamaModel { return &LlamaModel{Config: LlamaConfig{Architecture: LlamaArchMistral, VocabSize: 32000, ContextSize: 32768, LayerCount: 32}} }

// NewMixtral constructs a Mixtral-flavored model.
func NewMixtral() *LlamaModel { return &LlamaModel{Config: LlamaConfig{Architecture: LlamaArchMixtral, VocabSize: 32000, ContextSize: 32768, LayerCount: 32}} }

// NewQwen constructs a Qwen-flavored model.
func NewQwen() *LlamaModel { return &LlamaModel{Config: LlamaConfig{Architecture: LlamaArchQwen, VocabSize: 151936, ContextSize: 32768, LayerCount: 32}} }

// NewGemma constructs a Gemma-flavored model.
func NewGemma() *LlamaModel { return &LlamaModel{Config: LlamaConfig{Architecture: LlamaArchGemma, VocabSize: 256000, ContextSize: 8192, LayerCount: 18}} }

// NewPhi constructs a Phi-flavored model.
func NewPhi() *LlamaModel { return &LlamaModel{Config: LlamaConfig{Architecture: LlamaArchPhi, VocabSize: 51200, ContextSize: 2048, LayerCount: 32}} }

// NewFalcon constructs a Falcon-flavored model.
func NewFalcon() *LlamaModel { return &LlamaModel{Config: LlamaConfig{Architecture: LlamaArchFalcon, VocabSize: 65024, ContextSize: 2048, LayerCount: 32}} }

// NewGpt2 constructs a GPT-2-flavored model.
func NewGpt2() *LlamaModel { return &LlamaModel{Config: LlamaConfig{Architecture: LlamaArchGpt2, VocabSize: 50257, ContextSize: 1024, LayerCount: 12}} }

// NewGptJ constructs a GPT-J-flavored model.
func NewGptJ() *LlamaModel { return &LlamaModel{Config: LlamaConfig{Architecture: LlamaArchGptJ, VocabSize: 50400, ContextSize: 2048, LayerCount: 28}} }

// NewGptNeoX constructs a GPT-NeoX-flavored model.
func NewGptNeoX() *LlamaModel { return &LlamaModel{Config: LlamaConfig{Architecture: LlamaArchGptNeoX, VocabSize: 50432, ContextSize: 2048, LayerCount: 32}} }

// Forward returns a zero-vector of VocabSize logits (placeholder behavior).
func (m *LlamaModel) Forward(_ []Token, _ *Session) (Logits, error) {
	return make(Logits, m.Config.VocabSize), nil
}

// VocabSize returns the configured vocabulary size.
func (m *LlamaModel) VocabSize() int { return m.Config.VocabSize }

// ContextSize returns the configured context size.
func (m *LlamaModel) ContextSize() int { return m.Config.ContextSize }

// LayerCount returns the configured layer count.
func (m *LlamaModel) LayerCount() int { return m.Config.LayerCount }
