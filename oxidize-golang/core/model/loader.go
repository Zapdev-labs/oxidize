package model

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
	"github.com/Zapdev-labs/oxidize/golang/hf"
)

// ModelSource represents where a model can be loaded from.
type ModelSource interface {
	modelSource()
}

// FileSource points to a local file path.
type FileSource struct{ Path string }

func (FileSource) modelSource() {}

// MemorySource is for in-memory model blobs.
type MemorySource struct {
	Name string
	Data []byte
}

func (MemorySource) modelSource() {}

// HFSource references a Hugging Face repository.
type HFSource struct{ Repo, Revision string }

func (HFSource) modelSource() {}

// QuantType mirrors the quantized type string for loader summaries.
type QuantType string

// LoaderConfig mirrors LoaderConfig.
type LoaderConfig struct {
	Source          ModelSource
	PreferredDType  string
	MaxMemoryBudget int64
	AllowFallback   bool
	HFFilename      string
	HFCacheDir      string
	Backend         string
	Threads         int
	ContextSize     int
	NGPULayers      int
	GPUs            int
	Parallelism     string
}

// NewLoaderConfig returns sensible defaults.
func NewLoaderConfig() LoaderConfig { return LoaderConfig{MaxMemoryBudget: 8 << 30, AllowFallback: true} }

// ModelLoader mirrors ModelLoader.
type ModelLoader struct {
	Source  ModelSource
	Config  LoaderConfig
	Metrics LoadMetrics
}

// LoadMetrics tracks loading statistics.
type LoadMetrics struct {
	BytesRead     int64
	ElapsedMillis int64
	LayerCount    int
}

// NewModelLoader constructs a loader.
func NewModelLoader(source ModelSource, config LoaderConfig) *ModelLoader {
	return &ModelLoader{Source: source, Config: config}
}

// Load dispatches to the appropriate loader.
func (l *ModelLoader) Load() (Model, error) {
	switch src := l.Source.(type) {
	case FileSource:
		return LoadGGUFModelFromPath(src.Path, l.Config)
	case MemorySource:
		return LoadGGUFModelFromBytes(src.Data, l.Config)
	case HFSource:
		return LoadGGUFModelFromHF(src.Repo, src.Revision, l.Config)
	default:
		return nil, fmt.Errorf("loader: unsupported source type %T", l.Source)
	}
}

// LoadGGUFModelFromPath loads a GGUF file from disk into an InferenceModel.
func LoadGGUFModelFromPath(path string, config LoaderConfig) (Model, error) {
	if path == "" {
		return nil, errors.New("loader: empty path")
	}
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		mapped, err := ggufcore.LoadMapped(path)
		if err != nil {
			return nil, fmt.Errorf("loader: %w", err)
		}
		inference, err := LoadInferenceFromGGUF(mapped)
		if err != nil {
			return nil, err
		}
		ApplyLoaderConfig(inference, config)
		return inference, nil
	}
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("loader: %w", err)
	}
	return nil, fmt.Errorf("loader: unsupported model path %q", path)
}

// LoadGGUFModelFromBytes loads a GGUF blob from memory.
func LoadGGUFModelFromBytes(data []byte, config LoaderConfig) (Model, error) {
	if len(data) == 0 {
		return nil, errors.New("loader: empty bytes")
	}
	_ = config
	tmp, err := os.CreateTemp("", "oxidize-*.gguf")
	if err != nil {
		return nil, err
	}
	path := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(path)
	}()
	if _, err := tmp.Write(data); err != nil {
		return nil, err
	}
	return LoadGGUFModelFromPath(path, config)
}

// LoadGGUFModelFromHF downloads a GGUF from Hugging Face and loads it.
func LoadGGUFModelFromHF(repo, revision string, config LoaderConfig) (Model, error) {
	if repo == "" {
		return nil, errors.New("loader: empty HF repo")
	}
	path, err := hf.ResolveGGUF(hf.ResolveOptions{
		Repo:     repo,
		Revision: revision,
		Filename: config.HFFilename,
		CacheDir: config.HFCacheDir,
	})
	if err != nil {
		return nil, err
	}
	return LoadGGUFModelFromPath(path, config)
}

// ModelType describes the on-disk format of a model.
type ModelType string

// Recognised model formats.
const (
	ModelTypeGGUF    ModelType = "gguf"
	ModelTypeSafeTensors ModelType = "safetensors"
	ModelTypeONNX    ModelType = "onnx"
	ModelTypePyTorch ModelType = "pytorch"
)

// DetectModelType infers the model type from a path or URL.
func DetectModelType(path string) ModelType {
	switch {
	case endsWith(path, ".gguf"):
		return ModelTypeGGUF
	case endsWith(path, ".safetensors"):
		return ModelTypeSafeTensors
	case endsWith(path, ".onnx"):
		return ModelTypeONNX
	case endsWith(path, ".pt"), endsWith(path, ".pth"):
		return ModelTypePyTorch
	}
	return ModelTypeGGUF
}

func endsWith(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}

func guessArchFromPath(path string) Architecture {
	lower := lowerString(path)
	switch {
	case contains(lower, "llama"):
		return ArchLlamaModel
	case contains(lower, "mistral"):
		return ArchMistralModel
	case contains(lower, "mixtral"):
		return ArchMixtralModel
	case contains(lower, "qwen"):
		return ArchQwenModel
	case contains(lower, "gemma"):
		return ArchGemmaModel
	case contains(lower, "phi"):
		return ArchPhiModel
	case contains(lower, "falcon"):
		return ArchFalconModel
	case contains(lower, "deepseek"):
		return ArchDeepSeekModel
	case contains(lower, "gpt2"):
		return ArchGpt2Model
	case contains(lower, "gptj"):
		return ArchGptJModel
	case contains(lower, "gptneox"):
		return ArchGptNeoXModel
	}
	return ArchLlamaModel
}

func lowerString(s string) string {
	out := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		out[i] = c
	}
	return string(out)
}

func contains(s, sub string) bool {
	if len(sub) == 0 {
		return true
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// GgufModelLoader mirrors GgufModelLoader.
type GgufModelLoader struct {
	Path   string
	Config LoaderConfig
}

// NewGgufModelLoader constructs a loader for GGUF files.
func NewGgufModelLoader(path string, config LoaderConfig) *GgufModelLoader {
	return &GgufModelLoader{Path: path, Config: config}
}

// Load reads the GGUF file and returns a model.
func (g *GgufModelLoader) Load() (Model, error) {
	return LoadGGUFModelFromPath(g.Path, g.Config)
}

// BaselineGgufModel mirrors BaselineGgufModel.
type BaselineGgufModel struct {
	Path    string
	Arch    Architecture
	Layers  int
	Hidden  int
	Heads   int
	KVHeads int
	Vocab   int
}

// NewBaselineGgufModel constructs a baseline that ships random weights. Used
// in tests and local smoke checks.
func NewBaselineGgufModel(arch Architecture, layers, hidden, heads, kvHeads, vocab int) *BaselineGgufModel {
	return &BaselineGgufModel{Path: "<baseline>", Arch: arch, Layers: layers, Hidden: hidden, Heads: heads, KVHeads: kvHeads, Vocab: vocab}
}

// Forward implements Model by returning synthetic logits.
func (b *BaselineGgufModel) Forward(tokens []Token, session *Session) (Logits, error) {
	if b.Vocab <= 0 {
		return nil, errors.New("baseline: empty vocab")
	}
	logits := make(Logits, b.Vocab)
	for i := range logits {
		logits[i] = float32(math.Sin(float64(len(tokens)+i))) * 0.01
	}
	return logits, nil
}

// ParamCount returns the approximate parameter count.
func (b *BaselineGgufModel) ParamCount() int64 {
	return int64(b.Layers) * int64(b.Hidden*b.Hidden) * 4
}
