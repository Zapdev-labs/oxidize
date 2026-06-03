// Package vision implements multimodal vision encoders and preprocessors.
package vision

import "errors"

// Modality mirrors Modality.
type Modality string

// Recognised modalities.
const (
	ModalityImage  Modality = "image"
	ModalityVideo  Modality = "video"
	ModalityAudio  Modality = "audio"
)

// Preprocessor mirrors VisionPreprocessor.
type Preprocessor interface {
	Process(raw []byte, modality Modality) (any, error)
}

// Encoder mirrors VisionEncoder.
type Encoder interface {
	Encode(pixels any) (Vec, error)
	Dims() []int
}

// Vec is an n-dimensional float vector.
type Vec []float32

var clipImageMean = [3]float32{0.48145466, 0.4578275, 0.40821073}
var clipImageStd = [3]float32{0.26862954, 0.26130258, 0.27577711}

// Config mirrors VisionConfig.
type Config struct {
	ImageSize           int
	PatchSize           int
	NumPatches          int
	NumChannels         int
	HiddenSize          int
	NumHeads            int
	NumHiddenLayers     int
	IntermediateSize    int
	LayerNormEps        float32
	ProjectionDim       int
	ImageMean           [3]float32
	ImageStd            [3]float32
	NumImageTokens      int
	ModelName           string
	PatchGridCols       int
	PatchGridRows       int
}

// DefaultConfig returns CLIP-Large / LLaVA 1.5 defaults.
func DefaultConfig() Config { return ClipLarge() }

// ClipLarge mirrors VisionConfig::clip_large.
func ClipLarge() Config {
	return withClipNormalization(336, 14, 1024, 16, 24, 4096, 1e-5, 4096, "clip-large")
}

// LLaVA15 mirrors VisionConfig::llava_1_5.
func LLaVA15() Config { return ClipLarge() }

// ClipBase mirrors VisionConfig::clip_base.
func ClipBase() Config {
	return withClipNormalization(224, 14, 768, 12, 12, 3072, 1e-5, 2048, "clip-base")
}

// QwenVL mirrors VisionConfig::qwen_vl.
func QwenVL() Config {
	return withClipNormalization(448, 14, 1664, 16, 48, 6656, 1e-6, 4096, "qwen-vl")
}

func withClipNormalization(imageSize, patchSize, hidden, heads, layers, intermediate int, eps float32, projection int, name string) Config {
	side := imageSize / patchSize
	if patchSize == 0 {
		side = 0
	}
	return Config{
		ImageSize:        imageSize,
		PatchSize:        patchSize,
		NumPatches:       side * side,
		NumChannels:      3,
		HiddenSize:       hidden,
		NumHeads:         heads,
		NumHiddenLayers:  layers,
		IntermediateSize: intermediate,
		LayerNormEps:     eps,
		ProjectionDim:    projection,
		ImageMean:        clipImageMean,
		ImageStd:         clipImageStd,
		NumImageTokens:   side * side,
		ModelName:        name,
	}
}

// NumPatchesPerSide returns image_size / patch_size.
func (c Config) NumPatchesPerSide() int {
	if c.PatchSize == 0 {
		return 0
	}
	return c.ImageSize / c.PatchSize
}

// Patch returns (cols, rows) for the configured image size.
func (c Config) Patch() (cols, rows int) {
	if c.PatchSize == 0 {
		return 0, 0
	}
	cols = c.ImageSize / c.PatchSize
	rows = c.ImageSize / c.PatchSize
	if c.PatchGridCols > 0 {
		cols = c.PatchGridCols
	}
	if c.PatchGridRows > 0 {
		rows = c.PatchGridRows
	}
	return
}

// Error mirrors VisionError.
type Error struct{ Message string }

func (e *Error) Error() string { return "vision: " + e.Message }

// ErrUnsupportedModality is returned when the modality is not handled.
var ErrUnsupportedModality = errors.New("vision: unsupported modality")

// StubEncoder is a placeholder encoder used by tests and examples.
type StubEncoder struct{ Cfg Config }

// NewStubEncoder constructs a stub encoder.
func NewStubEncoder(cfg Config) *StubEncoder { return &StubEncoder{Cfg: cfg} }

// Encode returns zeros.
func (e *StubEncoder) Encode(pixels any) (Vec, error) {
	if pixels == nil {
		return nil, &Error{Message: "nil pixels"}
	}
	cols, rows := e.Cfg.Patch()
	out := make(Vec, cols*rows*e.Cfg.HiddenSize)
	return out, nil
}

// Dims returns the encoder output dimensions.
func (e *StubEncoder) Dims() []int {
	cols, rows := e.Cfg.Patch()
	return []int{1, cols*rows, e.Cfg.HiddenSize}
}

// StubPreprocessor is a placeholder preprocessor.
type StubPreprocessor struct{ Cfg Config }

// NewStubPreprocessor constructs a stub preprocessor.
func NewStubPreprocessor(cfg Config) *StubPreprocessor { return &StubPreprocessor{Cfg: cfg} }

// Process returns the raw bytes wrapped in a slice.
func (p *StubPreprocessor) Process(raw []byte, modality Modality) (any, error) {
	if modality != ModalityImage {
		return nil, ErrUnsupportedModality
	}
	return raw, nil
}

// MultimodalPrompt mirrors MultimodalPrompt.
type MultimodalPrompt struct {
	Text   string
	Tokens []int
	Images []Vec
}

// Encode returns a placeholder multimodal embedding.
func Encode(prompt MultimodalPrompt, enc Encoder) (Vec, error) {
	var out Vec
	for _, img := range prompt.Images {
		encoded, err := enc.Encode(img)
		if err != nil {
			return nil, err
		}
		out = append(out, encoded...)
	}
	return out, nil
}
