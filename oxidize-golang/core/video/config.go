package video

import "fmt"

// VisionConfig is the subset of vision-encoder configuration the video stack
type VisionConfig struct {
	ImageSize        int
	PatchSize        int
	HiddenSize       int
	NumHeads         int
	NumHiddenLayers  int
	IntermediateSize int
	LayerNormEps     float32
	ProjectionDim    int
	ImageMean        [3]float32
	ImageStd         [3]float32
	NumImageTokens   int
}

// ClipBaseVision returns a small CLIP-base style vision config.
func ClipBaseVision() VisionConfig {
	return VisionConfig{
		ImageSize:        224,
		PatchSize:        14,
		HiddenSize:       768,
		NumHeads:         12,
		NumHiddenLayers:  12,
		IntermediateSize: 3072,
		LayerNormEps:     1e-5,
		ProjectionDim:    2048,
		ImageMean:        [3]float32{0.48145466, 0.4578275, 0.40821073},
		ImageStd:         [3]float32{0.26862954, 0.26130258, 0.27577711},
		NumImageTokens:   256,
	}
}

// NumPatchesPerSide returns image_size / patch_size.
func (c VisionConfig) NumPatchesPerSide() int {
	if c.PatchSize == 0 {
		return 0
	}
	return c.ImageSize / c.PatchSize
}

// NumPatches returns the patch grid count (side * side).
func (c VisionConfig) NumPatches() int {
	side := c.NumPatchesPerSide()
	return side * side
}

// PatchDim returns the flattened patch dimension (3 * patch_size^2).
func (c VisionConfig) PatchDim() int {
	return 3 * c.PatchSize * c.PatchSize
}

// Validate checks basic vision-config invariants.
func (c VisionConfig) Validate() error {
	if c.ImageSize == 0 || c.PatchSize == 0 {
		return &Error{Message: "image_size and patch_size must be non-zero"}
	}
	if c.ImageSize%c.PatchSize != 0 {
		return &Error{Message: "image_size must be divisible by patch_size"}
	}
	if c.ProjectionDim == 0 {
		return &Error{Message: "projection_dim must be non-zero"}
	}
	return nil
}

// TemporalConfig configures the temporal encoder stacked on top of the
// per-frame vision encoder. Mirrors oxidize-core/src/video/config.rs:TemporalConfig.
type TemporalConfig struct {
	HiddenSize       int
	NumLayers        int
	NumHeads         int
	IntermediateSize int
	RmsNormEps       float32
	MaxFrames        int
	RopeTheta        float32
	UseClsToken      bool
	// LayerDropout is stored for checkpoint compatibility but unused at
	// inference time.
	LayerDropout float32
}

// DefaultTemporalConfig mirrors Rust TemporalConfig::default().
func DefaultTemporalConfig() TemporalConfig {
	return TemporalConfig{
		HiddenSize:       1024,
		NumLayers:        2,
		NumHeads:         8,
		IntermediateSize: 4096,
		RmsNormEps:       1e-5,
		MaxFrames:        32,
		RopeTheta:        10000.0,
		UseClsToken:      true,
		LayerDropout:     0.0,
	}
}

// HeadDim returns hidden_size / num_heads (0 when num_heads is 0).
func (tc TemporalConfig) HeadDim() int {
	if tc.NumHeads == 0 {
		return 0
	}
	return tc.HiddenSize / tc.NumHeads
}

// Validate mirrors TemporalConfig::validate.
func (tc TemporalConfig) Validate() error {
	if tc.HiddenSize == 0 {
		return &Error{Message: "hidden_size must be non-zero"}
	}
	if tc.NumHeads == 0 {
		return &Error{Message: "num_heads must be non-zero"}
	}
	if tc.HiddenSize%tc.NumHeads != 0 {
		return &Error{Message: "hidden_size must be divisible by num_heads"}
	}
	if tc.NumLayers == 0 {
		return &Error{Message: "num_layers must be non-zero"}
	}
	if tc.IntermediateSize == 0 {
		return &Error{Message: "intermediate_size must be non-zero"}
	}
	if tc.MaxFrames == 0 {
		return &Error{Message: "max_frames must be non-zero"}
	}
	if tc.RmsNormEps <= 0.0 {
		return &Error{Message: "rms_norm_eps must be positive"}
	}
	if tc.RopeTheta <= 0.0 {
		return &Error{Message: "rope_theta must be positive"}
	}
	return nil
}

// TemporalPool selects how per-frame patch embeddings are aggregated into one
// vector per frame. Mirrors config.rs:TemporalPool.
type TemporalPool uint8

const (
	// PoolMean averages all patch embeddings per frame.
	PoolMean TemporalPool = iota
	// PoolClsToken uses the first (class) token per frame.
	PoolClsToken
	// PoolLastToken uses the last patch token per frame.
	PoolLastToken
)

// VideoConfig is the top-level video model configuration. Mirrors
// config.rs:VideoConfig.
type VideoConfig struct {
	Vision   VisionConfig
	Temporal TemporalConfig
	Sampling FrameSamplingStrategy
	// TargetFrames is the number of frames the sampler produces; must be
	// <= Temporal.MaxFrames.
	TargetFrames int
	// LLMHiddenSize is the output projection dim. When 0 the temporal hidden
	// size is reused.
	LLMHiddenSize     int
	Pool              TemporalPool
	VideoStartTokenID uint32
	VideoEndTokenID   uint32
}

// DefaultVideoConfig returns a small CPU-friendly video config.
func DefaultVideoConfig() VideoConfig {
	vision := ClipBaseVision()
	temporal := DefaultTemporalConfig()
	temporal.HiddenSize = vision.ProjectionDim
	temporal.NumLayers = 2
	temporal.NumHeads = 4
	temporal.IntermediateSize = 2048
	temporal.MaxFrames = 16
	return VideoConfig{
		Vision:        vision,
		Temporal:      temporal,
		Sampling:      SampleUniform,
		TargetFrames:  8,
		LLMHiddenSize: 0,
		Pool:          PoolMean,
	}
}

// EffectiveLLMHidden returns LLMHiddenSize, falling back to the temporal hidden
// size when 0.
func (c VideoConfig) EffectiveLLMHidden() int {
	if c.LLMHiddenSize == 0 {
		return c.Temporal.HiddenSize
	}
	return c.LLMHiddenSize
}

// Validate mirrors VideoConfig::validate.
func (c VideoConfig) Validate() error {
	if err := c.Vision.Validate(); err != nil {
		return &Error{Message: fmt.Sprintf("vision: %v", err)}
	}
	if c.Vision.ProjectionDim != c.Temporal.HiddenSize {
		return &Error{Message: fmt.Sprintf(
			"temporal.hidden_size (%d) must equal vision.projection_dim (%d)",
			c.Temporal.HiddenSize, c.Vision.ProjectionDim)}
	}
	if err := c.Temporal.Validate(); err != nil {
		return err
	}
	if c.TargetFrames == 0 {
		return &Error{Message: "target_frames must be non-zero"}
	}
	if c.TargetFrames > c.Temporal.MaxFrames {
		return &Error{Message: fmt.Sprintf(
			"target_frames (%d) exceeds temporal.max_frames (%d)",
			c.TargetFrames, c.Temporal.MaxFrames)}
	}
	return nil
}
