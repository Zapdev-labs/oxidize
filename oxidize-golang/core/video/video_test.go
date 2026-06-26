package video

import "testing"

func blackFrame(w, h int) DecodedFrame {
	f, err := NewDecodedFrame(w, h, make([]byte, w*h*3))
	if err != nil {
		panic(err)
	}
	return *f
}

func TestResizingDecoderPreservesCountAndShape(t *testing.T) {
	frames := []DecodedFrame{blackFrame(2, 2), blackFrame(4, 4)}
	dec := ResizingDecoder{Inner: RawFrameDecoder{}, TargetWidth: 8, TargetHeight: 8}
	out, err := dec.Decode(VideoSource{Frames: frames})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 frames, got %d", len(out))
	}
	for _, f := range out {
		if f.Width != 8 || f.Height != 8 || len(f.Data) != 8*8*3 {
			t.Fatalf("bad resized frame %dx%d len=%d", f.Width, f.Height, len(f.Data))
		}
	}
}

func TestTemporalConfigValidate(t *testing.T) {
	cfg := DefaultTemporalConfig()
	cfg.HiddenSize = 8
	cfg.NumHeads = 2
	cfg.NumLayers = 1
	cfg.IntermediateSize = 16
	cfg.MaxFrames = 4
	cfg.UseClsToken = false
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
	if cfg.HeadDim() != 4 {
		t.Fatalf("head_dim = %d, want 4", cfg.HeadDim())
	}
	bad := cfg
	bad.NumHeads = 3 // 8 not divisible by 3
	if err := bad.Validate(); err == nil {
		t.Fatal("expected divisibility error")
	}
}

func tinyTemporalConfig() TemporalConfig {
	return TemporalConfig{
		HiddenSize:       8,
		NumLayers:        1,
		NumHeads:         2,
		IntermediateSize: 16,
		RmsNormEps:       1e-5,
		MaxFrames:        4,
		RopeTheta:        10000.0,
		UseClsToken:      false,
		LayerDropout:     0.0,
	}
}

func TestForwardTemporalZeroWeights(t *testing.T) {
	cfg := tinyTemporalConfig()
	w := ZeroTemporalWeights(cfg)
	input := make([]float32, cfg.HiddenSize)
	for i := range input {
		input[i] = float32(i) * 0.1
	}
	ws := NewTemporalWorkspace(cfg)
	out, err := ForwardTemporal(cfg, &w, input, 1, ws)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != cfg.HiddenSize {
		t.Fatalf("len = %d, want %d", len(out), cfg.HiddenSize)
	}
}

func TestForwardTemporalRejectsEmpty(t *testing.T) {
	cfg := tinyTemporalConfig()
	w := ZeroTemporalWeights(cfg)
	ws := NewTemporalWorkspace(cfg)
	if _, err := ForwardTemporal(cfg, &w, nil, 0, ws); err == nil {
		t.Fatal("expected frame-count error")
	}
}

func TestForwardTemporalMultiFrame(t *testing.T) {
	cfg := tinyTemporalConfig()
	w := ZeroTemporalWeights(cfg)
	h := cfg.HiddenSize
	input := make([]float32, 3*h)
	for i := range input {
		input[i] = 0.1
	}
	ws := NewTemporalWorkspace(cfg)
	out, err := ForwardTemporal(cfg, &w, input, 3, ws)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3*h {
		t.Fatalf("len = %d, want %d", len(out), 3*h)
	}
}

// identityFrameEncoder returns a fixed [num_patches, projection_dim] embedding.
type identityFrameEncoder struct {
	numPatches    int
	projectionDim int
}

func (e identityFrameEncoder) EncodeFrame(p ImagePatches) ([]float32, error) {
	out := make([]float32, e.numPatches*e.projectionDim)
	for i := range out {
		out[i] = 0.05
	}
	return out, nil
}

func tinyVideoConfig() VideoConfig {
	vision := VisionConfig{
		ImageSize:        4,
		PatchSize:        2,
		HiddenSize:       4,
		NumHeads:         1,
		NumHiddenLayers:  1,
		IntermediateSize: 8,
		LayerNormEps:     1e-5,
		ProjectionDim:    4,
		ImageStd:         [3]float32{1, 1, 1},
		NumImageTokens:   4,
	}
	temporal := tinyTemporalConfig()
	temporal.HiddenSize = vision.ProjectionDim
	temporal.NumHeads = 2
	temporal.IntermediateSize = 8
	temporal.MaxFrames = 4
	return VideoConfig{
		Vision:        vision,
		Temporal:      temporal,
		Sampling:      SampleUniform,
		TargetFrames:  2,
		LLMHiddenSize: 0,
		Pool:          PoolMean,
	}
}

func TestVideoEncoderEncodeEmpty(t *testing.T) {
	cfg := tinyVideoConfig()
	enc, err := NewVideoEncoder(cfg, identityFrameEncoder{cfg.Vision.NumPatches(), cfg.Vision.ProjectionDim})
	if err != nil {
		t.Fatal(err)
	}
	ws := NewVideoEncoderWorkspace(cfg)
	out, err := enc.Encode(&VideoFrames{}, ws)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 0 {
		t.Fatalf("expected empty output, got %d", len(out))
	}
}

func TestVideoEncoderEncodeFrames(t *testing.T) {
	cfg := tinyVideoConfig()
	np := cfg.Vision.NumPatches()
	enc, err := NewVideoEncoder(cfg, identityFrameEncoder{np, cfg.Vision.ProjectionDim})
	if err != nil {
		t.Fatal(err)
	}
	pd := cfg.Vision.PatchDim()
	frame := ImagePatches{Data: make([]float32, np*pd), NumPatches: np, PatchDim: pd}
	vf := &VideoFrames{
		Frames:  []ImagePatches{frame, frame},
		Widths:  []int{4, 4},
		Heights: []int{4, 4},
	}
	ws := NewVideoEncoderWorkspace(cfg)
	out, err := enc.Encode(vf, ws)
	if err != nil {
		t.Fatal(err)
	}
	llm := cfg.EffectiveLLMHidden()
	if len(out) != 2*llm {
		t.Fatalf("len = %d, want %d", len(out), 2*llm)
	}
}

func TestVideoEncoderTooManyFrames(t *testing.T) {
	cfg := tinyVideoConfig()
	cfg.Temporal.MaxFrames = 2
	cfg.TargetFrames = 2
	np := cfg.Vision.NumPatches()
	enc, err := NewVideoEncoder(cfg, identityFrameEncoder{np, cfg.Vision.ProjectionDim})
	if err != nil {
		t.Fatal(err)
	}
	pd := cfg.Vision.PatchDim()
	frame := ImagePatches{Data: make([]float32, np*pd), NumPatches: np, PatchDim: pd}
	vf := &VideoFrames{Frames: []ImagePatches{frame, frame, frame}}
	ws := NewVideoEncoderWorkspace(cfg)
	if _, err := enc.Encode(vf, ws); err == nil {
		t.Fatal("expected frame-count error")
	}
}

func TestVideoPreprocessorParallel(t *testing.T) {
	cfg := tinyVideoConfig().Vision
	vp := NewVideoPreprocessor(cfg)
	frames := []DecodedFrame{blackFrame(4, 4), blackFrame(4, 4), blackFrame(4, 4)}
	vf, err := vp.PreprocessFrames(frames)
	if err != nil {
		t.Fatal(err)
	}
	if vf.FrameCount() != 3 {
		t.Fatalf("frame count = %d, want 3", vf.FrameCount())
	}
	for _, p := range vf.Frames {
		if p.NumPatches != cfg.NumPatches() || p.PatchDim != cfg.PatchDim() {
			t.Fatalf("bad patch shape %d/%d", p.NumPatches, p.PatchDim)
		}
	}
}

func TestVideoPreprocessorRejectsMixedResolution(t *testing.T) {
	cfg := tinyVideoConfig().Vision
	vp := NewVideoPreprocessor(cfg)
	frames := []DecodedFrame{blackFrame(4, 4), blackFrame(6, 4)}
	if _, err := vp.PreprocessFrames(frames); err == nil {
		t.Fatal("expected mixed-resolution error")
	}
}

func TestLoadWeightsRejectsWrongProjection(t *testing.T) {
	cfg := tinyVideoConfig()
	np := cfg.Vision.NumPatches()
	enc, err := NewVideoEncoder(cfg, identityFrameEncoder{np, cfg.Vision.ProjectionDim})
	if err != nil {
		t.Fatal(err)
	}
	bad := ZeroVideoEncoderWeights(cfg)
	bad.Projection = make([]float32, 3)
	if err := enc.LoadWeights(bad); err == nil {
		t.Fatal("expected weight-shape error")
	}
}

func TestRawFrameDecoder(t *testing.T) {
	frame, err := NewDecodedFrame(2, 2, make([]byte, 12))
	if err != nil {
		t.Fatal(err)
	}
	dec := RawFrameDecoder{}
	out, err := dec.Decode(VideoSource{SingleImage: frame})
	if err != nil || len(out) != 1 {
		t.Fatalf("decode: %v len=%d", err, len(out))
	}
}

func TestSampleIndicesUniform(t *testing.T) {
	idx, err := SampleIndices(100, 8, SampleUniform)
	if err != nil {
		t.Fatal(err)
	}
	if len(idx) != 8 {
		t.Fatalf("expected 8 indices, got %d", len(idx))
	}
}

func TestVideoPromptBuildSequence(t *testing.T) {
	table := make([]float32, 4*2)
	for i := range table {
		table[i] = float32(i)
	}
	p := NewVideoPrompt()
	p.AddText([]uint32{0, 1})
	out, err := p.BuildSequence(table, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("expected 4 floats, got %d", len(out))
	}
}
