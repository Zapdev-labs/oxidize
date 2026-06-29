package video

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
)

// FrameEncoder encodes one frame's patch tensor into a flattened
// [num_patches, projection_dim] embedding matrix. It abstracts the per-frame
// vision encoder so the video stack stays decoupled from a concrete
// implementation. Mirrors the VisionEncoder::encode call used in encoder.rs.
type FrameEncoder interface {
	EncodeFrame(patches ImagePatches) ([]float32, error)
}

// VideoEncoderWeights holds the temporal encoder weights plus the LLM
// projection. Mirrors encoder.rs:VideoEncoderWeights.
type VideoEncoderWeights struct {
	Temporal TemporalWeights
	// PreProjectionNorm is applied before the LLM projection (len hidden_size).
	PreProjectionNorm []float32
	// Projection maps hidden_size -> llm_hidden_size (len hidden*llm).
	Projection []float32
	// FramePosEmbedding is the learnable frame-position embedding
	// (len max_frames*hidden). When empty, no positional info is added.
	FramePosEmbedding []float32
}

// ZeroVideoEncoderWeights builds zero-initialized weights for cfg.
func ZeroVideoEncoderWeights(cfg VideoConfig) VideoEncoderWeights {
	h := cfg.Temporal.HiddenSize
	llm := cfg.EffectiveLLMHidden()
	preNorm := make([]float32, h)
	for i := range preNorm {
		preNorm[i] = 1.0
	}
	return VideoEncoderWeights{
		Temporal:          ZeroTemporalWeights(cfg.Temporal),
		PreProjectionNorm: preNorm,
		Projection:        make([]float32, h*llm),
		FramePosEmbedding: make([]float32, cfg.Temporal.MaxFrames*h),
	}
}

// VideoEncoderWorkspace holds reusable scratch buffers for VideoEncoder.Encode.
// Mirrors encoder.rs:VideoEncoderWorkspace.
type VideoEncoderWorkspace struct {
	FrameTemporal []float32
	Projected     []float32
	TemporalWs    *TemporalWorkspace
}

// NewVideoEncoderWorkspace allocates scratch buffers for cfg's worst case.
func NewVideoEncoderWorkspace(cfg VideoConfig) *VideoEncoderWorkspace {
	llm := cfg.EffectiveLLMHidden()
	return &VideoEncoderWorkspace{
		FrameTemporal: make([]float32, cfg.Temporal.MaxFrames*cfg.Temporal.HiddenSize),
		Projected:     make([]float32, cfg.Temporal.MaxFrames*llm),
		TemporalWs:    NewTemporalWorkspace(cfg.Temporal),
	}
}

// VideoEncoder runs the vision encoder per frame, pools each frame to a vector,
// runs the temporal encoder over the frame axis, and projects to the LLM hidden
// size. Mirrors encoder.rs:VideoEncoder.
type VideoEncoder struct {
	config  VideoConfig
	vision  FrameEncoder
	weights VideoEncoderWeights
}

// NewVideoEncoder validates cfg and builds an encoder with zero weights.
func NewVideoEncoder(config VideoConfig, vision FrameEncoder) (*VideoEncoder, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if vision == nil {
		return nil, &Error{Message: "vision frame encoder must not be nil"}
	}
	return &VideoEncoder{
		config:  config,
		vision:  vision,
		weights: ZeroVideoEncoderWeights(config),
	}, nil
}

// Config returns the encoder configuration.
func (ve *VideoEncoder) Config() VideoConfig { return ve.config }

// Weights returns the current weights.
func (ve *VideoEncoder) Weights() *VideoEncoderWeights { return &ve.weights }

// LoadWeights validates and installs new temporal/projection weights.
// Mirrors encoder.rs:VideoEncoder::load_weights.
func (ve *VideoEncoder) LoadWeights(w VideoEncoderWeights) error {
	cfg := ve.config
	h := cfg.Temporal.HiddenSize
	llm := cfg.EffectiveLLMHidden()
	inter := cfg.Temporal.IntermediateSize
	if err := checkLen("pre_projection_norm", len(w.PreProjectionNorm), h); err != nil {
		return err
	}
	if err := checkLen("projection", len(w.Projection), h*llm); err != nil {
		return err
	}
	if err := checkLen("frame_pos_embedding", len(w.FramePosEmbedding), cfg.Temporal.MaxFrames*h); err != nil {
		return err
	}
	if len(w.Temporal.Layers) != cfg.Temporal.NumLayers {
		return &Error{Message: fmt.Sprintf("temporal_layers: expected %d got %d", cfg.Temporal.NumLayers, len(w.Temporal.Layers))}
	}
	for li := range w.Temporal.Layers {
		l := &w.Temporal.Layers[li]
		if err := checkLen("temporal_layer.attn_norm", len(l.AttnNorm), h); err != nil {
			return err
		}
		if err := checkLen("temporal_layer.q_proj", len(l.QProj), h*h); err != nil {
			return err
		}
		if err := checkLen("temporal_layer.k_proj", len(l.KProj), h*h); err != nil {
			return err
		}
		if err := checkLen("temporal_layer.v_proj", len(l.VProj), h*h); err != nil {
			return err
		}
		if err := checkLen("temporal_layer.o_proj", len(l.OProj), h*h); err != nil {
			return err
		}
		if err := checkLen("temporal_layer.ffn_gate", len(l.FfnGate), h*inter); err != nil {
			return err
		}
		if err := checkLen("temporal_layer.ffn_up", len(l.FfnUp), h*inter); err != nil {
			return err
		}
		if err := checkLen("temporal_layer.ffn_down", len(l.FfnDown), inter*h); err != nil {
			return err
		}
	}
	if err := checkLen("temporal_final_norm", len(w.Temporal.FinalNorm), h); err != nil {
		return err
	}
	if cfg.Temporal.UseClsToken && len(w.Temporal.ClsToken) != h {
		return &Error{Message: fmt.Sprintf("temporal_cls_token: expected %d got %d", h, len(w.Temporal.ClsToken))}
	}
	if !cfg.Temporal.UseClsToken && len(w.Temporal.ClsToken) != 0 {
		return &Error{Message: fmt.Sprintf("temporal_cls_token: expected 0 got %d", len(w.Temporal.ClsToken))}
	}
	ve.weights = w
	return nil
}

// Encode produces a [num_frames, llm_hidden_size] token matrix from
// preprocessed frames. Mirrors encoder.rs:VideoEncoder::encode.
func (ve *VideoEncoder) Encode(frames *VideoFrames, ws *VideoEncoderWorkspace) ([]float32, error) {
	if err := ve.config.Validate(); err != nil {
		return nil, err
	}
	nFrames := frames.FrameCount()
	if nFrames == 0 {
		return []float32{}, nil
	}
	if nFrames > ve.config.Temporal.MaxFrames {
		return nil, &Error{Message: fmt.Sprintf(
			"frame count %d out of range [1, %d]", nFrames, ve.config.Temporal.MaxFrames)}
	}
	if ws == nil {
		ws = NewVideoEncoderWorkspace(ve.config)
	}

	projectionDim := ve.config.Vision.ProjectionDim
	hidden := ve.config.Temporal.HiddenSize
	llm := ve.config.EffectiveLLMHidden()
	numPatches := ve.config.Vision.NumPatches()

	// ---- 1. Vision encoder per frame (parallel) ----
	pooled := make([][]float32, nFrames)
	errs := make([]error, nFrames)
	workers := runtime.GOMAXPROCS(0)
	if workers > nFrames {
		workers = nFrames
	}
	if workers < 1 {
		workers = 1
	}
	var next int
	var mu sync.Mutex
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				i := next
				next++
				mu.Unlock()
				if i >= nFrames {
					return
				}
				emb, err := ve.vision.EncodeFrame(frames.Frames[i])
				if err != nil {
					errs[i] = err
					continue
				}
				pooled[i] = emb
			}
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			return nil, &Error{Message: fmt.Sprintf("frame %d vision encode: %v", i, err)}
		}
	}

	// ---- Pool each frame to a single hidden-sized vector ----
	for i := 0; i < nFrames; i++ {
		patches := pooled[i]
		if len(patches) != numPatches*projectionDim {
			return nil, &Error{Message: fmt.Sprintf(
				"vision embedding length %d != num_patches*projection_dim (%d*%d)",
				len(patches), numPatches, projectionDim)}
		}
		dst := ws.FrameTemporal[i*hidden : (i+1)*hidden]
		switch ve.config.Pool {
		case PoolMean:
			for d := 0; d < projectionDim; d++ {
				var sum float32
				for p := 0; p < numPatches; p++ {
					sum += patches[p*projectionDim+d]
				}
				dst[d] = sum / float32(numPatches)
			}
		case PoolClsToken:
			copy(dst[:projectionDim], patches[:projectionDim])
		case PoolLastToken:
			start := (numPatches - 1) * projectionDim
			copy(dst[:projectionDim], patches[start:start+projectionDim])
		}
		for d := projectionDim; d < hidden; d++ {
			dst[d] = 0.0
		}
		if len(ve.weights.FramePosEmbedding) != 0 {
			pos := ve.weights.FramePosEmbedding[i*hidden : (i+1)*hidden]
			for d := 0; d < hidden; d++ {
				dst[d] += pos[d]
			}
		}
	}

	// ---- 2. Temporal self-attention ----
	temporalInput := ws.FrameTemporal[:nFrames*hidden]
	temporalOut, err := ForwardTemporal(ve.config.Temporal, &ve.weights.Temporal, temporalInput, nFrames, ws.TemporalWs)
	if err != nil {
		return nil, err
	}

	// ---- 3. Pre-projection norm + LLM projection ----
	offset := 0
	if ve.config.Temporal.UseClsToken {
		offset = 1
	}
	normalized := make([]float32, hidden)
	for i := 0; i < nFrames; i++ {
		src := temporalOut[(i+offset)*hidden : (i+offset+1)*hidden]
		if err := tensor.RMSNormF32(src, ve.weights.PreProjectionNorm, normalized, ve.config.Temporal.RmsNormEps); err != nil {
			return nil, &Error{Message: fmt.Sprintf("pre_proj norm: %v", err)}
		}
		dst := ws.Projected[i*llm : (i+1)*llm]
		if err := tensor.GemmF32(normalized, ve.weights.Projection, 1, hidden, llm, dst); err != nil {
			return nil, &Error{Message: fmt.Sprintf("projection: %v", err)}
		}
	}

	out := make([]float32, nFrames*llm)
	copy(out, ws.Projected[:nFrames*llm])
	return out, nil
}

func checkLen(name string, actual, expected int) error {
	if actual == expected {
		return nil
	}
	return &Error{Message: fmt.Sprintf("%s shape mismatch: expected %d got %d", name, expected, actual)}
}
