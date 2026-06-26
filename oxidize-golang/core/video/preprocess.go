package video

import (
	"fmt"
	"runtime"
	"sync"
)

// ImagePatches holds the flattened patch tensor for a single frame:
// `num_patches` rows of length `patch_dim`, row-major. Mirrors the relevant
// fields of oxidize-core's vision::ImagePatches.
type ImagePatches struct {
	Data           []float32
	NumPatches     int
	PatchDim       int
	OriginalWidth  int
	OriginalHeight int
}

// VideoFrames is the container of preprocessed per-frame patches consumed by
// the VideoEncoder. Mirrors preprocess.rs:VideoFrames.
type VideoFrames struct {
	Frames  []ImagePatches
	Widths  []int
	Heights []int
}

// FrameCount returns the number of frames.
func (vf *VideoFrames) FrameCount() int {
	if vf == nil {
		return 0
	}
	return len(vf.Frames)
}

// TotalPatches sums patch counts across all frames.
func (vf *VideoFrames) TotalPatches() int {
	total := 0
	for i := range vf.Frames {
		total += vf.Frames[i].NumPatches
	}
	return total
}

// FramePreprocessFn extracts patch tensors from a single decoded RGB frame.
// It mirrors ImagePreprocessor::preprocess_rgb and is supplied by the caller so
// the video package stays decoupled from a specific vision implementation.
type FramePreprocessFn func(data []byte, width, height int) (ImagePatches, error)

// VideoPreprocessor validates frame consistency and runs per-frame
// preprocessing in parallel. Mirrors preprocess.rs:VideoPreprocessor.
type VideoPreprocessor struct {
	Config     VisionConfig
	Preprocess FramePreprocessFn
}

// NewVideoPreprocessor builds a preprocessor with a default patch extractor
// derived from the vision config. The default extractor performs a
// nearest-neighbor resize to ImageSize and splits the frame into a grid of
// flattened RGB patches, normalized with the config's mean/std.
func NewVideoPreprocessor(config VisionConfig) *VideoPreprocessor {
	vp := &VideoPreprocessor{Config: config}
	vp.Preprocess = vp.defaultPreprocess
	return vp
}

// PreprocessFrames preprocesses a sequence of decoded RGB frames in parallel.
// All frames must share the same resolution. Mirrors VideoPreprocessor::preprocess.
func (vp *VideoPreprocessor) PreprocessFrames(frames []DecodedFrame) (*VideoFrames, error) {
	if len(frames) == 0 {
		return &VideoFrames{}, nil
	}
	first := &frames[0]
	for idx := 1; idx < len(frames); idx++ {
		if frames[idx].Width != first.Width || frames[idx].Height != first.Height {
			return nil, &Error{Message: fmt.Sprintf(
				"frame %d has dims %dx%d, expected %dx%d",
				idx, frames[idx].Width, frames[idx].Height, first.Width, first.Height)}
		}
	}
	if vp.Preprocess == nil {
		vp.Preprocess = vp.defaultPreprocess
	}

	out := make([]ImagePatches, len(frames))
	errs := make([]error, len(frames))

	workers := runtime.GOMAXPROCS(0)
	if workers > len(frames) {
		workers = len(frames)
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
				if i >= len(frames) {
					return
				}
				patches, err := vp.Preprocess(frames[i].Data, frames[i].Width, frames[i].Height)
				if err != nil {
					errs[i] = err
					continue
				}
				out[i] = patches
			}
		}()
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			return nil, &Error{Message: fmt.Sprintf("frame %d preprocess: %v", i, err)}
		}
	}

	widths := make([]int, len(frames))
	heights := make([]int, len(frames))
	for i := range frames {
		widths[i] = frames[i].Width
		heights[i] = frames[i].Height
	}
	return &VideoFrames{Frames: out, Widths: widths, Heights: heights}, nil
}

// defaultPreprocess resizes to ImageSize, then extracts a grid of normalized,
// flattened RGB patches matching the vision config's patch layout.
func (vp *VideoPreprocessor) defaultPreprocess(data []byte, width, height int) (ImagePatches, error) {
	cfg := vp.Config
	if err := cfg.Validate(); err != nil {
		return ImagePatches{}, err
	}
	size := cfg.ImageSize
	resized := data
	if width != size || height != size {
		resized = resizeRGBNearest(data, width, height, size, size)
	}
	side := cfg.NumPatchesPerSide()
	ps := cfg.PatchSize
	numPatches := side * side
	patchDim := cfg.PatchDim()
	out := make([]float32, numPatches*patchDim)

	mean := cfg.ImageMean
	std := cfg.ImageStd
	for py := 0; py < side; py++ {
		for px := 0; px < side; px++ {
			patchIdx := py*side + px
			base := patchIdx * patchDim
			d := 0
			for yy := 0; yy < ps; yy++ {
				srcY := py*ps + yy
				for xx := 0; xx < ps; xx++ {
					srcX := px*ps + xx
					srcIdx := (srcY*size + srcX) * 3
					for c := 0; c < 3; c++ {
						v := float32(resized[srcIdx+c]) / 255.0
						s := std[c]
						if s == 0 {
							s = 1
						}
						out[base+d] = (v - mean[c]) / s
						d++
					}
				}
			}
		}
	}
	return ImagePatches{
		Data:           out,
		NumPatches:     numPatches,
		PatchDim:       patchDim,
		OriginalWidth:  width,
		OriginalHeight: height,
	}, nil
}
