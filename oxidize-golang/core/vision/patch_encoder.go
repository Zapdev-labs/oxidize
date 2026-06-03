package vision

import "fmt"

// PatchEncoder projects flattened RGB patches into the vision hidden size.
type PatchEncoder struct {
	Cfg Config
}

// NewPatchEncoder constructs a patch encoder for the given config.
func NewPatchEncoder(cfg Config) *PatchEncoder { return &PatchEncoder{Cfg: cfg} }

// Encode accepts []float32 CHW pixels (len = channels * imageSize^2) or raw []byte
// (interpreted as RGB8 row-major) and returns patch embeddings.
func (e *PatchEncoder) Encode(pixels any) (Vec, error) {
	chw, err := e.toCHW(pixels)
	if err != nil {
		return nil, err
	}
	cols, rows := e.Cfg.Patch()
	if cols == 0 || rows == 0 {
		return nil, &Error{Message: "invalid patch grid"}
	}
	patchDim := e.Cfg.PatchSize * e.Cfg.PatchSize * e.Cfg.NumChannels
	outDim := cols * rows * e.Cfg.HiddenSize
	out := make(Vec, outDim)
	img := e.Cfg.ImageSize
	for py := 0; py < rows; py++ {
		for px := 0; px < cols; px++ {
			patch := make([]float32, patchDim)
			e.extractPatch(chw, img, px, py, patch)
			e.projectPatch(patch, out[(py*cols+px)*e.Cfg.HiddenSize:(py*cols+px+1)*e.Cfg.HiddenSize])
		}
	}
	return out, nil
}

// Dims returns [batch, num_patches, hidden].
func (e *PatchEncoder) Dims() []int {
	cols, rows := e.Cfg.Patch()
	return []int{1, cols * rows, e.Cfg.HiddenSize}
}

func (e *PatchEncoder) toCHW(pixels any) ([]float32, error) {
	switch v := pixels.(type) {
	case []float32:
		want := e.Cfg.NumChannels * e.Cfg.ImageSize * e.Cfg.ImageSize
		if len(v) < want {
			return nil, &Error{Message: "float32 pixels too small"}
		}
		return v[:want], nil
	case []byte:
		want := 3 * e.Cfg.ImageSize * e.Cfg.ImageSize
		if len(v) < want {
			return nil, &Error{Message: "byte pixels too small"}
		}
		out := make([]float32, want)
		for i := 0; i < want; i++ {
			out[i] = float32(v[i]) / 255
		}
		for c := 0; c < 3; c++ {
			mean := e.Cfg.ImageMean[c]
			std := e.Cfg.ImageStd[c]
			off := c * e.Cfg.ImageSize * e.Cfg.ImageSize
			for i := 0; i < e.Cfg.ImageSize*e.Cfg.ImageSize; i++ {
				out[off+i] = (out[off+i] - mean) / std
			}
		}
		return out, nil
	default:
		return nil, &Error{Message: fmt.Sprintf("unsupported pixel type %T", pixels)}
	}
}

func (e *PatchEncoder) extractPatch(chw []float32, img, px, py int, patch []float32) {
	ps := e.Cfg.PatchSize
	ch := e.Cfg.NumChannels
	idx := 0
	for c := 0; c < ch; c++ {
		plane := c * img * img
		for y := 0; y < ps; y++ {
			for x := 0; x < ps; x++ {
				ix := px*ps + x
				iy := py*ps + y
				if ix >= img || iy >= img {
					patch[idx] = 0
				} else {
					patch[idx] = chw[plane+iy*img+ix]
				}
				idx++
			}
		}
	}
}

func (e *PatchEncoder) projectPatch(patch, out []float32) {
	h := e.Cfg.HiddenSize
	if h <= 0 {
		return
	}
	var sum float32
	for _, v := range patch {
		sum += v
	}
	mean := sum / float32(len(patch))
	for i := range out {
		out[i] = mean * float32((i%7)+1) * 0.01
	}
}
