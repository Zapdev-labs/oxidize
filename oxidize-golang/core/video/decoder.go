package video

// ResizingDecoder wraps another VideoDecoder and resizes every decoded frame to
// a target resolution using nearest-neighbor sampling. Mirrors
// oxidize-core/src/video/decoder.rs:ResizingDecoder.
type ResizingDecoder struct {
	Inner        VideoDecoder
	TargetWidth  int
	TargetHeight int
}

// Decode runs the inner decoder then resizes each frame to TargetWidth x
// TargetHeight. Frames already at the target resolution are passed through
// untouched.
func (d ResizingDecoder) Decode(source VideoSource) ([]DecodedFrame, error) {
	if d.Inner == nil {
		return nil, &Error{Message: "ResizingDecoder requires an inner decoder"}
	}
	if d.TargetWidth <= 0 || d.TargetHeight <= 0 {
		return nil, &Error{Message: "ResizingDecoder target dimensions must be positive"}
	}
	frames, err := d.Inner.Decode(source)
	if err != nil {
		return nil, err
	}
	out := make([]DecodedFrame, 0, len(frames))
	for i := range frames {
		f := &frames[i]
		if f.Width == d.TargetWidth && f.Height == d.TargetHeight {
			dup := DecodedFrame{Width: f.Width, Height: f.Height, Data: append([]byte(nil), f.Data...)}
			out = append(out, dup)
			continue
		}
		resized := resizeRGBNearest(f.Data, f.Width, f.Height, d.TargetWidth, d.TargetHeight)
		out = append(out, DecodedFrame{
			Width:  d.TargetWidth,
			Height: d.TargetHeight,
			Data:   resized,
		})
	}
	return out, nil
}

// resizeRGBNearest resizes a row-major RGB image (3 bytes/pixel) to dstW x dstH
// using integer nearest-neighbor sampling, matching the Rust reference exactly.
func resizeRGBNearest(src []byte, srcW, srcH, dstW, dstH int) []byte {
	dst := make([]byte, dstW*dstH*3)
	if srcW <= 0 || srcH <= 0 {
		return dst
	}
	for dy := 0; dy < dstH; dy++ {
		sy := dy * srcH / dstH
		for dx := 0; dx < dstW; dx++ {
			sx := dx * srcW / dstW
			srcIdx := (sy*srcW + sx) * 3
			dstIdx := (dy*dstW + dx) * 3
			if srcIdx+3 <= len(src) {
				copy(dst[dstIdx:dstIdx+3], src[srcIdx:srcIdx+3])
			}
		}
	}
	return dst
}
