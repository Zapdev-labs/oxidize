// Package video implements CPU-first video understanding helpers ported from
// oxidize-core/src/video/.
package video

import (
	"errors"
	"fmt"
)

// FrameSamplingStrategy selects how frames are subsampled from a clip.
type FrameSamplingStrategy uint8

const (
	SampleUniform FrameSamplingStrategy = iota
	SampleDense
	SampleAdaptive
)

// Config holds video preprocessing defaults.
type Config struct {
	TargetFrames int
	Strategy     FrameSamplingStrategy
	DenseStride  int
}

// DefaultConfig returns sensible defaults for short clips.
func DefaultConfig() Config {
	return Config{TargetFrames: 8, Strategy: SampleUniform, DenseStride: 1}
}

// Error is returned for invalid video inputs.
type Error struct{ Message string }

func (e *Error) Error() string { return "video: " + e.Message }

var (
	ErrEmptySample        = errors.New("video: empty frame sample")
	ErrFrameCountOutRange = errors.New("video: frame count out of range")
)

// DecodedFrame is a single RGB frame in row-major layout (3 bytes per pixel).
type DecodedFrame struct {
	Width  int
	Height int
	Data   []byte
}

// NewDecodedFrame validates dimensions and payload length.
func NewDecodedFrame(width, height int, data []byte) (*DecodedFrame, error) {
	expected := width * height * 3
	if width <= 0 || height <= 0 || len(data) != expected {
		return nil, &Error{Message: fmt.Sprintf("invalid frame %dx%d bytes=%d", width, height, len(data))}
	}
	out := make([]byte, len(data))
	copy(out, data)
	return &DecodedFrame{Width: width, Height: height, Data: out}, nil
}

// VideoSource identifies input to a decoder.
type VideoSource struct {
	Frames      []DecodedFrame
	SingleImage *DecodedFrame
}

// VideoDecoder decodes a source into RGB frames.
type VideoDecoder interface {
	Decode(source VideoSource) ([]DecodedFrame, error)
}

// RawFrameDecoder returns pre-decoded frames unchanged.
type RawFrameDecoder struct{}

func (RawFrameDecoder) Decode(source VideoSource) ([]DecodedFrame, error) {
	if len(source.Frames) > 0 {
		out := make([]DecodedFrame, len(source.Frames))
		copy(out, source.Frames)
		return out, nil
	}
	if source.SingleImage != nil {
		return []DecodedFrame{*source.SingleImage}, nil
	}
	return nil, ErrFrameCountOutRange
}

// RepetitiveFrameDecoder repeats a single image n times (CLI --video-frame mode).
type RepetitiveFrameDecoder struct{ Count int }

func (d RepetitiveFrameDecoder) Decode(source VideoSource) ([]DecodedFrame, error) {
	n := d.Count
	if n <= 0 {
		n = 1
	}
	img := source.SingleImage
	if img == nil && len(source.Frames) == 1 {
		img = &source.Frames[0]
	}
	if img == nil {
		return nil, ErrFrameCountOutRange
	}
	out := make([]DecodedFrame, n)
	for i := range out {
		dup := *img
		dup.Data = append([]byte(nil), img.Data...)
		out[i] = dup
	}
	return out, nil
}
