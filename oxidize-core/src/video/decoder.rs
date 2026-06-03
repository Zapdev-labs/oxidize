//! Video decoding primitives.
//!
//! The core video model accepts a stream of RGB frames in row-major layout.
//! This module defines a [`VideoDecoder`] trait plus CPU-friendly reference
//! implementations:
//!
//! * [`RawFrameDecoder`] — used when the caller has already decoded the video
//!   (e.g. via FFmpeg) and just wants the model to consume the frames.
//! * [`RepetitiveFrameDecoder`] — synthesizes `n` copies of a single image.
//!   Useful for tests and the CLI's `--video-frame` mode where the user
//!   supplies a single image as a stand-in for a real video.
//!
//! No video codec is implemented in-tree: pulling in FFmpeg or a pure-Rust
//! decoder is out of scope for the current CPU-first design. The trait
//! surface is small and decoupled so an FFmpeg-backed decoder can be added
//! later without touching the rest of the video stack.

use super::error::VideoError;

/// A single decoded RGB frame. Width and height are in pixels; data is
/// row-major, three bytes per pixel, no padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl DecodedFrame {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self, VideoError> {
        let expected = width
            .checked_mul(height)
            .and_then(|p| p.checked_mul(3))
            .ok_or_else(|| VideoError::InvalidConfig("frame dimensions overflow".into()))?;
        if data.len() != expected {
            return Err(VideoError::InvalidFrame {
                expected_bytes: expected,
                actual_bytes: data.len(),
                width,
                height,
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

/// Decodes a video into a sequence of RGB frames. Implementations are
/// expected to be deterministic: calling `decode` multiple times on the same
/// input should return the same frames in the same order.
pub trait VideoDecoder {
    /// Decode all frames from `source`. The returned vector may be empty if
    /// the source is empty; the model will then return an error.
    fn decode(&self, source: &VideoSource<'_>) -> Result<Vec<DecodedFrame>, VideoError>;
}

/// Source identifier passed to a decoder. Decoders that need an external
/// resource (file, URL, network stream) match on the variant they support
/// and ignore the rest.
#[derive(Debug, Clone, Copy)]
pub enum VideoSource<'a> {
    /// Pre-decoded RGB frames. Most decoders can use this directly.
    Frames(&'a [DecodedFrame]),
    /// A single image that should be treated as a one-frame video. Useful
    /// for image-to-video models and tests.
    SingleImage(&'a DecodedFrame),
}

/// Reference decoder that simply returns the source frames unchanged.
#[derive(Debug, Default, Clone, Copy)]
pub struct RawFrameDecoder;

impl VideoDecoder for RawFrameDecoder {
    fn decode(&self, source: &VideoSource<'_>) -> Result<Vec<DecodedFrame>, VideoError> {
        match source {
            VideoSource::Frames(frames) => Ok(frames.to_vec()),
            VideoSource::SingleImage(frame) => Ok(vec![(*frame).clone()]),
        }
    }
}

/// Decoder that repeats a single image `n` times. Used by the CLI to
/// treat an image as a 1-frame video without writing a real codec.
#[derive(Debug, Clone, Copy)]
pub struct RepetitiveFrameDecoder {
    pub repeat: usize,
}

impl Default for RepetitiveFrameDecoder {
    fn default() -> Self {
        Self { repeat: 1 }
    }
}

impl VideoDecoder for RepetitiveFrameDecoder {
    fn decode(&self, source: &VideoSource<'_>) -> Result<Vec<DecodedFrame>, VideoError> {
        let frame = match source {
            VideoSource::Frames(frames) if !frames.is_empty() => frames[0].clone(),
            VideoSource::SingleImage(frame) => (*frame).clone(),
            VideoSource::Frames(_) => {
                return Err(VideoError::DecoderFailed(
                    "RepetitiveFrameDecoder requires at least one frame".into(),
                ));
            }
        };
        if self.repeat == 0 {
            return Err(VideoError::DecoderFailed(
                "RepetitiveFrameDecoder.repeat must be >= 1".into(),
            ));
        }
        Ok((0..self.repeat).map(|_| frame.clone()).collect())
    }
}

/// Decoder that resizes every frame to a target resolution before returning
/// it. Wraps another decoder and applies a nearest-neighbor resize to keep
/// the dependency surface small.
pub struct ResizingDecoder<D: VideoDecoder> {
    pub inner: D,
    pub target_width: usize,
    pub target_height: usize,
}

impl<D: VideoDecoder> VideoDecoder for ResizingDecoder<D> {
    fn decode(&self, source: &VideoSource<'_>) -> Result<Vec<DecodedFrame>, VideoError> {
        let frames = self.inner.decode(source)?;
        let mut out = Vec::with_capacity(frames.len());
        for frame in frames {
            if frame.width == self.target_width && frame.height == self.target_height {
                out.push(frame);
                continue;
            }
            let resized = resize_rgb_nearest(
                &frame.data,
                frame.width,
                frame.height,
                self.target_width,
                self.target_height,
            );
            out.push(DecodedFrame {
                width: self.target_width,
                height: self.target_height,
                data: resized,
            });
        }
        Ok(out)
    }
}

fn resize_rgb_nearest(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0_u8; dst_w * dst_h * 3];
    for dy in 0..dst_h {
        let sy = dy * src_h / dst_h;
        for dx in 0..dst_w {
            let sx = dx * src_w / dst_w;
            let src_idx = (sy * src_w + sx) * 3;
            let dst_idx = (dy * dst_w + dx) * 3;
            dst[dst_idx..dst_idx + 3].copy_from_slice(&src[src_idx..src_idx + 3]);
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    fn black_frame(w: usize, h: usize) -> DecodedFrame {
        DecodedFrame::new(w, h, vec![0; w * h * 3]).unwrap()
    }

    #[test]
    fn raw_decoder_returns_frames_unchanged() {
        let frames = vec![black_frame(4, 4), black_frame(4, 4)];
        let out = RawFrameDecoder
            .decode(&VideoSource::Frames(&frames))
            .unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn repetitive_decoder_repeats_image_n_times() {
        let frame = black_frame(2, 2);
        let decoder = RepetitiveFrameDecoder { repeat: 5 };
        let out = decoder.decode(&VideoSource::SingleImage(&frame)).unwrap();
        assert_eq!(out.len(), 5);
        for f in &out {
            assert_eq!(f.data, frame.data);
        }
    }

    #[test]
    fn resizing_decoder_preserves_count_and_shape() {
        let frames = vec![black_frame(2, 2), black_frame(4, 4)];
        let decoder = ResizingDecoder {
            inner: RawFrameDecoder,
            target_width: 8,
            target_height: 8,
        };
        let out = decoder.decode(&VideoSource::Frames(&frames)).unwrap();
        assert_eq!(out.len(), 2);
        for f in &out {
            assert_eq!(f.width, 8);
            assert_eq!(f.height, 8);
            assert_eq!(f.data.len(), 8 * 8 * 3);
        }
    }

    #[test]
    fn frame_construction_validates_length() {
        let err = DecodedFrame::new(2, 2, vec![0; 7]).unwrap_err();
        assert!(matches!(err, VideoError::InvalidFrame { .. }));
    }
}
