//! Multimodal prompt helper for video inputs.
//!
//! Mirrors [`MultimodalPrompt`](crate::vision::MultimodalPrompt) but tracks
//! a per-frame video embedding matrix instead of a flat per-image
//! embedding vector. The video tokens can be interleaved with text token
//! embeddings using the same segment/position API the vision prompt uses.

use super::error::VideoError;

/// Build sequence of embeddings for a video + text prompt.
///
/// The prompt is a list of `PromptSegment`s. A `Video` segment contributes
/// `num_frames` embeddings (one row of the video matrix per frame).
/// `Text` segments contribute one embedding per token id.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoPrompt {
    pub segments: Vec<PromptSegment>,
    /// Reserved for `video_start` / `video_end` token embedding rows.
    /// When non-empty, two rows are emitted around each `Video` segment.
    pub video_start_embedding: Vec<f32>,
    pub video_end_embedding: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PromptSegment {
    Text(Vec<u32>),
    /// `embeddings` is laid out as `[num_frames, llm_hidden_size]`.
    Video {
        embeddings: Vec<f32>,
        num_frames: usize,
        llm_hidden_size: usize,
    },
}

impl VideoPrompt {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            video_start_embedding: Vec::new(),
            video_end_embedding: Vec::new(),
        }
    }

    pub fn add_text(&mut self, token_ids: Vec<u32>) {
        self.segments.push(PromptSegment::Text(token_ids));
    }

    pub fn add_video(&mut self, embeddings: Vec<f32>, num_frames: usize, llm_hidden_size: usize) {
        self.segments.push(PromptSegment::Video {
            embeddings,
            num_frames,
            llm_hidden_size,
        });
    }

    /// Flatten the prompt to a single `[seq, llm_hidden_size]` row-major
    /// embedding matrix using the supplied token embedding table for the
    /// text segments.
    pub fn build_sequence(
        &self,
        text_embedding_table: &[f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, VideoError> {
        let llm_hidden = self.infer_hidden_size(hidden_size)?;
        let total_rows = self.count_rows(hidden_size, llm_hidden)?;
        let mut out = Vec::with_capacity(total_rows * llm_hidden);

        for segment in &self.segments {
            match segment {
                PromptSegment::Text(token_ids) => {
                    for &token in token_ids {
                        let token = token as usize;
                        if token >= vocab_size {
                            return Err(VideoError::InvalidConfig(format!(
                                "token id {token} exceeds vocab size {vocab_size}"
                            )));
                        }
                        let start = token * hidden_size;
                        if start + hidden_size > text_embedding_table.len() {
                            return Err(VideoError::InvalidConfig(format!(
                                "token embedding table too small: need at least {} floats",
                                start + hidden_size
                            )));
                        }
                        if hidden_size == llm_hidden {
                            out.extend_from_slice(
                                &text_embedding_table[start..start + hidden_size],
                            );
                        } else {
                            // Pad/truncate to llm_hidden.
                            let take = hidden_size.min(llm_hidden);
                            out.extend_from_slice(&text_embedding_table[start..start + take]);
                            out.extend(std::iter::repeat(0.0_f32).take(llm_hidden - take));
                        }
                    }
                }
                PromptSegment::Video {
                    embeddings,
                    num_frames,
                    llm_hidden_size,
                } => {
                    if *llm_hidden_size != llm_hidden {
                        return Err(VideoError::InvalidConfig(format!(
                            "video llm_hidden_size {llm_hidden_size} != prompt llm_hidden {llm_hidden}"
                        )));
                    }
                    if embeddings.len() != num_frames * llm_hidden {
                        return Err(VideoError::InvalidConfig(format!(
                            "video embeddings length {} != num_frames * llm_hidden ({} * {})",
                            embeddings.len(),
                            num_frames,
                            llm_hidden
                        )));
                    }
                    if !self.video_start_embedding.is_empty() {
                        out.extend_from_slice(&self.video_start_embedding);
                    }
                    out.extend_from_slice(embeddings);
                    if !self.video_end_embedding.is_empty() {
                        out.extend_from_slice(&self.video_end_embedding);
                    }
                }
            }
        }
        Ok(out)
    }

    fn infer_hidden_size(&self, text_hidden: usize) -> Result<usize, VideoError> {
        for segment in &self.segments {
            if let PromptSegment::Video {
                llm_hidden_size, ..
            } = segment
            {
                return Ok(*llm_hidden_size);
            }
        }
        Ok(text_hidden)
    }

    fn count_rows(&self, text_hidden: usize, llm_hidden: usize) -> Result<usize, VideoError> {
        let mut total = 0usize;
        for segment in &self.segments {
            match segment {
                PromptSegment::Text(token_ids) => {
                    total = total
                        .checked_add(token_ids.len())
                        .ok_or_else(|| VideoError::InvalidConfig("row count overflow".into()))?;
                }
                PromptSegment::Video {
                    num_frames,
                    llm_hidden_size,
                    ..
                } => {
                    if *llm_hidden_size != llm_hidden {
                        return Err(VideoError::InvalidConfig(format!(
                            "video llm_hidden_size {llm_hidden_size} != prompt llm_hidden {llm_hidden}"
                        )));
                    }
                    let extra = (self.video_start_embedding.len() > 0) as usize
                        + (self.video_end_embedding.len() > 0) as usize;
                    total = total
                        .checked_add(num_frames + extra)
                        .ok_or_else(|| VideoError::InvalidConfig("row count overflow".into()))?;
                }
            }
        }
        // Suppress unused warning for `text_hidden` when there are no
        // video segments: we still validate the inference above.
        let _ = text_hidden;
        Ok(total)
    }
}

impl Default for VideoPrompt {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_prompt_uses_embedding_table() {
        let mut prompt = VideoPrompt::new();
        prompt.add_text(vec![1, 2]);
        // 2 tokens, hidden = 3
        let table = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let seq = prompt.build_sequence(&table, 3, 3).unwrap();
        assert_eq!(seq, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn text_and_video_segments_are_concatenated() {
        let mut prompt = VideoPrompt::new();
        prompt.add_text(vec![1]);
        prompt.add_video(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2, 3);
        let table = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let seq = prompt.build_sequence(&table, 2, 3).unwrap();
        assert_eq!(seq, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn start_end_wrappers_are_added_when_configured() {
        let mut prompt = VideoPrompt::new();
        prompt.video_start_embedding = vec![100.0, 100.0, 100.0];
        prompt.video_end_embedding = vec![200.0, 200.0, 200.0];
        prompt.add_video(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let table: Vec<f32> = vec![];
        let seq = prompt.build_sequence(&table, 0, 3).unwrap();
        assert_eq!(
            seq,
            vec![
                100.0, 100.0, 100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 200.0, 200.0, 200.0
            ]
        );
    }

    #[test]
    fn rejects_token_out_of_range() {
        let mut prompt = VideoPrompt::new();
        prompt.add_text(vec![10]);
        let err = prompt.build_sequence(&vec![0.0; 6], 3, 2).unwrap_err();
        assert!(matches!(err, VideoError::InvalidConfig(_)));
    }
}
