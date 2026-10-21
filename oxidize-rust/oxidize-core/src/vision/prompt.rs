use super::VisionError;

#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalPrompt {
    pub text_segments: Vec<String>,
    pub image_embeddings: Vec<Vec<f32>>,
    pub image_positions: Vec<usize>,
}

impl MultimodalPrompt {
    pub fn new() -> Self {
        Self {
            text_segments: Vec::new(),
            image_embeddings: Vec::new(),
            image_positions: Vec::new(),
        }
    }

    pub fn add_text(&mut self, text: impl Into<String>) {
        self.text_segments.push(text.into());
    }

    pub fn add_image(&mut self, embeddings: Vec<f32>, position: usize) {
        self.image_embeddings.push(embeddings);
        self.image_positions.push(position);
    }

    pub fn build_sequence(
        &self,
        text_token_ids: Vec<Vec<u32>>,
        text_embedding_table: &[f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        self.try_build_sequence(
            &text_token_ids,
            text_embedding_table,
            vocab_size,
            hidden_size,
        )
        .expect("valid multimodal prompt")
    }

    pub fn try_build_sequence(
        &self,
        text_token_ids: &[Vec<u32>],
        text_embedding_table: &[f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, VisionError> {
        self.validate(
            text_token_ids,
            text_embedding_table,
            vocab_size,
            hidden_size,
        )?;

        let mut sequence = Vec::new();
        let mut image_idx = 0;
        for (segment_idx, token_ids) in text_token_ids.iter().enumerate() {
            self.append_images(&mut sequence, segment_idx, &mut image_idx);
            append_text_embeddings(
                &mut sequence,
                token_ids,
                text_embedding_table,
                vocab_size,
                hidden_size,
            )?;
        }

        self.append_images(&mut sequence, self.text_segments.len(), &mut image_idx);
        if image_idx != self.image_embeddings.len() {
            return invalid_prompt("image positions must be sorted and within text segment range");
        }

        Ok(sequence)
    }

    fn validate(
        &self,
        text_token_ids: &[Vec<u32>],
        text_embedding_table: &[f32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<(), VisionError> {
        if self.image_embeddings.len() != self.image_positions.len() {
            return invalid_prompt("image_embeddings and image_positions length mismatch");
        }
        if text_token_ids.len() != self.text_segments.len() {
            return invalid_prompt(&format!(
                "expected {} tokenized text segments, got {}",
                self.text_segments.len(),
                text_token_ids.len()
            ));
        }
        validate_embedding_table(text_embedding_table, vocab_size, hidden_size)
    }

    fn append_images(&self, sequence: &mut Vec<f32>, position: usize, image_idx: &mut usize) {
        while *image_idx < self.image_positions.len()
            && self.image_positions[*image_idx] == position
        {
            sequence.extend_from_slice(&self.image_embeddings[*image_idx]);
            *image_idx += 1;
        }
    }
}

impl Default for MultimodalPrompt {
    fn default() -> Self {
        Self::new()
    }
}

fn append_text_embeddings(
    sequence: &mut Vec<f32>,
    token_ids: &[u32],
    embedding_table: &[f32],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<(), VisionError> {
    for &token_id in token_ids {
        let token_id = token_id as usize;
        if token_id >= vocab_size {
            return invalid_prompt(&format!(
                "token id {token_id} exceeds vocab size {vocab_size}"
            ));
        }
        let start = token_id * hidden_size;
        sequence.extend_from_slice(&embedding_table[start..start + hidden_size]);
    }
    Ok(())
}

fn validate_embedding_table(
    embedding_table: &[f32],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<(), VisionError> {
    let expected = vocab_size
        .checked_mul(hidden_size)
        .ok_or_else(|| VisionError::InvalidPrompt("embedding table length overflow".into()))?;

    if embedding_table.len() == expected {
        Ok(())
    } else {
        invalid_prompt(&format!(
            "embedding table length must be {expected}, got {}",
            embedding_table.len()
        ))
    }
}

fn invalid_prompt<T>(message: &str) -> Result<T, VisionError> {
    Err(VisionError::InvalidPrompt(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_sequence_with_images_between_text_segments() {
        let mut prompt = MultimodalPrompt::new();
        prompt.add_text("Describe this image:");
        prompt.add_image(vec![1.0_f32; 6], 1);
        prompt.add_text("What do you see?");

        let text_tokens = vec![vec![1_u32, 2], vec![3_u32]];
        let embedding_table = vec![
            0.0, 0.0, 0.0, 0.0, //
            1.0, 1.0, 1.0, 1.0, //
            2.0, 2.0, 2.0, 2.0, //
            3.0, 3.0, 3.0, 3.0,
        ];
        let sequence = prompt
            .try_build_sequence(&text_tokens, &embedding_table, 4, 4)
            .unwrap();

        assert_eq!(sequence.len(), 2 * 4 + 6 + 4);
        assert_eq!(&sequence[8..14], &[1.0; 6]);
    }

    #[test]
    fn rejects_out_of_range_tokens() {
        let mut prompt = MultimodalPrompt::new();
        prompt.add_text("bad token");

        let err = prompt
            .try_build_sequence(&[vec![4_u32]], &[0.0; 16], 4, 4)
            .unwrap_err();

        assert!(matches!(err, VisionError::InvalidPrompt(_)));
    }
}
