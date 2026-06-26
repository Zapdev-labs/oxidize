use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{
    AdamW, Linear, Matrix, argmax, cross_entropy_with_logits, relu_backward_in_place, relu_in_place,
};

/// A compact, fully-trainable video classifier:
/// learned patch embedding → ReLU → mean-pool patches within each frame →
/// concatenate the per-frame embeddings (keeping temporal structure) →
/// 2-layer MLP head. Small enough to train on CPU, expressive enough to learn
/// real structure from short clips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoClassifier {
    patch_dim: usize,
    embed_dim: usize,
    hidden_size: usize,
    classes: usize,
    num_frames: usize,
    patches_per_frame: usize,
    patch_embed: Linear,
    head1: Linear,
    head2: Linear,
}

struct ForwardCache {
    patch_pre: Matrix,
    clip_feat: Matrix,
    hidden_pre: Matrix,
    hidden_act: Matrix,
    logits: Matrix,
}

impl VideoClassifier {
    pub fn new(
        patch_dim: usize,
        embed_dim: usize,
        hidden_size: usize,
        classes: usize,
        num_frames: usize,
        tokens_per_clip: usize,
        seed: u64,
    ) -> Self {
        let num_frames = num_frames.max(1);
        let patches_per_frame = (tokens_per_clip / num_frames).max(1);
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            patch_dim,
            embed_dim,
            hidden_size,
            classes,
            num_frames,
            patches_per_frame,
            patch_embed: Linear::new(patch_dim, embed_dim, &mut rng),
            head1: Linear::new(num_frames * embed_dim, hidden_size, &mut rng),
            head2: Linear::new(hidden_size, classes, &mut rng),
        }
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    pub fn patch_dim(&self) -> usize {
        self.patch_dim
    }

    pub fn tokens_per_clip(&self) -> usize {
        self.num_frames * self.patches_per_frame
    }

    fn forward(&self, input: &Matrix) -> ForwardCache {
        let rows = input.rows();
        let batch = rows / self.tokens_per_clip();

        let mut patch_pre = Matrix::zeros(rows, self.embed_dim);
        self.patch_embed.forward(input, &mut patch_pre);
        let mut patch_act = patch_pre.clone();
        relu_in_place(patch_act.data_mut());

        // Mean-pool patches within each frame: [B*F*P, D] → [B*F, D], then view
        // as [B, F*D] so the head sees every frame's embedding side by side.
        let frame_emb = mean_pool(&patch_act, self.patches_per_frame);
        let clip_feat = reshape(frame_emb, batch, self.num_frames * self.embed_dim);

        let mut hidden_pre = Matrix::zeros(batch, self.hidden_size);
        self.head1.forward(&clip_feat, &mut hidden_pre);
        let mut hidden_act = hidden_pre.clone();
        relu_in_place(hidden_act.data_mut());

        let mut logits = Matrix::zeros(batch, self.classes);
        self.head2.forward(&hidden_act, &mut logits);

        ForwardCache {
            patch_pre,
            clip_feat,
            hidden_pre,
            hidden_act,
            logits,
        }
    }

    /// One optimization step over a batch; returns the mean cross-entropy loss.
    pub(crate) fn train_step(
        &mut self,
        input: &Matrix,
        labels: &[usize],
        optimizer: &mut AdamW,
    ) -> f32 {
        let batch = labels.len();
        let cache = self.forward(input);

        let mut logits_grad = Matrix::zeros(batch, self.classes);
        let loss = cross_entropy_with_logits(&cache.logits, labels, &mut logits_grad);

        self.patch_embed.zero_grad();
        self.head1.zero_grad();
        self.head2.zero_grad();

        let mut hidden_grad = Matrix::zeros(batch, self.hidden_size);
        self.head2
            .backward(&cache.hidden_act, &logits_grad, Some(&mut hidden_grad));
        relu_backward_in_place(&cache.hidden_pre, &mut hidden_grad);

        let mut clip_feat_grad = Matrix::zeros(batch, self.num_frames * self.embed_dim);
        self.head1
            .backward(&cache.clip_feat, &hidden_grad, Some(&mut clip_feat_grad));

        let frame_grad = reshape(clip_feat_grad, batch * self.num_frames, self.embed_dim);
        let mut patch_grad = unpool(&frame_grad, self.patches_per_frame);
        relu_backward_in_place(&cache.patch_pre, &mut patch_grad);
        self.patch_embed.backward(input, &patch_grad, None);

        optimizer.next_step();
        optimizer.step(&mut self.patch_embed);
        optimizer.step(&mut self.head1);
        optimizer.step(&mut self.head2);

        loss
    }

    pub fn predict(&self, input: &Matrix) -> Vec<usize> {
        let cache = self.forward(input);
        cache
            .logits
            .data()
            .chunks_exact(self.classes)
            .map(argmax)
            .collect()
    }

    /// Mean cross-entropy loss over a batch without updating weights.
    pub fn loss(&self, input: &Matrix, labels: &[usize]) -> f32 {
        let cache = self.forward(input);
        let mut grad = Matrix::zeros(labels.len(), self.classes);
        cross_entropy_with_logits(&cache.logits, labels, &mut grad)
    }
}

/// Average the `tokens_per_clip` rows belonging to each clip: [B*T, D] → [B, D].
fn mean_pool(input: &Matrix, tokens_per_clip: usize) -> Matrix {
    let dim = input.cols();
    let batch = input.rows() / tokens_per_clip;
    let mut pooled = Matrix::zeros(batch, dim);
    let src = input.data();
    let dst = pooled.data_mut();
    let inv = 1.0 / tokens_per_clip as f32;
    for clip in 0..batch {
        let out = &mut dst[clip * dim..(clip + 1) * dim];
        for token in 0..tokens_per_clip {
            let row = &src[(clip * tokens_per_clip + token) * dim..][..dim];
            for (o, &value) in out.iter_mut().zip(row) {
                *o += value;
            }
        }
        for o in out.iter_mut() {
            *o *= inv;
        }
    }
    pooled
}

/// Backward of `mean_pool`: spread each clip's gradient equally across its tokens.
fn unpool(pooled_grad: &Matrix, tokens_per_clip: usize) -> Matrix {
    let dim = pooled_grad.cols();
    let batch = pooled_grad.rows();
    let mut grad = Matrix::zeros(batch * tokens_per_clip, dim);
    let src = pooled_grad.data();
    let dst = grad.data_mut();
    let inv = 1.0 / tokens_per_clip as f32;
    for clip in 0..batch {
        let pooled_row = &src[clip * dim..(clip + 1) * dim];
        for token in 0..tokens_per_clip {
            let row = &mut dst[(clip * tokens_per_clip + token) * dim..][..dim];
            for (g, &value) in row.iter_mut().zip(pooled_row) {
                *g = value * inv;
            }
        }
    }
    grad
}

/// Reinterpret a matrix's contiguous data under a new (rows, cols) shape.
fn reshape(m: Matrix, rows: usize, cols: usize) -> Matrix {
    Matrix::from_vec(rows, cols, m.data().to_vec()).unwrap_or_else(|_| Matrix::zeros(rows, cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_then_unpool_preserves_totals() {
        let input =
            Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("matrix");
        let pooled = mean_pool(&input, 2);
        assert_eq!(pooled.rows(), 2);
        // clip 0 rows (1,2),(3,4) -> mean (2,3); clip 1 (5,6),(7,8) -> (6,7)
        assert_eq!(pooled.data(), &[2.0, 3.0, 6.0, 7.0]);

        let grad = unpool(&pooled, 2);
        assert_eq!(grad.rows(), 4);
        // each token receives pooled/tokens
        assert_eq!(grad.data(), &[1.0, 1.5, 1.0, 1.5, 3.0, 3.5, 3.0, 3.5]);
    }

    #[test]
    fn predict_returns_one_label_per_clip() {
        let tokens = 3;
        let patch_dim = 4;
        let model = VideoClassifier::new(patch_dim, 5, 6, 3, 1, tokens, 1);
        let batch = 2;
        let input = Matrix::zeros(batch * tokens, patch_dim);
        let preds = model.predict(&input);
        assert_eq!(preds.len(), batch);
        assert!(preds.iter().all(|&p| p < 3));
    }

    #[test]
    fn train_step_reduces_loss_on_separable_clips() {
        // Two classes: class 0 clips are all -1, class 1 clips are all +1.
        let tokens = 2;
        let patch_dim = 3;
        let mut model = VideoClassifier::new(patch_dim, 8, 8, 2, 2, tokens, 7);
        let mut opt = AdamW::new(0.05, 0.0);

        let mut data = Vec::new();
        let mut labels = Vec::new();
        for class in 0..2 {
            for _ in 0..8 {
                let value = if class == 0 { -1.0 } else { 1.0 };
                for _ in 0..(tokens * patch_dim) {
                    data.push(value);
                }
                labels.push(class);
            }
        }
        let input = Matrix::from_vec(labels.len() * tokens, patch_dim, data).expect("matrix");

        let first = model.loss(&input, &labels);
        for _ in 0..50 {
            model.train_step(&input, &labels, &mut opt);
        }
        let last = model.loss(&input, &labels);
        assert!(last < first, "expected {last} < {first}");
    }
}
