use std::{error::Error, fmt, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub mod video;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows.saturating_mul(cols)],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("matrix shape overflows usize: {rows}x{cols}"))?;
        if data.len() != expected {
            bail!(
                "invalid matrix data length: expected {expected} values for {rows}x{cols}, got {}",
                data.len()
            );
        }
        Ok(Self { rows, cols, data })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    inputs: Matrix,
    labels: Vec<usize>,
    classes: usize,
}

impl Dataset {
    pub fn new(inputs: Matrix, labels: Vec<usize>, classes: usize) -> TrainResult<Self> {
        if inputs.rows == 0 {
            return Err(TrainingError::EmptyDataset);
        }
        if inputs.rows != labels.len() {
            return Err(TrainingError::LabelCountMismatch {
                rows: inputs.rows,
                labels: labels.len(),
            });
        }
        if classes == 0 {
            return Err(TrainingError::InvalidClassCount { classes });
        }
        if let Some((index, label)) = labels
            .iter()
            .copied()
            .enumerate()
            .find(|(_, label)| *label >= classes)
        {
            return Err(TrainingError::LabelOutOfRange {
                index,
                label,
                classes,
            });
        }
        Ok(Self {
            inputs,
            labels,
            classes,
        })
    }

    pub fn inputs(&self) -> &Matrix {
        &self.inputs
    }

    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    pub fn classes(&self) -> usize {
        self.classes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub hidden_size: usize,
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch_size: 32,
            learning_rate: 1e-3,
            weight_decay: 0.01,
            hidden_size: 128,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingReport {
    pub final_loss: f32,
    pub accuracy: f32,
    pub samples: usize,
    pub features: usize,
    pub classes: usize,
    pub epoch_losses: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrainingError {
    EmptyDataset,
    InvalidBatchSize {
        batch_size: usize,
    },
    InvalidHiddenSize {
        hidden_size: usize,
    },
    InvalidClassCount {
        classes: usize,
    },
    LabelCountMismatch {
        rows: usize,
        labels: usize,
    },
    LabelOutOfRange {
        index: usize,
        label: usize,
        classes: usize,
    },
}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => write!(f, "dataset is empty"),
            Self::InvalidBatchSize { batch_size } => {
                write!(f, "batch_size must be greater than zero, got {batch_size}")
            }
            Self::InvalidHiddenSize { hidden_size } => {
                write!(
                    f,
                    "hidden_size must be greater than zero, got {hidden_size}"
                )
            }
            Self::InvalidClassCount { classes } => {
                write!(f, "classes must be greater than zero, got {classes}")
            }
            Self::LabelCountMismatch { rows, labels } => {
                write!(f, "input rows ({rows}) do not match label count ({labels})")
            }
            Self::LabelOutOfRange {
                index,
                label,
                classes,
            } => write!(
                f,
                "label at index {index} is {label}, but class count is {classes}"
            ),
        }
    }
}

impl Error for TrainingError {}

pub type TrainResult<T> = std::result::Result<T, TrainingError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlpClassifier {
    input_size: usize,
    hidden_size: usize,
    classes: usize,
    l1: Linear,
    l2: Linear,
}

impl MlpClassifier {
    pub fn new(input_size: usize, hidden_size: usize, classes: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            input_size,
            hidden_size,
            classes,
            l1: Linear::new(input_size, hidden_size, &mut rng),
            l2: Linear::new(hidden_size, classes, &mut rng),
        }
    }

    pub fn predict(&self, inputs: &Matrix) -> Vec<usize> {
        let (_, _, logits) = self.forward(inputs);
        logits
            .data
            .par_chunks(self.classes)
            .map(argmax)
            .collect::<Vec<_>>()
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    fn forward(&self, inputs: &Matrix) -> (Matrix, Matrix, Matrix) {
        let mut hidden_pre = Matrix::zeros(inputs.rows, self.hidden_size);
        self.l1.forward(inputs, &mut hidden_pre);
        let mut hidden = hidden_pre.clone();
        relu_in_place(&mut hidden.data);
        let mut logits = Matrix::zeros(inputs.rows, self.classes);
        self.l2.forward(&hidden, &mut logits);
        (hidden_pre, hidden, logits)
    }

    fn backward(
        &mut self,
        inputs: &Matrix,
        hidden_pre: &Matrix,
        hidden: &Matrix,
        logits_grad: &Matrix,
    ) {
        let mut hidden_grad = Matrix::zeros(inputs.rows, self.hidden_size);
        self.l2
            .backward(hidden, logits_grad, Some(&mut hidden_grad));
        relu_backward_in_place(hidden_pre, &mut hidden_grad);
        self.l1.backward(inputs, &hidden_grad, None);
    }

    fn zero_grad(&mut self) {
        self.l1.zero_grad();
        self.l2.zero_grad();
    }

    fn step(&mut self, optimizer: &mut AdamW) {
        optimizer.step(&mut self.l1);
        optimizer.step(&mut self.l2);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Linear {
    weights: Matrix,
    bias: Vec<f32>,
    // Transient training state — never persisted. A deserialized layer is
    // inference-only until these are re-initialized.
    #[serde(skip)]
    grad_weights: Matrix,
    #[serde(skip)]
    grad_bias: Vec<f32>,
    #[serde(skip)]
    adam_w_m: Vec<f32>,
    #[serde(skip)]
    adam_w_v: Vec<f32>,
    #[serde(skip)]
    adam_b_m: Vec<f32>,
    #[serde(skip)]
    adam_b_v: Vec<f32>,
}

impl Linear {
    fn new(input: usize, output: usize, rng: &mut StdRng) -> Self {
        let scale = (2.0 / (input + output).max(1) as f32).sqrt();
        let mut weights = vec![0.0; input * output];
        for value in &mut weights {
            *value = (rand::Rng::r#gen::<f32>(rng) * 2.0 - 1.0) * scale;
        }
        Self {
            weights: Matrix {
                rows: input,
                cols: output,
                data: weights,
            },
            bias: vec![0.0; output],
            grad_weights: Matrix::zeros(input, output),
            grad_bias: vec![0.0; output],
            adam_w_m: vec![0.0; input * output],
            adam_w_v: vec![0.0; input * output],
            adam_b_m: vec![0.0; output],
            adam_b_v: vec![0.0; output],
        }
    }

    fn forward(&self, input: &Matrix, output: &mut Matrix) {
        let in_features = input.cols;
        let out_features = self.bias.len();
        output
            .data
            .par_chunks_mut(out_features)
            .enumerate()
            .for_each(|(row, output_row)| {
                let input_row = &input.data[row * in_features..(row + 1) * in_features];
                for (out_col, output_value) in output_row.iter_mut().enumerate() {
                    let mut sum = self.bias[out_col];
                    for (in_col, input_value) in input_row.iter().copied().enumerate() {
                        sum += input_value * self.weights.data[in_col * out_features + out_col];
                    }
                    *output_value = sum;
                }
            });
    }

    fn backward(
        &mut self,
        input: &Matrix,
        grad_output: &Matrix,
        mut grad_input: Option<&mut Matrix>,
    ) {
        let in_features = input.cols;
        let out_features = grad_output.cols;
        let batch = input.rows;

        self.grad_weights
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, grad)| {
                let in_col = idx / out_features;
                let out_col = idx % out_features;
                let mut sum = 0.0;
                for row in 0..batch {
                    sum += input.data[row * in_features + in_col]
                        * grad_output.data[row * out_features + out_col];
                }
                *grad = sum;
            });

        self.grad_bias
            .par_iter_mut()
            .enumerate()
            .for_each(|(out_col, grad)| {
                let mut sum = 0.0;
                for row in 0..batch {
                    sum += grad_output.data[row * out_features + out_col];
                }
                *grad = sum;
            });

        if let Some(grad_input) = grad_input.as_mut() {
            grad_input
                .data
                .par_chunks_mut(in_features)
                .enumerate()
                .for_each(|(row, grad_row)| {
                    for (in_col, grad_value) in grad_row.iter_mut().enumerate() {
                        let mut sum = 0.0;
                        for out_col in 0..out_features {
                            sum += grad_output.data[row * out_features + out_col]
                                * self.weights.data[in_col * out_features + out_col];
                        }
                        *grad_value = sum;
                    }
                });
        }
    }

    fn zero_grad(&mut self) {
        self.grad_weights.data.fill(0.0);
        self.grad_bias.fill(0.0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct AdamW {
    learning_rate: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
}

impl AdamW {
    fn new(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step: 0,
        }
    }

    fn next_step(&mut self) {
        self.step = self.step.saturating_add(1);
    }

    fn step(&self, layer: &mut Linear) {
        adamw_update(
            &mut layer.weights.data,
            &layer.grad_weights.data,
            &mut layer.adam_w_m,
            &mut layer.adam_w_v,
            *self,
            true,
        );
        adamw_update(
            &mut layer.bias,
            &layer.grad_bias,
            &mut layer.adam_b_m,
            &mut layer.adam_b_v,
            *self,
            false,
        );
    }
}

pub fn train_classifier(
    dataset: &Dataset,
    config: TrainingConfig,
) -> TrainResult<(MlpClassifier, TrainingReport)> {
    if config.batch_size == 0 {
        return Err(TrainingError::InvalidBatchSize {
            batch_size: config.batch_size,
        });
    }
    if config.hidden_size == 0 {
        return Err(TrainingError::InvalidHiddenSize {
            hidden_size: config.hidden_size,
        });
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut model = MlpClassifier::new(
        dataset.inputs.cols,
        config.hidden_size,
        dataset.classes,
        config.seed,
    );
    let mut optimizer = AdamW::new(config.learning_rate, config.weight_decay);
    let mut indices = (0..dataset.inputs.rows).collect::<Vec<_>>();
    let mut epoch_losses = Vec::with_capacity(config.epochs);

    for _ in 0..config.epochs {
        indices.shuffle(&mut rng);
        let mut weighted_loss = 0.0;
        let mut seen = 0usize;

        for batch_indices in indices.chunks(config.batch_size) {
            let batch = gather_batch(dataset, batch_indices);
            model.zero_grad();
            let (hidden_pre, hidden, logits) = model.forward(&batch.inputs);
            let mut logits_grad = Matrix::zeros(batch.inputs.rows, dataset.classes);
            let loss = cross_entropy_with_logits(&logits, &batch.labels, &mut logits_grad);
            model.backward(&batch.inputs, &hidden_pre, &hidden, &logits_grad);
            optimizer.next_step();
            model.step(&mut optimizer);
            weighted_loss += loss * batch.inputs.rows as f32;
            seen += batch.inputs.rows;
        }

        epoch_losses.push(weighted_loss / seen.max(1) as f32);
    }

    let (final_loss, accuracy) = evaluate_classifier(&model, dataset);
    Ok((
        model,
        TrainingReport {
            final_loss,
            accuracy,
            samples: dataset.inputs.rows,
            features: dataset.inputs.cols,
            classes: dataset.classes,
            epoch_losses,
        },
    ))
}

pub fn evaluate_classifier(model: &MlpClassifier, dataset: &Dataset) -> (f32, f32) {
    let (_, _, logits) = model.forward(&dataset.inputs);
    let mut grad = Matrix::zeros(dataset.inputs.rows, dataset.classes);
    let loss = cross_entropy_with_logits(&logits, &dataset.labels, &mut grad);
    let predictions = model.predict(&dataset.inputs);
    let correct = predictions
        .iter()
        .zip(dataset.labels.iter())
        .filter(|(predicted, expected)| predicted == expected)
        .count();
    (loss, correct as f32 / dataset.labels.len().max(1) as f32)
}

pub fn load_csv_dataset(path: impl AsRef<Path>, label_column: Option<usize>) -> Result<Dataset> {
    let text = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("failed to read {}", path.as_ref().display()))?;
    let mut rows = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cells = line.split(',').map(str::trim).collect::<Vec<_>>();
        let label_col = label_column.unwrap_or(cells.len().saturating_sub(1));
        if label_col >= cells.len() {
            bail!(
                "label column {label_col} is out of bounds for line {} with {} columns",
                line_no + 1,
                cells.len()
            );
        }

        let parsed = parse_csv_row(&cells, label_col);
        match parsed {
            Ok(row) => rows.push(row),
            Err(error) if line_no == 0 => {
                tracing_like_header_skip(&error);
            }
            Err(error) => {
                return Err(error).with_context(|| format!("invalid CSV line {}", line_no + 1));
            }
        }
    }

    if rows.is_empty() {
        bail!("CSV dataset has no data rows");
    }
    let feature_count = rows[0].0.len();
    if rows
        .iter()
        .any(|(features, _)| features.len() != feature_count)
    {
        bail!("CSV rows have inconsistent feature counts");
    }
    let classes = rows.iter().map(|(_, label)| *label).max().unwrap_or(0) + 1;
    let mut data = Vec::with_capacity(rows.len() * feature_count);
    let mut labels = Vec::with_capacity(rows.len());
    for (features, label) in rows {
        data.extend(features);
        labels.push(label);
    }
    let inputs = Matrix::from_vec(labels.len(), feature_count, data)?;
    Dataset::new(inputs, labels, classes).map_err(|error| anyhow!(error))
}

fn tracing_like_header_skip(_error: &anyhow::Error) {}

fn parse_csv_row(cells: &[&str], label_col: usize) -> Result<(Vec<f32>, usize)> {
    let mut features = Vec::with_capacity(cells.len().saturating_sub(1));
    let mut label = None;
    for (idx, cell) in cells.iter().enumerate() {
        if idx == label_col {
            label = Some(
                cell.parse::<usize>()
                    .with_context(|| format!("failed to parse label '{cell}' as usize"))?,
            );
        } else {
            features.push(
                cell.parse::<f32>()
                    .with_context(|| format!("failed to parse feature '{cell}' as f32"))?,
            );
        }
    }
    Ok((features, label.expect("label column already checked")))
}

fn gather_batch(dataset: &Dataset, indices: &[usize]) -> Dataset {
    let feature_count = dataset.inputs.cols;
    let mut inputs = Vec::with_capacity(indices.len() * feature_count);
    let mut labels = Vec::with_capacity(indices.len());
    for &idx in indices {
        let start = idx * feature_count;
        inputs.extend_from_slice(&dataset.inputs.data[start..start + feature_count]);
        labels.push(dataset.labels[idx]);
    }
    Dataset {
        inputs: Matrix {
            rows: indices.len(),
            cols: feature_count,
            data: inputs,
        },
        labels,
        classes: dataset.classes,
    }
}

fn relu_in_place(values: &mut [f32]) {
    values.par_iter_mut().for_each(|value| {
        if *value < 0.0 {
            *value = 0.0;
        }
    });
}

fn relu_backward_in_place(pre_activation: &Matrix, grad: &mut Matrix) {
    grad.data
        .par_iter_mut()
        .zip(pre_activation.data.par_iter())
        .for_each(|(grad, activation)| {
            if *activation <= 0.0 {
                *grad = 0.0;
            }
        });
}

fn cross_entropy_with_logits(logits: &Matrix, labels: &[usize], grad: &mut Matrix) -> f32 {
    let classes = logits.cols;
    let inv_batch = 1.0 / logits.rows.max(1) as f32;
    let loss_sum: f32 = grad
        .data
        .par_chunks_mut(classes)
        .zip(logits.data.par_chunks(classes))
        .zip(labels.par_iter())
        .map(|((grad_row, logits_row), label)| {
            let max_logit = logits_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum = logits_row
                .iter()
                .map(|value| (*value - max_logit).exp())
                .sum::<f32>();
            let log_sum_exp = max_logit + exp_sum.ln();
            for (idx, grad) in grad_row.iter_mut().enumerate() {
                let probability = (logits_row[idx] - log_sum_exp).exp();
                *grad = (probability - usize::from(idx == *label) as f32) * inv_batch;
            }
            log_sum_exp - logits_row[*label]
        })
        .sum();
    loss_sum * inv_batch
}

fn adamw_update(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    optimizer: AdamW,
    apply_weight_decay: bool,
) {
    let bias_correction1 = 1.0 - optimizer.beta1.powi(optimizer.step as i32);
    let bias_correction2 = 1.0 - optimizer.beta2.powi(optimizer.step as i32);
    params
        .par_iter_mut()
        .zip(grads.par_iter())
        .zip(m.par_iter_mut())
        .zip(v.par_iter_mut())
        .for_each(|(((param, grad), m), v)| {
            if apply_weight_decay {
                *param *= 1.0 - optimizer.learning_rate * optimizer.weight_decay;
            }
            *m = optimizer.beta1 * *m + (1.0 - optimizer.beta1) * *grad;
            *v = optimizer.beta2 * *v + (1.0 - optimizer.beta2) * *grad * *grad;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            *param -= optimizer.learning_rate * m_hat / (v_hat.sqrt() + optimizer.eps);
        });
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(idx, _)| idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linearly_separable_dataset() -> Dataset {
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for x in -5..=5 {
            for y in -5..=5 {
                let xf = x as f32 / 5.0;
                let yf = y as f32 / 5.0;
                data.push(xf);
                data.push(yf);
                labels.push(usize::from(xf + yf > 0.0));
            }
        }
        let inputs = Matrix::from_vec(labels.len(), 2, data).expect("valid matrix");
        Dataset::new(inputs, labels, 2).expect("valid dataset")
    }

    #[test]
    fn trains_classifier_to_reduce_loss() {
        let dataset = linearly_separable_dataset();
        let initial = MlpClassifier::new(2, 8, 2, 7);
        let (initial_loss, _) = evaluate_classifier(&initial, &dataset);
        let (_, report) = train_classifier(
            &dataset,
            TrainingConfig {
                epochs: 50,
                batch_size: 16,
                learning_rate: 0.03,
                weight_decay: 0.0,
                hidden_size: 8,
                seed: 7,
            },
        )
        .expect("training succeeds");

        assert!(report.final_loss < initial_loss);
        assert!(report.accuracy > 0.85);
        assert_eq!(report.epoch_losses.len(), 50);
    }

    #[test]
    fn rejects_invalid_dataset_shapes() {
        let inputs = Matrix::from_vec(2, 2, vec![0.0; 4]).expect("valid matrix");
        let error = Dataset::new(inputs, vec![0], 2).expect_err("mismatched labels should fail");
        assert!(matches!(
            error,
            TrainingError::LabelCountMismatch { rows: 2, labels: 1 }
        ));
    }

    #[test]
    fn loads_csv_dataset_with_header() {
        let path =
            std::env::temp_dir().join(format!("oxidize-train-test-{}.csv", std::process::id()));
        std::fs::write(&path, "x,y,label\n0.0,0.0,0\n1.0,1.0,1\n").expect("write csv");
        let dataset = load_csv_dataset(&path, None).expect("load csv");
        let _ = std::fs::remove_file(&path);

        assert_eq!(dataset.inputs.rows(), 2);
        assert_eq!(dataset.inputs.cols(), 2);
        assert_eq!(dataset.labels(), &[0, 1]);
        assert_eq!(dataset.classes(), 2);
    }
}
