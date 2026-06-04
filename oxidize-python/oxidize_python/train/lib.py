"""CSV classifier training mirroring oxidize-train."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


class TrainingError(Exception):
    pass


@dataclass
class Matrix:
    rows: int
    cols: int
    data: list[float]

    @classmethod
    def zeros(cls, rows: int, cols: int) -> Matrix:
        return cls(rows, cols, [0.0] * (rows * cols))

    @classmethod
    def from_vec(cls, rows: int, cols: int, data: list[float]) -> Matrix:
        expected = rows * cols
        if len(data) != expected:
            raise TrainingError(f"invalid matrix data length: expected {expected}, got {len(data)}")
        return cls(rows, cols, list(data))


@dataclass
class Dataset:
    inputs: Matrix
    labels: list[int]
    classes: int

    @classmethod
    def new(cls, inputs: Matrix, labels: list[int], classes: int) -> Dataset:
        if inputs.rows == 0:
            raise TrainingError("dataset is empty")
        if inputs.rows != len(labels):
            raise TrainingError("label count mismatch")
        if classes == 0:
            raise TrainingError("invalid class count")
        for i, label in enumerate(labels):
            if label >= classes:
                raise TrainingError(f"label {label} out of range at index {i}")
        return cls(inputs, labels, classes)


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    hidden_size: int = 128
    seed: int = 42


@dataclass
class TrainingReport:
    final_loss: float
    accuracy: float
    samples: int
    features: int
    classes: int
    epoch_losses: list[float] = field(default_factory=list)


@dataclass
class Linear:
    weights: Matrix
    bias: list[float]
    grad_weights: Matrix
    grad_bias: list[float]

    @classmethod
    def new(cls, in_features: int, out_features: int, rng: random.Random) -> Linear:
        scale = math.sqrt(2.0 / in_features)
        w = [rng.gauss(0, scale) for _ in range(in_features * out_features)]
        return cls(
            weights=Matrix.from_vec(out_features, in_features, w),
            bias=[0.0] * out_features,
            grad_weights=Matrix.zeros(out_features, in_features),
            grad_bias=[0.0] * out_features,
        )

    def forward(self, inputs: Matrix, output: Matrix) -> None:
        for r in range(inputs.rows):
            for o in range(output.cols):
                s = self.bias[o]
                for c in range(inputs.cols):
                    s += (
                        inputs.data[r * inputs.cols + c]
                        * self.weights.data[o * self.weights.cols + c]
                    )
                output.data[r * output.cols + o] = s

    def backward(
        self,
        inputs: Matrix,
        output_grad: Matrix,
        input_grad: Matrix | None,
    ) -> None:
        for r in range(inputs.rows):
            for o in range(output_grad.cols):
                g = output_grad.data[r * output_grad.cols + o]
                self.grad_bias[o] += g
                for c in range(inputs.cols):
                    idx_w = o * self.weights.cols + c
                    in_idx = r * inputs.cols + c
                    self.grad_weights.data[idx_w] += g * inputs.data[in_idx]
                    if input_grad is not None:
                        input_grad.data[in_idx] += g * self.weights.data[idx_w]

    def zero_grad(self) -> None:
        self.grad_weights = Matrix.zeros(self.weights.rows, self.weights.cols)
        self.grad_bias = [0.0] * len(self.bias)


@dataclass
class AdamW:
    lr: float
    weight_decay: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    step_count: int = 0
    m_w: list[float] = field(default_factory=list)
    v_w: list[float] = field(default_factory=list)
    m_b: list[float] = field(default_factory=list)
    v_b: list[float] = field(default_factory=list)

    def step(self, layer: Linear) -> None:
        self.step_count += 1
        t = self.step_count
        if not self.m_w:
            n = len(layer.weights.data)
            self.m_w = [0.0] * n
            self.v_w = [0.0] * n
            self.m_b = [0.0] * len(layer.bias)
            self.v_b = [0.0] * len(layer.bias)
        for i, w in enumerate(layer.weights.data):
            g = layer.grad_weights.data[i] + self.weight_decay * w
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * g
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * g * g
            m_hat = self.m_w[i] / (1 - self.beta1**t)
            v_hat = self.v_w[i] / (1 - self.beta2**t)
            layer.weights.data[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        for i, b in enumerate(layer.bias):
            g = layer.grad_bias[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * g
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * g * g
            m_hat = self.m_b[i] / (1 - self.beta1**t)
            v_hat = self.v_b[i] / (1 - self.beta2**t)
            layer.bias[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


@dataclass
class MlpClassifier:
    input_size: int
    hidden_size: int
    classes: int
    l1: Linear
    l2: Linear

    @classmethod
    def new(cls, input_size: int, hidden_size: int, classes: int, seed: int) -> MlpClassifier:
        rng = random.Random(seed)
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            classes=classes,
            l1=Linear.new(input_size, hidden_size, rng),
            l2=Linear.new(hidden_size, classes, rng),
        )

    def predict(self, inputs: Matrix) -> list[int]:
        _, _, logits = self._forward(inputs)
        return [
            _argmax(logits.data[r * self.classes : (r + 1) * self.classes])
            for r in range(inputs.rows)
        ]

    def _forward(self, inputs: Matrix) -> tuple[Matrix, Matrix, Matrix]:
        hidden_pre = Matrix.zeros(inputs.rows, self.hidden_size)
        self.l1.forward(inputs, hidden_pre)
        hidden = Matrix.from_vec(inputs.rows, self.hidden_size, list(hidden_pre.data))
        _relu_inplace(hidden.data)
        logits = Matrix.zeros(inputs.rows, self.classes)
        self.l2.forward(hidden, logits)
        return hidden_pre, hidden, logits

    def _backward(
        self,
        inputs: Matrix,
        hidden_pre: Matrix,
        hidden: Matrix,
        logits_grad: Matrix,
    ) -> None:
        hidden_grad = Matrix.zeros(inputs.rows, self.hidden_size)
        self.l2.backward(hidden, logits_grad, hidden_grad)
        _relu_backward(hidden_pre.data, hidden_grad.data)
        self.l1.backward(inputs, hidden_grad, None)

    def train(self, dataset: Dataset, config: TrainingConfig) -> TrainingReport:
        if config.batch_size <= 0:
            raise TrainingError("invalid batch size")
        if config.hidden_size <= 0:
            raise TrainingError("invalid hidden size")
        optimizer = AdamW(config.learning_rate, config.weight_decay)
        epoch_losses: list[float] = []
        rng = random.Random(config.seed)
        indices = list(range(dataset.inputs.rows))
        for _ in range(config.epochs):
            rng.shuffle(indices)
            total_loss = 0.0
            batches = 0
            for start in range(0, len(indices), config.batch_size):
                batch_idx = indices[start : start + config.batch_size]
                batch = _subset_rows(dataset.inputs, batch_idx)
                labels = [dataset.labels[i] for i in batch_idx]
                hidden_pre, hidden, logits = self._forward(batch)
                loss, grad = _softmax_cross_entropy(logits, labels, self.classes)
                total_loss += loss
                batches += 1
                self.l2.zero_grad()
                self.l1.zero_grad()
                self._backward(batch, hidden_pre, hidden, grad)
                optimizer.step(self.l1)
                optimizer.step(self.l2)
            epoch_losses.append(total_loss / max(batches, 1))
        preds = self.predict(dataset.inputs)
        correct = sum(1 for p, y in zip(preds, dataset.labels) if p == y)
        acc = correct / len(dataset.labels) if dataset.labels else 0.0
        return TrainingReport(
            final_loss=epoch_losses[-1] if epoch_losses else 0.0,
            accuracy=acc,
            samples=dataset.inputs.rows,
            features=dataset.inputs.cols,
            classes=dataset.classes,
            epoch_losses=epoch_losses,
        )


def load_csv_dataset(path: Path, label_column: str = "label") -> Dataset:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise TrainingError("empty csv")
        features = [f for f in reader.fieldnames if f != label_column]
        rows_data: list[list[float]] = []
        labels: list[int] = []
        class_set: set[int] = set()
        for row in reader:
            rows_data.append([float(row[f]) for f in features])
            lab = int(row[label_column])
            labels.append(lab)
            class_set.add(lab)
        if not rows_data:
            raise TrainingError("empty dataset")
        flat = [v for row in rows_data for v in row]
        inputs = Matrix.from_vec(len(rows_data), len(features), flat)
        classes = max(class_set) + 1
        return Dataset.new(inputs, labels, classes)


def save_report(path: Path, report: TrainingReport) -> None:
    path.write_text(
        json.dumps(
            {
                "final_loss": report.final_loss,
                "accuracy": report.accuracy,
                "samples": report.samples,
                "features": report.features,
                "classes": report.classes,
                "epoch_losses": report.epoch_losses,
            },
            indent=2,
        )
    )


def _relu_inplace(data: list[float]) -> None:
    for i, v in enumerate(data):
        if v < 0:
            data[i] = 0.0


def _relu_backward(pre: list[float], grad: list[float]) -> None:
    for i, p in enumerate(pre):
        if p <= 0:
            grad[i] = 0.0


def _argmax(row: list[float]) -> int:
    return max(range(len(row)), key=lambda i: row[i])


def _subset_rows(matrix: Matrix, indices: list[int]) -> Matrix:
    cols = matrix.cols
    data: list[float] = []
    for r in indices:
        base = r * cols
        data.extend(matrix.data[base : base + cols])
    return Matrix(len(indices), cols, data)


def _softmax_cross_entropy(logits: Matrix, labels: list[int], classes: int) -> tuple[float, Matrix]:
    grad = Matrix.zeros(logits.rows, classes)
    loss = 0.0
    for r, label in enumerate(labels):
        row = logits.data[r * classes : (r + 1) * classes]
        max_v = max(row)
        exps = [math.exp(v - max_v) for v in row]
        total = sum(exps)
        probs = [e / total for e in exps]
        for c, p in enumerate(probs):
            grad.data[r * classes + c] = p - (1.0 if c == label else 0.0)
            if c == label:
                loss -= math.log(p + 1e-12)
    loss /= max(len(labels), 1)
    for v in grad.data:
        v /= max(len(labels), 1)
    return loss, grad
