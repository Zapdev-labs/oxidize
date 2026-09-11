from __future__ import annotations

from dataclasses import dataclass, field


def build_target_layer_ids(n_target: int, n_layers: int) -> list[int]:
    start = 1
    end = max(start, n_layers - 3)
    if n_target <= 1:
        return [min(start, n_layers - 1)]
    span = end - start
    raw = [round(start + i * span / (n_target - 1)) for i in range(n_target)]
    ids = list(dict.fromkeys(raw))
    if len(ids) < n_target:
        raise ValueError(
            f"cannot pick {n_target} distinct target layers from a {n_layers}-layer model "
            f"(only {len(ids)} distinct ids in [{start}, {end}]); lower num_target_layers"
        )
    return ids


def positional_loss_weights(block_predict: int, gamma: float) -> list[float]:
    if gamma <= 0:
        raise ValueError(f"loss decay gamma must be > 0, got {gamma}")
    if block_predict <= 0:
        return []
    raw = [pow(2.718281828459045, -i / gamma) for i in range(block_predict)]
    mean = sum(raw) / len(raw)
    return [w / mean for w in raw]


@dataclass
class DFlashTrainConfig:
    target_model: str = "Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16"
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft"
    hidden_size: int = 5120
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    vocab_size: int = 248320
    block_size: int = 16
    mask_token_id: int = 248070
    num_target_layers: int = 5
    target_n_layers: int = 64
    target_layer_ids: list[int] = field(default_factory=list)
    max_seq_len: int = 1024
    max_anchors: int = 48
    loss_decay_gamma: float = 7.0
    lr: float = 6e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.04
    grad_clip: float = 1.0
    # Passes over the hidden-cache; training stops at whichever of `epochs`
    # or `max_steps` is reached first. 0 means no epoch limit (max_steps only).
    epochs: int = 0
    max_steps: int = 2000
    grad_accum: int = 8
    seed: int = 42
    load_in_4bit: bool = True
    # Remote code from the target repo is only executed when explicitly opted
    # in, and then only at a pinned revision (commit sha) of that repo.
    trust_remote_code: bool = False
    target_revision: str = ""
    # Keep the frozen target embeddings / lm_head on the training device in
    # bf16. Set False to keep them on CPU for low-GPU-memory runs.
    io_on_device: bool = True
    max_samples: int = 40_000
    hub_repo: str = ""
    hub_private: bool = True

    def __post_init__(self) -> None:
        if not self.target_layer_ids:
            self.target_layer_ids = build_target_layer_ids(
                self.num_target_layers, self.target_n_layers
            )
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError(f"target_layer_ids contains duplicates: {self.target_layer_ids}")
        if len(self.target_layer_ids) != self.num_target_layers:
            self.num_target_layers = len(self.target_layer_ids)
        if self.trust_remote_code and not self.target_revision:
            raise ValueError("trust_remote_code requires a pinned target_revision")

    @property
    def n_feat(self) -> int:
        return self.hidden_size * self.num_target_layers

    @property
    def block_predict(self) -> int:
        return max(self.block_size - 1, 1)

