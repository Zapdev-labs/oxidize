"""Layer-wise LRU model mirroring oxidize-golang/core/model/layer_wise.go."""

from __future__ import annotations

import threading
from collections import OrderedDict

from oxidize_python.core.kv_cache import Cache, EvictionStrategy, Quantization
from oxidize_python.core.kv_cache import Config as KvConfig
from oxidize_python.core.model.inference import InferenceConfig, InferenceModel, WeightStorage, Workspace
from oxidize_python.core.model.model import EmptyInputError, Logits, Session, Token


class LayerWiseModel:
    def __init__(
        self,
        config: InferenceConfig,
        storage: WeightStorage,
        cache_size: int = 4,
        inner: InferenceModel | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.inner = inner
        self.workspace = Workspace(config.hidden_size * 4)
        self.cache_size = cache_size if cache_size > 0 else 4
        kv_cfg = KvConfig(
            layer_count=config.layer_count,
            context_size=config.context_size,
            head_count=config.num_key_value_heads,
            head_dim=config.kv_head_dim(),
            dtype="f32",
            quantization=Quantization.ASYMMETRIC,
            eviction=EvictionStrategy.SLIDING_WINDOW,
        )
        self.kv_cache = Cache.new(kv_cfg)
        self._cache: OrderedDict[int, None] = OrderedDict()
        self._mu = threading.Lock()

    def forward(self, tokens: list[Token], session: Session) -> Logits:
        if not tokens:
            raise EmptyInputError
        if self.config.layer_count > 0:
            for t in tokens:
                self._touch_layer(int(t) % self.config.layer_count)
        if self.inner is not None:
            return self.inner.forward(tokens, session)
        return [0.0] * self.config.vocab_size

    def _touch_layer(self, idx: int) -> None:
        with self._mu:
            if idx in self._cache:
                self._cache.move_to_end(idx)
                return
            self._cache[idx] = None
            self._cache.move_to_end(idx)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def vocab_size(self) -> int:
        return self.config.vocab_size

    def context_size(self) -> int:
        return self.config.context_size

    def layer_count(self) -> int:
        return self.config.layer_count


def new_layer_wise_from_inference(inner: InferenceModel, cache_size: int) -> LayerWiseModel:
    if inner is None:
        from oxidize_python.core.model.inference_config import default_inference_config

        return LayerWiseModel(default_inference_config(), WeightStorage(), cache_size)
    model = LayerWiseModel(inner.config, inner.storage, cache_size, inner=inner)
    model.kv_cache = inner.kv_cache
    return model


def new_layer_wise_from_gguf(file: object, cache_size: int) -> LayerWiseModel:
    from oxidize_python.core.ggufcore.gguf import MappedFile
    from oxidize_python.core.model.inference_config import (
        default_inference_config,
        inference_config_from_gguf,
    )

    cfg = (
        inference_config_from_gguf(file)
        if hasattr(file, "metadata")
        else default_inference_config()
    )
    storage = WeightStorage(file=MappedFile(bytes_data=None, parsed=file))
    return LayerWiseModel(cfg, storage, cache_size)
