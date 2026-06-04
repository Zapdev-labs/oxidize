"""DFlash forward helpers mirroring oxidize-golang/core/model/dflash_forward.go."""

from __future__ import annotations

from oxidize_python.core.model.dflash import DFlashDraftModel
from oxidize_python.core.tensor.ops import rms_norm_f32


class DFlashForwardError(Exception):
    pass


def cache_target_hidden(model: DFlashDraftModel, hidden: list[float]) -> None:
    want = model.config.target_hidden_width()
    if len(hidden) != want:
        raise DFlashForwardError(
            f"target hidden width mismatch: expected {want}, actual {len(hidden)}"
        )
    if not hasattr(model, "target_hidden_cache"):
        model.target_hidden_cache = []
    model.target_hidden_cache.append(list(hidden))


def clear_speculative_caches(model: DFlashDraftModel) -> None:
    if model.stack is not None:
        model.stack.reset_cache()
    if hasattr(model, "target_hidden_cache"):
        model.target_hidden_cache.clear()


def dflash_target_context(
    model: DFlashDraftModel, target_hidden: list[float]
) -> list[float] | None:
    h = model.config.hidden_size
    if not target_hidden or not getattr(model, "fc", None) or not model.fc.is_loaded():
        return None
    fused = model.fc.gemv(target_hidden)
    for i in range(min(h, len(getattr(model, "fc_bias", [])))):
        fused[i] += model.fc_bias[i]
    if not getattr(model, "hidden_norm", None):
        return fused
    ctx = [0.0] * h
    rms_norm_f32(fused[:h], model.hidden_norm[:h], ctx, model.config.rms_norm_eps)
    return ctx


def forward_token(
    model: DFlashDraftModel, token: int, target_hidden: list[float] | None = None
) -> list[float]:
    if model.stack is None:
        raise DFlashForwardError("DFlash draft model has no decoder stack")
    target_context = dflash_target_context(model, target_hidden or [])
    return model.stack.forward_token_with_context(token, target_context, _kv_context_factory(model))


def _kv_context_factory(model: DFlashDraftModel):
    def kv_context(
        layer_idx: int, ctx: list[float] | None
    ) -> tuple[list[float] | None, list[float] | None]:
        if ctx is None or model.stack is None:
            return None, None
        layer = model.stack.layers[layer_idx]
        kv_len = model.config.num_key_value_heads * model.config.head_dim()
        if (
            not layer.attention.k_proj.is_loaded()
            or not layer.attention.v_proj.is_loaded()
            or layer.attention.k_proj.input_dim() != len(ctx)
        ):
            return None, None
        k_ctx = [0.0] * kv_len
        v_ctx = [0.0] * kv_len
        layer.attention.k_proj.gemv(ctx, k_ctx)
        layer.attention.v_proj.gemv(ctx, v_ctx)
        return k_ctx, v_ctx

    return kv_context
