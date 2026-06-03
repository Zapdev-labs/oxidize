"""DFlash heuristic draft mirroring oxidize-golang/core/model/dflash_heuristic.go."""

from __future__ import annotations

import math
from dataclasses import dataclass

from oxidize_python.core.model.dflash import DFlashConfig
from oxidize_python.core.model.model import Logits, Model, Session, Token
from oxidize_python.core.model.sampling import SamplingConfig, _softmax, sample


@dataclass
class DFlashDecodeOpts:
    draft_steps: int = 4
    max_acceptance_rate: float = 0.95
    min_draft_confidence: float = 0.1
    temperature: float = 1.0


def default_dflash_decode_opts() -> DFlashDecodeOpts:
    return DFlashDecodeOpts()


@dataclass
class DFlashStats:
    drafts_generated: int = 0
    drafts_accepted: int = 0
    speedup_estimate: float = 0.0
    avg_draft_latency: float = 0.0

    def acceptance_rate(self) -> float:
        if self.drafts_generated == 0:
            return 0.0
        return self.drafts_accepted / self.drafts_generated


class HeuristicDFlashDraft:
    def __init__(
        self,
        target: Model,
        config: DFlashConfig,
        decode: DFlashDecodeOpts | None = None,
    ) -> None:
        self.target = target
        self.config = config
        self.decode = decode or default_dflash_decode_opts()
        if self.decode.draft_steps <= 0:
            self.decode.draft_steps = config.block_size or 4
        self.stats = DFlashStats()
        self.cache: list[Token] = []

    def forward(self, tokens: list[Token], session: Session) -> Logits:
        return self.target.forward(tokens, session)

    def vocab_size(self) -> int:
        return self.target.vocab_size()

    def context_size(self) -> int:
        return self.target.context_size()

    def layer_count(self) -> int:
        return self.target.layer_count()

    def reset(self) -> None:
        self.cache.clear()

    def generate_draft(self, prompt: list[Token], steps: int = 0) -> list[Token]:
        if steps <= 0:
            steps = self.decode.draft_steps
        out = list(prompt)
        sess = Session()
        for _ in range(steps):
            logits = self.target.forward(out, sess)
            tok = sample(
                logits,
                SamplingConfig(temperature=self.decode.temperature, top_p=1.0),
                None,
            )
            out.append(tok)
            self.cache.append(tok)
        self.stats.drafts_generated += steps
        return out[len(prompt) :]

    def verify_drafts(self, drafts: list[Token]) -> list[Token]:
        if not drafts:
            return []
        sess = Session()
        logits = self.target.forward(drafts, sess)
        probs = _softmax(logits)
        accepted: list[Token] = []
        for t in drafts:
            prob = probs[int(t)] if int(t) < len(probs) else 0.0
            if prob >= self.decode.min_draft_confidence:
                accepted.append(t)
        self.stats.drafts_accepted += len(accepted)
        return accepted


def draft_confidence(target: Model, drafts: list[Token]) -> list[float]:
    if not drafts:
        return []
    sess = Session()
    logits = target.forward(drafts, sess)
    probs = _softmax(logits)
    return [probs[int(t)] if int(t) < len(probs) else 0.0 for t in drafts]


class DFlashEngine:
    def __init__(
        self,
        target: Model,
        draft: HeuristicDFlashDraft,
        config: DFlashConfig,
    ) -> None:
        self.target = target
        self.draft = draft
        self.config = config
        self.decode = draft.decode
        self.stats = DFlashStats()

    def step(self, prompt: list[Token]) -> list[Token]:
        drafts = self.draft.generate_draft(prompt, self.decode.draft_steps)
        accepted = self.draft.verify_drafts(drafts)
        ratio = self.stats.acceptance_rate()
        self.stats.speedup_estimate = 1 + math.ceil(self.decode.draft_steps * ratio)
        return accepted

    def max_acceptance_rate(self) -> float:
        return self.decode.max_acceptance_rate

    def is_stable(self) -> bool:
        return self.stats.acceptance_rate() <= self.decode.max_acceptance_rate
