"""Sampling utilities mirroring oxidize-golang/core/model/sampling.go."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from oxidize_python.core.model.model import Logits, Model, Session, Token

NEG_INF_F32 = float("-inf")


class SamplingError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"sampling: {message}")


@dataclass
class RepetitionPenaltyConfig:
    penalty: float = 1.0
    last_n: int = 64
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


def default_repetition_penalty() -> RepetitionPenaltyConfig:
    return RepetitionPenaltyConfig(penalty=1.0, last_n=64)


@dataclass
class MirostatConfig:
    tau: float = 5.0
    eta: float = 0.1
    m: int = 5


def default_mirostat() -> MirostatConfig:
    return MirostatConfig(tau=5.0, eta=0.1, m=5)


@dataclass
class NewlinePenalty:
    count: int = 0
    reward: float = 0.0


class GrammarSymbol(Protocol):
    pass


@dataclass
class GrammarTerminal:
    token_id: int


@dataclass
class GrammarNonTerminal:
    name: str


@dataclass
class GrammarConstraint:
    start: str
    productions: dict[str, list[list[GrammarSymbol]]] = field(default_factory=dict)
    max_states: int = 20000

    def add_production(self, name: str, body: list[GrammarSymbol]) -> None:
        self.productions.setdefault(name, []).append(body)

    def allows_token(self, token: Token, _history: list[Token]) -> bool:
        productions = self.productions.get(self.start)
        if not productions:
            return True
        for prod in productions:
            for sym in prod:
                if isinstance(sym, GrammarTerminal) and sym.token_id == token:
                    return True
        return len(productions) == 0


@dataclass
class XtcSamplerConfig:
    probability: float = 0.0
    threshold: float = 0.0


@dataclass
class DrySamplerConfig:
    multiplier: float = 0.0
    base: float = 0.0
    allowed_length: int = 0
    penalty_last_n: int = 0
    sequence_breakers: list[str] = field(default_factory=list)


@dataclass
class SamplerChain:
    steps: list[str] = field(default_factory=list)


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    typical_p: float = 0.0
    tail_free_z: float = 0.0
    repetition: RepetitionPenaltyConfig = field(default_factory=default_repetition_penalty)
    mirostat: MirostatConfig = field(default_factory=MirostatConfig)
    newline_penalty: NewlinePenalty = field(default_factory=NewlinePenalty)
    grammar: GrammarConstraint | None = None
    suppressed_tokens: list[Token] | None = None
    xtc: XtcSamplerConfig | None = None
    dry: DrySamplerConfig | None = None
    chain: SamplerChain | None = None


def default_sampling_config() -> SamplingConfig:
    return SamplingConfig(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        typical_p=0.0,
        tail_free_z=0.0,
        repetition=default_repetition_penalty(),
        mirostat=MirostatConfig(),
        newline_penalty=NewlinePenalty(),
        suppressed_tokens=None,
    )


def greedy(logits: Logits) -> Token:
    if not logits:
        raise SamplingError("empty logits")
    best = 0
    best_val = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > best_val:
            best_val = logits[i]
            best = i
    return best


def sample(
    logits: Logits,
    config: SamplingConfig,
    rng: random.Random | None = None,
) -> Token:
    if not logits:
        raise SamplingError("empty logits")
    if rng is None:
        rng = random.Random(1)
    work = list(logits)
    if config.repetition.penalty != 1.0:
        work = _apply_repetition_penalty(work, config.repetition, None)
    if config.temperature > 0 and config.temperature != 1.0:
        work = [v / config.temperature for v in work]
    if config.suppressed_tokens:
        sup = set(config.suppressed_tokens)
        for t in sup:
            if t < len(work):
                work[t] = NEG_INF_F32
    if config.top_k > 0:
        work = _top_k(work, config.top_k)
    if config.min_p > 0:
        work = _min_p(work, config.min_p)
    if 0 < config.top_p < 1.0:
        work = _top_p(work, config.top_p)
    probs = _softmax(work)
    return _sample_categorical(probs, rng)


def sample_with_repetition(
    logits: Logits,
    config: SamplingConfig,
    history: list[Token],
    rng: random.Random | None = None,
) -> Token:
    work = list(logits)
    if config.repetition.penalty != 1.0:
        work = _apply_repetition_penalty(work, config.repetition, history)
    return sample(work, config, rng)


def sample_with_repetition_and_grammar(
    logits: Logits,
    config: SamplingConfig,
    history: list[Token],
    rng: random.Random | None = None,
) -> Token:
    work = list(logits)
    if config.grammar is not None:
        for t in range(len(work)):
            if not config.grammar.allows_token(t, history):
                work[t] = NEG_INF_F32
    return sample_with_repetition(work, config, history, rng)


@dataclass
class SpeculativeDecodeResult:
    accepted: list[Token]
    bonus_token: Token | None
    num_drafts: int


def speculative_decode(
    draft: Model,
    target: Model,
    draft_tokens: int,
    session: Session,
) -> tuple[list[Token], int]:
    if draft_tokens <= 0:
        draft_tokens = 4
    from oxidize_python.core.model.generation import GenerationConfig, GenerationStream

    draft_out: list[Token] = []
    ds = GenerationStream(
        draft,
        session,
        GenerationConfig(
            max_new_tokens=draft_tokens,
            stop_token=0,
            sampling=default_sampling_config(),
        ),
    )
    while True:
        tok, done, err = ds.next()
        if err:
            raise err
        draft_out.append(tok)
        if done:
            break
    logits = target.forward(draft_out, session)
    best_tok = greedy(logits)
    accepted = list(draft_out)
    accepted.append(best_tok)
    return accepted, len(accepted)


def sample_mirostat(
    logits: Logits,
    config: MirostatConfig,
    last_surprise: float,
) -> tuple[Token, float]:
    if not logits:
        raise SamplingError("empty logits")
    sorted_logits = sorted(logits, reverse=True)
    probs = _softmax(sorted_logits)
    log_probs = [math.log(p) for p in probs]
    best_idx = 0
    best_dist = abs(log_probs[0] + config.tau)
    for i, lp in enumerate(log_probs):
        dist = abs(lp + config.tau)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    surprise = -log_probs[best_idx]
    if last_surprise == 0:
        updated = surprise
    else:
        updated = (1 - config.eta) * last_surprise + config.eta * surprise
    return best_idx, updated


@dataclass
class BeamSearchResult:
    tokens: list[Token]
    score: float


def beam_search(
    model: Model,
    prompt: list[Token],
    beam_width: int,
    max_steps: int,
    session: Session,
) -> BeamSearchResult:
    if beam_width <= 0:
        beam_width = 1

    @dataclass
    class _Beam:
        tokens: list[Token]
        score: float

    beams = [_Beam(tokens=list(prompt), score=0.0)]
    for _ in range(max_steps):
        candidates: list[_Beam] = []
        for b in beams:
            logits = model.forward(b.tokens, session)
            probs = _softmax(logits)
            for i, p in enumerate(probs):
                if math.isinf(p) or math.isnan(p):
                    continue
                toks = list(b.tokens)
                toks.append(i)
                candidates.append(_Beam(tokens=toks, score=b.score + math.log(p + 1e-12)))
        candidates.sort(key=lambda x: x.score, reverse=True)
        beams = candidates[:beam_width]
    if not beams:
        return BeamSearchResult(tokens=[], score=0.0)
    best = beams[0]
    norm = float(len(best.tokens))
    return BeamSearchResult(tokens=best.tokens, score=best.score / norm if norm else 0.0)


def _apply_repetition_penalty(
    logits: Logits,
    cfg: RepetitionPenaltyConfig,
    history: list[Token] | None,
) -> Logits:
    if cfg.penalty == 1.0 or not history:
        return logits
    hist = history
    if cfg.last_n > 0 and len(hist) > cfg.last_n:
        hist = hist[-cfg.last_n :]
    freq: dict[Token, int] = {}
    for t in hist:
        freq[t] = freq.get(t, 0) + 1
    out = list(logits)
    for t, count in freq.items():
        if t < len(out):
            if out[t] >= 0:
                out[t] = out[t] / cfg.penalty
            else:
                out[t] = out[t] * cfg.penalty
            if cfg.frequency_penalty != 0:
                out[t] -= count * cfg.frequency_penalty
            if cfg.presence_penalty != 0 and count > 0:
                out[t] -= cfg.presence_penalty
    return out


def _top_k(logits: Logits, k: int) -> Logits:
    idx = _sorted_indices(logits)
    cutoff = logits[idx[0]]
    if k < len(idx):
        cutoff = logits[idx[k - 1]]
    out = list(logits)
    for i, v in enumerate(out):
        if v < cutoff:
            out[i] = NEG_INF_F32
    return out


def _min_p(logits: Logits, p: float) -> Logits:
    probs = _softmax(logits)
    max_idx = max(range(len(probs)), key=lambda i: probs[i])
    cutoff = probs[max_idx] * p
    out = list(logits)
    for i, pv in enumerate(probs):
        if pv < cutoff and i != max_idx:
            out[i] = NEG_INF_F32
    return out


def _top_p(logits: Logits, p: float) -> Logits:
    idx = _sorted_indices(logits)
    cum_prob = 0.0
    cutoff = logits[idx[0]]
    for i in idx:
        cum_prob += math.exp(logits[i] - logits[idx[0]])
        if cum_prob >= p:
            cutoff = logits[i]
            break
    out = list(logits)
    for i, v in enumerate(out):
        if v < cutoff:
            out[i] = NEG_INF_F32
    return out


def _softmax(logits: Logits) -> list[float]:
    if not logits:
        return []
    max_val = max(logits)
    out = [math.exp(v - max_val) for v in logits]
    total = sum(out)
    inv = 1.0 / total if total else 0.0
    return [x * inv for x in out]


def _sorted_indices(logits: Logits) -> list[int]:
    return sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)


def _sample_categorical(probs: list[float], rng: random.Random) -> Token:
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
    return len(probs) - 1
