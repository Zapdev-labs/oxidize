"""Sampling utilities mirroring oxidize-golang/core/model/sampling.go."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from oxidize_python.core.model.model import Logits, Model, Session, Token

NEG_INF_F32 = float("-inf")
MIN_POSITIVE_F32 = 1.1754944e-38


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
    locally_typical_tau: float = 0.0
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
        locally_typical_tau=0.0,
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
    # Typical-P (entropy-based typicality filtering)
    if 0 < config.typical_p < 1.0:
        work = _typical_p(work, config.typical_p)
    # Tail-free (second-derivative cutoff)
    if 0 < config.tail_free_z < 1.0:
        work = _tail_free_z(work, config.tail_free_z)
    # Locally-typical (tau-based entropy deviation filtering)
    if config.locally_typical_tau > 0:
        work = _locally_typical_tau(work, config.locally_typical_tau)
    # Fast unfiltered path for large vocabularies when no rank/probability
    # filters are active: avoids an allocating full softmax pass.
    if (
        len(work) >= 4096
        and config.top_k == 0
        and config.min_p == 0
        and (config.top_p == 0 or config.top_p >= 1.0)
        and config.typical_p == 0
        and config.tail_free_z == 0
        and config.locally_typical_tau == 0
    ):
        temp = config.temperature if config.temperature > 0 else 1.0
        return sample_unfiltered(work, temp, rng)
    probs = _softmax(work)
    return _sample_categorical(probs, rng)


def sample_unfiltered(
    logits: Logits,
    temperature: float,
    rng: random.Random | None = None,
) -> Token:
    """Draw a token directly from temperature-scaled logits without building an
    intermediate normalized softmax slice. Mirrors sample_unfiltered in
    sampling.rs and is a fast path for large vocabularies."""
    if not logits:
        return 0
    if rng is None:
        rng = random.Random(1)
    if temperature <= 0:
        temperature = 1.0
    max_logit = max(logits)
    raw_sum = 0.0
    for v in logits:
        raw_sum += math.exp((v - max_logit) / temperature)
    if raw_sum <= 0 or math.isinf(raw_sum) or math.isnan(raw_sum):
        return greedy(logits)
    target = rng.random() * raw_sum
    cumulative = 0.0
    for i, v in enumerate(logits):
        cumulative += math.exp((v - max_logit) / temperature)
        if target <= cumulative:
            return i
    return greedy(logits)


def _typical_p(logits: Logits, p: float) -> Logits:
    """Keep the minimal set of tokens (ordered by closeness of their surprise to
    the distribution entropy) whose cumulative probability reaches p. Mirrors
    apply_typical_sampling in sampling.rs."""
    if p <= 0 or not logits:
        return logits
    probs = _softmax(logits)
    entropy = 0.0
    for pr in probs:
        pr = max(pr, MIN_POSITIVE_F32)
        entropy -= pr * math.log(pr)
    cands = []
    for i, pr in enumerate(probs):
        surprise = -math.log(max(pr, MIN_POSITIVE_F32))
        cands.append((abs(surprise - entropy), i))
    cands.sort(key=lambda c: c[0])
    keep = [False] * len(probs)
    cum = 0.0
    for _, idx in cands:
        keep[idx] = True
        cum += probs[idx]
        if cum >= p:
            break
    out = list(logits)
    for i in range(len(out)):
        if not keep[i]:
            out[i] = NEG_INF_F32
    return out


def _tail_free_z(logits: Logits, z: float) -> Logits:
    """Remove the low-probability tail using the second derivative of the sorted
    probability curve. Mirrors apply_tail_free_sampling in sampling.rs."""
    if z <= 0 or len(logits) <= 2:
        return logits
    idx = _sorted_indices(logits)
    probs = _softmax(logits)
    second_deriv = []
    for i in range(len(idx) - 2):
        d1 = probs[idx[i]] - probs[idx[i + 1]]
        d2 = probs[idx[i + 1]] - probs[idx[i + 2]]
        second_deriv.append(abs(d1 - d2))
    sd_sum = sum(second_deriv)
    if sd_sum <= 0 or math.isinf(sd_sum) or math.isnan(sd_sum):
        return logits
    cutoff = len(idx)
    cum = 0.0
    for i, sd in enumerate(second_deriv):
        cum += sd / sd_sum
        if cum >= z:
            cutoff = max(i + 2, 1)
            break
    keep = [False] * len(logits)
    for i in range(min(cutoff, len(idx))):
        keep[idx[i]] = True
    out = list(logits)
    for i in range(len(out)):
        if not keep[i]:
            out[i] = NEG_INF_F32
    return out


def _locally_typical_tau(logits: Logits, tau: float) -> Logits:
    """Keep tokens whose surprise lies within tau*entropy of the distribution
    entropy. Mirrors apply_locally_typical_sampling in sampling.rs."""
    if tau <= 0 or not logits:
        return logits
    probs = _softmax(logits)
    entropy = 0.0
    for pr in probs:
        pr = max(pr, MIN_POSITIVE_F32)
        entropy -= pr * math.log(pr)
    deviation_limit = entropy * tau
    keep = [False] * len(probs)
    any_kept = False
    for i, pr in enumerate(probs):
        surprise = -math.log(max(pr, MIN_POSITIVE_F32))
        if abs(surprise - entropy) <= deviation_limit:
            keep[i] = True
            any_kept = True
    if not any_kept:
        max_idx = max(range(len(probs)), key=lambda i: probs[i])
        keep[max_idx] = True
    out = list(logits)
    for i in range(len(out)):
        if not keep[i]:
            out[i] = NEG_INF_F32
    return out


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


@dataclass
class SpeculativeVerifyResult:
    """Mirrors SpeculativeDecodeResult in sampling.rs and carries enough detail
    for statistics tracking."""

    tokens: list[Token]
    accepted_draft_tokens: int
    used_residual_fallback: bool


def _softmax_probs_temp(logits: Logits, temperature: float) -> list[float]:
    """Temperature-scaled softmax over logits."""
    if not logits:
        raise SamplingError("empty logits")
    if temperature <= 0 or math.isinf(temperature) or math.isnan(temperature):
        temperature = 1.0
    max_logit = max(logits)
    out = [math.exp((v - max_logit) / temperature) for v in logits]
    total = sum(out)
    if total <= 0 or math.isinf(total) or math.isnan(total):
        raise SamplingError("non-finite softmax sum")
    inv = 1.0 / total
    return [x * inv for x in out]


def _residual_probs(target: list[float], draft: list[float]) -> list[float]:
    """Compute the normalized residual distribution max(p-q, 0). Mirrors
    residual_probs in sampling.rs."""
    out = []
    total = 0.0
    for i in range(len(target)):
        d = draft[i] if i < len(draft) else 0.0
        r = target[i] - d
        if r < 0:
            r = 0.0
        out.append(r)
        total += r
    if total <= 0:
        out = list(target)
        tsum = sum(target)
        if tsum > 0:
            inv = 1.0 / tsum
            out = [v * inv for v in out]
        return out
    inv = 1.0 / total
    return [v * inv for v in out]


def _sample_probabilities(probs: list[float], r: float) -> int:
    """Draw an index from a normalized distribution using a single random value."""
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


def speculative_decode_logits(
    draft_tokens: list[Token],
    draft_logits: list[Logits],
    target_logits: list[Logits],
    config: SamplingConfig,
    randoms: list[float],
) -> SpeculativeVerifyResult:
    """Verify draft tokens against precomputed target logits using the
    speculative acceptance/rejection rule with residual fallback. Faithful port
    of speculative_decode in sampling.rs.

    - draft_tokens:  proposed tokens (len = N)
    - draft_logits:  draft model logits per proposed token (len = N)
    - target_logits: target model logits (len = N+1; last is the bonus position)
    - randoms:       random draws in [0,1) (len >= N+1)
    """
    n = len(draft_tokens)
    if (
        n == 0
        or len(draft_logits) != n
        or len(target_logits) != n + 1
        or len(randoms) < n + 1
    ):
        raise SamplingError("invalid speculative inputs")
    greedy_mode = config.temperature <= 0 or config.top_k == 1
    verify_temp = 1.0 if greedy_mode else config.temperature
    emitted: list[Token] = []
    for step in range(n):
        draft_tok = draft_tokens[step]
        if greedy_mode:
            target_argmax = greedy(target_logits[step])
            if draft_tok == target_argmax:
                emitted.append(draft_tok)
                continue
            emitted.append(target_argmax)
            return SpeculativeVerifyResult(
                tokens=emitted,
                accepted_draft_tokens=step,
                used_residual_fallback=True,
            )
        draft_probs = _softmax_probs_temp(draft_logits[step], verify_temp)
        target_probs = _softmax_probs_temp(target_logits[step], verify_temp)
        if len(draft_probs) != len(target_probs):
            raise SamplingError("speculative vocab mismatch")
        ti = int(draft_tok)
        if ti >= len(draft_probs):
            raise SamplingError("speculative token out of range")
        q = max(draft_probs[ti], MIN_POSITIVE_F32)
        p = target_probs[ti]
        accept_prob = min(p / q, 1.0)
        if randoms[step] <= accept_prob:
            emitted.append(draft_tok)
            continue
        residual = _residual_probs(target_probs, draft_probs)
        sampled = _sample_probabilities(residual, randoms[step])
        emitted.append(sampled)
        return SpeculativeVerifyResult(
            tokens=emitted,
            accepted_draft_tokens=step,
            used_residual_fallback=True,
        )
    # All drafts accepted: sample the bonus token from the final position.
    final_rng = random.Random(int.from_bytes(_f32_bits(randoms[n]), "little"))
    final_tok = sample(list(target_logits[n]), config, final_rng)
    emitted.append(final_tok)
    return SpeculativeVerifyResult(
        tokens=emitted,
        accepted_draft_tokens=n,
        used_residual_fallback=False,
    )


def _f32_bits(value: float) -> bytes:
    import struct

    return struct.pack("<f", value)


def sample_mirostat_v2(
    logits: Logits,
    temperature: float,
    config: MirostatConfig,
    mu: float,
    rand_value: float,
) -> tuple[Token, float]:
    """Fully-validated Mirostat v2 sampler mirroring sample_mirostat in
    sampling.rs: validates temperature, tau/eta/mu and the random draw, builds a
    temperature-scaled softmax, picks the token whose surprisal is closest to the
    running target mu, and returns the updated mu."""
    if not logits:
        raise SamplingError("empty logits")
    if math.isinf(temperature) or math.isnan(temperature) or temperature <= 0:
        raise SamplingError("invalid temperature")
    if (
        math.isinf(config.tau)
        or math.isnan(config.tau)
        or config.tau <= 0
        or math.isinf(config.eta)
        or math.isnan(config.eta)
        or config.eta <= 0
        or math.isinf(mu)
        or math.isnan(mu)
    ):
        raise SamplingError("invalid mirostat parameters")
    if math.isinf(rand_value) or math.isnan(rand_value) or rand_value < 0 or rand_value >= 1:
        raise SamplingError("invalid random")
    probs = _softmax_probs_temp(logits, temperature)
    indexed = list(enumerate(probs))
    # Order by closeness of surprisal to the running target mu.
    indexed.sort(
        key=lambda e: abs(-math.log(max(e[1], MIN_POSITIVE_F32)) - mu)
    )
    total = sum(p for _, p in indexed)
    chosen = indexed[0]
    if total > 0:
        target = rand_value * total
        cum = 0.0
        for e in indexed:
            cum += e[1]
            if target <= cum:
                chosen = e
                break
    observed = -math.log(max(chosen[1], MIN_POSITIVE_F32))
    updated_mu = mu - config.eta * (observed - config.tau)
    return chosen[0], updated_mu


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


def beam_search_logits(
    logits_per_step: list[Logits],
    beam_width: int,
    eos_token: Token | None = None,
) -> BeamSearchResult:
    """Run beam search over precomputed per-step logits with EOS early stopping
    and final length-aware selection. Mirrors beam_search in sampling.rs."""
    if beam_width <= 0:
        raise SamplingError("invalid beam width")
    if not logits_per_step:
        raise SamplingError("invalid beam search inputs")
    for layer in logits_per_step:
        if not layer:
            raise SamplingError("invalid beam search inputs")

    @dataclass
    class _Beam:
        tokens: list[Token]
        score: float
        finished: bool

    beams = [_Beam(tokens=[], score=0.0, finished=False)]
    for step_logits in logits_per_step:
        probs = _softmax(step_logits)
        candidates: list[_Beam] = []
        for b in beams:
            if b.finished:
                candidates.append(b)
                continue
            for tok_idx, p in enumerate(probs):
                if p <= 0 or math.isinf(p) or math.isnan(p):
                    continue
                nxt = list(b.tokens)
                nxt.append(tok_idx)
                finished = eos_token is not None and eos_token == tok_idx
                candidates.append(
                    _Beam(
                        tokens=nxt,
                        score=b.score + math.log(p),
                        finished=finished,
                    )
                )
        if not candidates:
            raise SamplingError("empty logits")
        candidates.sort(key=lambda x: x.score, reverse=True)
        beams = candidates[:beam_width]
        if all(b.finished for b in beams):
            break
    best = beams[0]
    for b in beams[1:]:
        if b.score > best.score:
            best = b
    return BeamSearchResult(tokens=best.tokens, score=best.score)


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
