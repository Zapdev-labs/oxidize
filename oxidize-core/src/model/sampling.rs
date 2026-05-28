use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GrammarSymbol {
    Terminal(u32),
    NonTerminal(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarConstraint {
    start: String,
    productions: HashMap<String, Vec<Vec<GrammarSymbol>>>,
}

impl GrammarConstraint {
    pub fn new(
        start: impl Into<String>,
        productions: HashMap<String, Vec<Vec<GrammarSymbol>>>,
    ) -> Result<Self, SamplingError> {
        let start = start.into();
        if start.is_empty() || !productions.contains_key(&start) {
            return Err(SamplingError::InvalidGrammarConstraint);
        }
        for alternatives in productions.values() {
            for production in alternatives {
                for symbol in production {
                    if let GrammarSymbol::NonTerminal(non_terminal) = symbol
                        && !productions.contains_key(non_terminal)
                    {
                        return Err(SamplingError::InvalidGrammarConstraint);
                    }
                }
            }
        }
        Ok(Self { start, productions })
    }

    pub fn allows_token(&self, generated_tokens: &[u32], token: u32) -> bool {
        let mut candidate = Vec::with_capacity(generated_tokens.len() + 1);
        candidate.extend_from_slice(generated_tokens);
        candidate.push(token);
        self.accepts_prefix(&candidate)
    }

    fn accepts_prefix(&self, prefix: &[u32]) -> bool {
        #[derive(Clone, PartialEq, Eq, Hash)]
        struct ParseState {
            stack: Vec<GrammarSymbol>,
            consumed: usize,
        }

        const MAX_STATES: usize = 20_000;
        const MAX_STACK_LEN: usize = 256;

        let mut queue = VecDeque::new();
        let mut seen = HashSet::new();
        let initial = ParseState {
            stack: vec![GrammarSymbol::NonTerminal(self.start.clone())],
            consumed: 0,
        };
        seen.insert(initial.clone());
        queue.push_back(initial);

        while let Some(state) = queue.pop_front() {
            if state.consumed == prefix.len() {
                return true;
            }
            if seen.len() >= MAX_STATES || state.stack.is_empty() {
                continue;
            }

            let mut next_stack = state.stack;
            let Some(symbol) = next_stack.pop() else {
                continue;
            };

            match symbol {
                GrammarSymbol::Terminal(token) => {
                    if prefix[state.consumed] == token {
                        let next = ParseState {
                            stack: next_stack,
                            consumed: state.consumed + 1,
                        };
                        if seen.insert(next.clone()) {
                            queue.push_back(next);
                        }
                    }
                }
                GrammarSymbol::NonTerminal(non_terminal) => {
                    let Some(alternatives) = self.productions.get(&non_terminal) else {
                        continue;
                    };
                    for production in alternatives {
                        let mut expanded = next_stack.clone();
                        for item in production.iter().rev() {
                            expanded.push(item.clone());
                        }
                        if expanded.len() > MAX_STACK_LEN {
                            continue;
                        }
                        let next = ParseState {
                            stack: expanded,
                            consumed: state.consumed,
                        };
                        if seen.insert(next.clone()) {
                            queue.push_back(next);
                        }
                    }
                }
            }
        }

        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub typical_p: Option<f32>,
    pub tail_free_z: Option<f32>,
    pub locally_typical_tau: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NewlinePenalty {
    pub token_id: u32,
    pub penalty: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepetitionPenaltyConfig {
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub newline_penalty: Option<NewlinePenalty>,
}

impl Default for RepetitionPenaltyConfig {
    fn default() -> Self {
        Self {
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            newline_penalty: None,
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
            typical_p: None,
            tail_free_z: None,
            locally_typical_tau: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MirostatConfig {
    pub tau: f32,
    pub eta: f32,
    pub mu: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingError {
    EmptyLogits,
    InvalidTemperature,
    InvalidTopK,
    InvalidTopP,
    InvalidMinP,
    InvalidTypicalP,
    InvalidTailFreeZ,
    InvalidLocallyTypicalTau,
    InvalidFrequencyPenalty,
    InvalidPresencePenalty,
    InvalidNewlinePenalty,
    InvalidMirostat,
    InvalidRandom,
    InvalidGrammarConstraint,
    NoValidGrammarToken,
    InvalidSpeculativeInputs,
    InvalidBeamWidth,
    InvalidBeamSearchInputs,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeDecodeResult {
    pub tokens: Vec<u32>,
    pub accepted_draft_tokens: usize,
    pub used_residual_fallback: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeamSearchResult {
    pub tokens: Vec<u32>,
    pub score: f32,
}

pub fn greedy(logits: &[f32]) -> Result<u32, SamplingError> {
    let Some((idx, _)) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
    else {
        return Err(SamplingError::EmptyLogits);
    };
    Ok(idx as u32)
}

pub fn sample(logits: &[f32], config: SamplingConfig, random: f32) -> Result<u32, SamplingError> {
    sample_with_repetition(
        logits,
        config,
        random,
        &[],
        RepetitionPenaltyConfig::default(),
    )
}

pub fn sample_with_repetition(
    logits: &[f32],
    config: SamplingConfig,
    random: f32,
    recent_tokens: &[u32],
    repetition: RepetitionPenaltyConfig,
) -> Result<u32, SamplingError> {
    sample_with_repetition_and_grammar(logits, config, random, recent_tokens, repetition, &[], None)
}

pub fn sample_with_repetition_and_grammar(
    logits: &[f32],
    config: SamplingConfig,
    random: f32,
    recent_tokens: &[u32],
    repetition: RepetitionPenaltyConfig,
    generated_tokens: &[u32],
    grammar: Option<&GrammarConstraint>,
) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if !config.temperature.is_finite() {
        return Err(SamplingError::InvalidTemperature);
    }
    if config.top_k == Some(0) {
        return Err(SamplingError::InvalidTopK);
    }
    if let Some(top_p) = config.top_p
        && (!top_p.is_finite() || top_p <= 0.0 || top_p > 1.0)
    {
        return Err(SamplingError::InvalidTopP);
    }
    if let Some(min_p) = config.min_p
        && (!min_p.is_finite() || min_p <= 0.0 || min_p > 1.0)
    {
        return Err(SamplingError::InvalidMinP);
    }
    if let Some(typical_p) = config.typical_p
        && (!typical_p.is_finite() || typical_p <= 0.0 || typical_p > 1.0)
    {
        return Err(SamplingError::InvalidTypicalP);
    }
    if let Some(tail_free_z) = config.tail_free_z
        && (!tail_free_z.is_finite() || tail_free_z <= 0.0 || tail_free_z > 1.0)
    {
        return Err(SamplingError::InvalidTailFreeZ);
    }
    if let Some(locally_typical_tau) = config.locally_typical_tau
        && (!locally_typical_tau.is_finite() || locally_typical_tau <= 0.0)
    {
        return Err(SamplingError::InvalidLocallyTypicalTau);
    }
    if !repetition.frequency_penalty.is_finite() || repetition.frequency_penalty < 0.0 {
        return Err(SamplingError::InvalidFrequencyPenalty);
    }
    if !repetition.presence_penalty.is_finite() || repetition.presence_penalty < 0.0 {
        return Err(SamplingError::InvalidPresencePenalty);
    }
    if let Some(newline_penalty) = repetition.newline_penalty
        && (!newline_penalty.penalty.is_finite() || newline_penalty.penalty < 0.0)
    {
        return Err(SamplingError::InvalidNewlinePenalty);
    }
    if !random.is_finite() || !(0.0..1.0).contains(&random) {
        return Err(SamplingError::InvalidRandom);
    }

    let has_repetition_penalty = repetition.frequency_penalty != 0.0
        || repetition.presence_penalty != 0.0
        || repetition.newline_penalty.is_some();
    if (config.temperature <= 0.0 || config.top_k == Some(1))
        && !has_repetition_penalty
        && grammar.is_none()
    {
        return greedy(logits);
    }
    if config.temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature);
    }
    let has_rank_filter = config.top_k.is_some()
        || config.top_p.is_some()
        || config.min_p.is_some()
        || config.typical_p.is_some()
        || config.tail_free_z.is_some()
        || config.locally_typical_tau.is_some();
    if logits.len() >= 4096 && !has_repetition_penalty && !has_rank_filter && grammar.is_none() {
        return sample_unfiltered(logits, config.temperature, random);
    }

    let mut adjusted_logits = logits.to_vec();
    apply_repetition_penalties(&mut adjusted_logits, recent_tokens, repetition);

    let max_logit = adjusted_logits
        .iter()
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .ok_or(SamplingError::EmptyLogits)?;
    let top_k_limit = config.top_k.filter(|top_k| *top_k < adjusted_logits.len());
    let mut indexed_probs = if let Some(top_k) = top_k_limit {
        let mut raw_sum = 0.0_f32;
        let mut top_candidates: Vec<(usize, f32)> = Vec::with_capacity(top_k);
        for (idx, logit) in adjusted_logits.iter().copied().enumerate() {
            let prob = ((logit - max_logit) / config.temperature).exp();
            raw_sum += prob;
            if top_candidates.len() < top_k {
                top_candidates.push((idx, prob));
            } else if let Some((min_idx, _)) = top_candidates
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.1.total_cmp(&b.1.1))
                && prob > top_candidates[min_idx].1
            {
                top_candidates[min_idx] = (idx, prob);
            }
        }
        if raw_sum <= 0.0 || !raw_sum.is_finite() {
            return greedy(logits);
        }
        for (_, p) in &mut top_candidates {
            *p /= raw_sum;
        }
        top_candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        top_candidates
    } else {
        let mut indexed_probs: Vec<(usize, f32)> = adjusted_logits
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, logit)| (idx, ((logit - max_logit) / config.temperature).exp()))
            .collect();

        let raw_sum: f32 = indexed_probs.iter().map(|(_, p)| *p).sum();
        if raw_sum <= 0.0 || !raw_sum.is_finite() {
            return greedy(logits);
        }
        for (_, p) in &mut indexed_probs {
            *p /= raw_sum;
        }

        indexed_probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        if let Some(top_k) = config.top_k
            && indexed_probs.len() > top_k
        {
            indexed_probs.truncate(top_k);
        }
        indexed_probs
    };

    if let Some(top_p) = config.top_p {
        let mut cumulative = 0.0_f32;
        let cutoff = indexed_probs
            .iter()
            .position(|(_, prob)| {
                cumulative += *prob;
                cumulative >= top_p
            })
            .map_or(indexed_probs.len(), |idx| idx + 1);
        indexed_probs.truncate(cutoff);
    }

    if let Some(min_p) = config.min_p {
        let max_prob = indexed_probs.first().map_or(0.0, |(_, p)| *p);
        let threshold = max_prob * min_p;
        indexed_probs.retain(|(_, prob)| *prob >= threshold);
    }

    if let Some(typical_p) = config.typical_p {
        apply_typical_sampling(&mut indexed_probs, typical_p);
    }

    if let Some(tail_free_z) = config.tail_free_z {
        apply_tail_free_sampling(&mut indexed_probs, tail_free_z);
    }

    if let Some(locally_typical_tau) = config.locally_typical_tau {
        apply_locally_typical_sampling(&mut indexed_probs, locally_typical_tau);
    }

    if let Some(grammar) = grammar {
        indexed_probs.retain(|(idx, _)| grammar.allows_token(generated_tokens, *idx as u32));
    }

    if indexed_probs.is_empty() {
        if grammar.is_some() {
            return Err(SamplingError::NoValidGrammarToken);
        }
        return greedy(logits);
    }

    let filtered_sum: f32 = indexed_probs.iter().map(|(_, p)| *p).sum();
    if filtered_sum <= 0.0 || !filtered_sum.is_finite() {
        return greedy(logits);
    }

    let mut cumulative = 0.0_f32;
    let target = random * filtered_sum;
    for (idx, prob) in indexed_probs {
        cumulative += prob;
        if target <= cumulative {
            return Ok(idx as u32);
        }
    }

    greedy(logits)
}

fn sample_unfiltered(logits: &[f32], temperature: f32, random: f32) -> Result<u32, SamplingError> {
    let max_logit = logits
        .iter()
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .ok_or(SamplingError::EmptyLogits)?;

    let mut raw_sum = 0.0_f32;
    for logit in logits {
        raw_sum += ((*logit - max_logit) / temperature).exp();
    }
    if raw_sum <= 0.0 || !raw_sum.is_finite() {
        return greedy(logits);
    }

    let target = random * raw_sum;
    let mut cumulative = 0.0_f32;
    for (idx, logit) in logits.iter().enumerate() {
        cumulative += ((*logit - max_logit) / temperature).exp();
        if target <= cumulative {
            return Ok(idx as u32);
        }
    }

    greedy(logits)
}

pub fn speculative_decode(
    draft_tokens: &[u32],
    draft_logits: &[Vec<f32>],
    target_logits: &[Vec<f32>],
    sampling_config: SamplingConfig,
    randoms: &[f32],
) -> Result<SpeculativeDecodeResult, SamplingError> {
    if draft_tokens.is_empty()
        || draft_logits.len() != draft_tokens.len()
        || target_logits.len() != draft_tokens.len() + 1
        || randoms.len() < draft_tokens.len() + 1
    {
        return Err(SamplingError::InvalidSpeculativeInputs);
    }

    let mut emitted = Vec::with_capacity(draft_tokens.len() + 1);
    for (step, draft_token) in draft_tokens.iter().copied().enumerate() {
        let draft_probs = softmax_probs(&draft_logits[step], 1.0)?;
        let target_probs = softmax_probs(&target_logits[step], 1.0)?;
        if draft_probs.len() != target_probs.len() {
            return Err(SamplingError::InvalidSpeculativeInputs);
        }
        let token_idx = draft_token as usize;
        if token_idx >= draft_probs.len() {
            return Err(SamplingError::InvalidSpeculativeInputs);
        }
        let q = draft_probs[token_idx].max(f32::MIN_POSITIVE);
        let p = target_probs[token_idx];
        let accept_prob = (p / q).min(1.0);

        if randoms[step] <= accept_prob {
            emitted.push(draft_token);
            continue;
        }

        let residual = residual_probs(&target_probs, &draft_probs);
        let sampled = sample_probabilities(&residual, randoms[step])?;
        emitted.push(sampled as u32);
        return Ok(SpeculativeDecodeResult {
            tokens: emitted,
            accepted_draft_tokens: step,
            used_residual_fallback: true,
        });
    }

    let final_token = sample(
        &target_logits[draft_tokens.len()],
        sampling_config,
        randoms[draft_tokens.len()],
    )?;
    emitted.push(final_token);
    Ok(SpeculativeDecodeResult {
        tokens: emitted,
        accepted_draft_tokens: draft_tokens.len(),
        used_residual_fallback: false,
    })
}

pub fn beam_search(
    logits_per_step: &[Vec<f32>],
    beam_width: usize,
    eos_token: Option<u32>,
) -> Result<BeamSearchResult, SamplingError> {
    if beam_width == 0 {
        return Err(SamplingError::InvalidBeamWidth);
    }
    if logits_per_step.is_empty() || logits_per_step.iter().any(Vec::is_empty) {
        return Err(SamplingError::InvalidBeamSearchInputs);
    }

    #[derive(Clone)]
    struct Beam {
        tokens: Vec<u32>,
        score: f32,
        finished: bool,
    }

    let mut beams = vec![Beam {
        tokens: Vec::new(),
        score: 0.0,
        finished: false,
    }];

    for step_logits in logits_per_step {
        let probs = softmax_probs(step_logits, 1.0)?;
        let mut candidates = Vec::new();

        for beam in &beams {
            if beam.finished {
                candidates.push(beam.clone());
                continue;
            }

            for (token_idx, prob) in probs.iter().copied().enumerate() {
                if prob <= 0.0 || !prob.is_finite() {
                    continue;
                }
                let mut next_tokens = beam.tokens.clone();
                next_tokens.push(token_idx as u32);
                let is_finished = eos_token.is_some_and(|eos| eos == token_idx as u32);
                candidates.push(Beam {
                    tokens: next_tokens,
                    score: beam.score + prob.ln(),
                    finished: is_finished,
                });
            }
        }

        if candidates.is_empty() {
            return Err(SamplingError::EmptyLogits);
        }

        candidates.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        beams = candidates.into_iter().take(beam_width).collect();

        if beams.iter().all(|beam| beam.finished) {
            break;
        }
    }

    let best = beams
        .into_iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .ok_or(SamplingError::EmptyLogits)?;
    Ok(BeamSearchResult {
        tokens: best.tokens,
        score: best.score,
    })
}

fn apply_repetition_penalties(
    logits: &mut [f32],
    recent_tokens: &[u32],
    repetition: RepetitionPenaltyConfig,
) {
    if logits.is_empty() {
        return;
    }
    if repetition.frequency_penalty == 0.0
        && repetition.presence_penalty == 0.0
        && repetition.newline_penalty.is_none()
    {
        return;
    }

    let mut frequencies = vec![0_u32; logits.len()];
    for token in recent_tokens {
        let idx = *token as usize;
        if idx < frequencies.len() {
            frequencies[idx] = frequencies[idx].saturating_add(1);
        }
    }

    for (idx, logit) in logits.iter_mut().enumerate() {
        let freq = frequencies[idx];
        if freq == 0 {
            continue;
        }
        *logit -= repetition.frequency_penalty * freq as f32;
        *logit -= repetition.presence_penalty;
    }

    if let Some(newline_penalty) = repetition.newline_penalty {
        let idx = newline_penalty.token_id as usize;
        if idx < logits.len() {
            logits[idx] -= newline_penalty.penalty;
        }
    }
}

pub fn sample_mirostat(
    logits: &[f32],
    temperature: f32,
    config: MirostatConfig,
    random: f32,
) -> Result<(u32, f32), SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature);
    }
    if !config.tau.is_finite()
        || config.tau <= 0.0
        || !config.eta.is_finite()
        || config.eta <= 0.0
        || !config.mu.is_finite()
    {
        return Err(SamplingError::InvalidMirostat);
    }
    if !random.is_finite() || !(0.0..1.0).contains(&random) {
        return Err(SamplingError::InvalidRandom);
    }

    let mut indexed_probs = build_sorted_probs(logits, temperature)?;
    let target_surprisal = config.mu;
    indexed_probs.sort_by(|a, b| {
        let a_surprise = -a.1.max(f32::MIN_POSITIVE).ln();
        let b_surprise = -b.1.max(f32::MIN_POSITIVE).ln();
        (a_surprise - target_surprisal)
            .abs()
            .total_cmp(&(b_surprise - target_surprisal).abs())
    });

    let chosen = weighted_pick(&indexed_probs, random).ok_or(SamplingError::EmptyLogits)?;
    let observed_surprisal = -chosen.1.max(f32::MIN_POSITIVE).ln();
    let updated_mu = config.mu - config.eta * (observed_surprisal - config.tau);

    Ok((chosen.0 as u32, updated_mu))
}

fn build_sorted_probs(
    logits: &[f32],
    temperature: f32,
) -> Result<Vec<(usize, f32)>, SamplingError> {
    let max_logit = logits
        .iter()
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .ok_or(SamplingError::EmptyLogits)?;
    let mut indexed_probs: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, logit)| (idx, ((logit - max_logit) / temperature).exp()))
        .collect();

    let raw_sum: f32 = indexed_probs.iter().map(|(_, p)| *p).sum();
    if raw_sum <= 0.0 || !raw_sum.is_finite() {
        return Err(SamplingError::EmptyLogits);
    }
    for (_, p) in &mut indexed_probs {
        *p /= raw_sum;
    }

    indexed_probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    Ok(indexed_probs)
}

fn apply_typical_sampling(indexed_probs: &mut Vec<(usize, f32)>, typical_p: f32) {
    if indexed_probs.is_empty() {
        return;
    }
    let entropy: f32 = indexed_probs
        .iter()
        .map(|(_, p)| {
            let p = p.max(f32::MIN_POSITIVE);
            -p * p.ln()
        })
        .sum();
    let mut by_typicality: Vec<(usize, f32, f32)> = indexed_probs
        .iter()
        .map(|(idx, prob)| {
            let surprise = -prob.max(f32::MIN_POSITIVE).ln();
            (*idx, *prob, (surprise - entropy).abs())
        })
        .collect();
    by_typicality.sort_unstable_by(|a, b| a.2.total_cmp(&b.2));

    let mut cumulative = 0.0_f32;
    let mut keep = Vec::with_capacity(by_typicality.len());
    for (idx, prob, _) in by_typicality {
        keep.push((idx, prob));
        cumulative += prob;
        if cumulative >= typical_p {
            break;
        }
    }
    keep.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    *indexed_probs = keep;
}

fn apply_tail_free_sampling(indexed_probs: &mut Vec<(usize, f32)>, tail_free_z: f32) {
    if indexed_probs.len() <= 2 {
        return;
    }
    let mut second_derivative = Vec::with_capacity(indexed_probs.len().saturating_sub(2));
    for i in 0..indexed_probs.len() - 2 {
        let d1 = indexed_probs[i].1 - indexed_probs[i + 1].1;
        let d2 = indexed_probs[i + 1].1 - indexed_probs[i + 2].1;
        second_derivative.push((d1 - d2).abs());
    }
    let sd_sum: f32 = second_derivative.iter().sum();
    if sd_sum <= 0.0 || !sd_sum.is_finite() {
        return;
    }

    let mut cumulative = 0.0_f32;
    let mut cutoff = indexed_probs.len();
    for (i, sd) in second_derivative.into_iter().enumerate() {
        cumulative += sd / sd_sum;
        if cumulative >= tail_free_z {
            cutoff = (i + 2).max(1);
            break;
        }
    }
    indexed_probs.truncate(cutoff);
}

fn apply_locally_typical_sampling(indexed_probs: &mut Vec<(usize, f32)>, locally_typical_tau: f32) {
    if indexed_probs.is_empty() {
        return;
    }
    let entropy: f32 = indexed_probs
        .iter()
        .map(|(_, p)| {
            let p = p.max(f32::MIN_POSITIVE);
            -p * p.ln()
        })
        .sum();
    let deviation_limit = entropy * locally_typical_tau;
    let mut filtered: Vec<(usize, f32)> = indexed_probs
        .iter()
        .copied()
        .filter(|(_, prob)| {
            let surprise = -prob.max(f32::MIN_POSITIVE).ln();
            (surprise - entropy).abs() <= deviation_limit
        })
        .collect();
    if filtered.is_empty() {
        filtered.push(indexed_probs[0]);
    }
    filtered.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    *indexed_probs = filtered;
}

fn weighted_pick(indexed_probs: &[(usize, f32)], random: f32) -> Option<(usize, f32)> {
    if indexed_probs.is_empty() {
        return None;
    }
    let filtered_sum: f32 = indexed_probs.iter().map(|(_, p)| *p).sum();
    if filtered_sum <= 0.0 || !filtered_sum.is_finite() {
        return None;
    }

    let mut cumulative = 0.0_f32;
    let target = random * filtered_sum;
    for (idx, prob) in indexed_probs.iter().copied() {
        cumulative += prob;
        if target <= cumulative {
            return Some((idx, prob));
        }
    }
    indexed_probs.last().copied()
}

fn softmax_probs(logits: &[f32], temperature: f32) -> Result<Vec<f32>, SamplingError> {
    let max_logit = logits
        .iter()
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .ok_or(SamplingError::EmptyLogits)?;
    let mut probs: Vec<f32> = logits
        .iter()
        .copied()
        .map(|logit| ((logit - max_logit) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return Err(SamplingError::EmptyLogits);
    }
    for prob in &mut probs {
        *prob /= sum;
    }
    Ok(probs)
}

fn residual_probs(target_probs: &[f32], draft_probs: &[f32]) -> Vec<f32> {
    let mut residual: Vec<f32> = target_probs
        .iter()
        .zip(draft_probs.iter())
        .map(|(p, q)| (p - q).max(0.0))
        .collect();
    let sum: f32 = residual.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        for prob in &mut residual {
            *prob /= sum;
        }
    }
    residual
}

fn sample_probabilities(probs: &[f32], random: f32) -> Result<usize, SamplingError> {
    if !random.is_finite() || !(0.0..1.0).contains(&random) {
        return Err(SamplingError::InvalidRandom);
    }
    if probs.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return probs
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(idx, _)| idx)
            .ok_or(SamplingError::EmptyLogits);
    }

    let mut cumulative = 0.0_f32;
    let target = random * sum;
    for (idx, prob) in probs.iter().copied().enumerate() {
        cumulative += prob;
        if target <= cumulative {
            return Ok(idx);
        }
    }
    Ok(probs.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_digit_grammar() -> GrammarConstraint {
        let productions = HashMap::from([
            (
                "S".to_string(),
                vec![vec![
                    GrammarSymbol::NonTerminal("D".to_string()),
                    GrammarSymbol::NonTerminal("D".to_string()),
                ]],
            ),
            (
                "D".to_string(),
                vec![
                    vec![GrammarSymbol::Terminal(0)],
                    vec![GrammarSymbol::Terminal(1)],
                ],
            ),
        ]);
        GrammarConstraint::new("S", productions).expect("grammar should be valid")
    }

    #[test]
    fn greedy_returns_highest_logit_index() {
        assert_eq!(greedy(&[0.5, 2.0, 1.0]).expect("greedy should succeed"), 1);
    }

    #[test]
    fn temperature_changes_sampling_sharpness() {
        let logits = [0.0, 0.2];
        let cooler = sample(
            &logits,
            SamplingConfig {
                temperature: 0.5,
                ..SamplingConfig::default()
            },
            0.55,
        )
        .expect("sampling should succeed");
        let hotter = sample(
            &logits,
            SamplingConfig {
                temperature: 2.0,
                ..SamplingConfig::default()
            },
            0.55,
        )
        .expect("sampling should succeed");

        assert_eq!(cooler, 1);
        assert_eq!(hotter, 0);
    }

    #[test]
    fn top_k_limits_candidate_set() {
        let token = sample(
            &[5.0, 4.0, 3.0, 2.0],
            SamplingConfig {
                top_k: Some(2),
                ..SamplingConfig::default()
            },
            0.99,
        )
        .expect("sampling should succeed");
        assert!(token <= 1);
    }

    #[test]
    fn top_p_uses_nucleus_subset() {
        let token = sample(
            &[5.0, 4.0, 3.0, 2.0],
            SamplingConfig {
                top_p: Some(0.6),
                ..SamplingConfig::default()
            },
            0.99,
        )
        .expect("sampling should succeed");
        assert_eq!(token, 0);
    }

    #[test]
    fn min_p_filters_low_probability_tail() {
        let token = sample(
            &[6.0, 5.9, 2.0, 1.0],
            SamplingConfig {
                min_p: Some(0.9),
                ..SamplingConfig::default()
            },
            0.99,
        )
        .expect("sampling should succeed");
        assert!(token <= 1);
    }

    #[test]
    fn typical_sampling_keeps_typical_tokens() {
        let token = sample(
            &[6.0, 5.95, 4.0, 1.0],
            SamplingConfig {
                typical_p: Some(0.5),
                ..SamplingConfig::default()
            },
            0.95,
        )
        .expect("sampling should succeed");
        assert!(token <= 1);
    }

    #[test]
    fn tail_free_sampling_trims_long_tail() {
        let token = sample(
            &[7.0, 5.0, 3.0, 2.0, 1.0],
            SamplingConfig {
                tail_free_z: Some(0.4),
                ..SamplingConfig::default()
            },
            0.99,
        )
        .expect("sampling should succeed");
        assert!(token <= 2);
    }

    #[test]
    fn locally_typical_sampling_prefers_entropy_band() {
        let token = sample(
            &[6.0, 5.9, 2.0, 1.5],
            SamplingConfig {
                locally_typical_tau: Some(0.7),
                ..SamplingConfig::default()
            },
            0.99,
        )
        .expect("sampling should succeed");
        assert!(token <= 1);
    }

    #[test]
    fn mirostat_sampling_returns_updated_mu() {
        let (token, next_mu) = sample_mirostat(
            &[5.0, 4.0, 3.0, 2.0],
            1.0,
            MirostatConfig {
                tau: 2.0,
                eta: 0.1,
                mu: 4.0,
            },
            0.4,
        )
        .expect("mirostat should succeed");
        assert!(token <= 3);
        assert!(next_mu.is_finite());
        assert_ne!(next_mu, 4.0);
    }

    #[test]
    fn rejects_invalid_sampling_inputs() {
        assert_eq!(
            sample(&[], SamplingConfig::default(), 0.3),
            Err(SamplingError::EmptyLogits)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    temperature: 0.0,
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidTemperature)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    top_k: Some(0),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidTopK)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    top_p: Some(1.5),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidTopP)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    min_p: Some(-0.1),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidMinP)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    typical_p: Some(0.0),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidTypicalP)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    tail_free_z: Some(1.2),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidTailFreeZ)
        );
        assert_eq!(
            sample(
                &[1.0],
                SamplingConfig {
                    locally_typical_tau: Some(0.0),
                    ..SamplingConfig::default()
                },
                0.3
            ),
            Err(SamplingError::InvalidLocallyTypicalTau)
        );
        assert_eq!(
            sample_with_repetition(
                &[1.0],
                SamplingConfig::default(),
                0.3,
                &[0],
                RepetitionPenaltyConfig {
                    frequency_penalty: -0.1,
                    ..RepetitionPenaltyConfig::default()
                }
            ),
            Err(SamplingError::InvalidFrequencyPenalty)
        );
        assert_eq!(
            sample_with_repetition(
                &[1.0],
                SamplingConfig::default(),
                0.3,
                &[0],
                RepetitionPenaltyConfig {
                    presence_penalty: -0.1,
                    ..RepetitionPenaltyConfig::default()
                }
            ),
            Err(SamplingError::InvalidPresencePenalty)
        );
        assert_eq!(
            sample_with_repetition(
                &[1.0, 1.0],
                SamplingConfig::default(),
                0.3,
                &[],
                RepetitionPenaltyConfig {
                    newline_penalty: Some(NewlinePenalty {
                        token_id: 1,
                        penalty: -0.1
                    }),
                    ..RepetitionPenaltyConfig::default()
                }
            ),
            Err(SamplingError::InvalidNewlinePenalty)
        );
        assert_eq!(
            sample(&[1.0], SamplingConfig::default(), 1.0),
            Err(SamplingError::InvalidRandom)
        );
        assert_eq!(
            sample_mirostat(
                &[1.0],
                1.0,
                MirostatConfig {
                    tau: 0.0,
                    eta: 0.1,
                    mu: 1.0
                },
                0.3
            ),
            Err(SamplingError::InvalidMirostat)
        );
    }

    #[test]
    fn frequency_penalty_discourages_frequent_tokens() {
        let token = sample_with_repetition(
            &[4.0, 3.9, 3.8],
            SamplingConfig {
                temperature: 0.3,
                ..SamplingConfig::default()
            },
            0.7,
            &[0, 0, 0, 0],
            RepetitionPenaltyConfig {
                frequency_penalty: 0.6,
                ..RepetitionPenaltyConfig::default()
            },
        )
        .expect("sampling should succeed");
        assert_ne!(token, 0);
    }

    #[test]
    fn presence_penalty_discourages_seen_tokens() {
        let token = sample_with_repetition(
            &[3.0, 2.9, 2.8],
            SamplingConfig {
                temperature: 0.3,
                ..SamplingConfig::default()
            },
            0.6,
            &[0],
            RepetitionPenaltyConfig {
                presence_penalty: 1.0,
                ..RepetitionPenaltyConfig::default()
            },
        )
        .expect("sampling should succeed");
        assert_ne!(token, 0);
    }

    #[test]
    fn newline_penalty_reduces_newline_token_choice() {
        let token = sample_with_repetition(
            &[5.0, 4.9],
            SamplingConfig {
                temperature: 0.2,
                ..SamplingConfig::default()
            },
            0.4,
            &[],
            RepetitionPenaltyConfig {
                newline_penalty: Some(NewlinePenalty {
                    token_id: 0,
                    penalty: 2.0,
                }),
                ..RepetitionPenaltyConfig::default()
            },
        )
        .expect("sampling should succeed");
        assert_eq!(token, 1);
    }

    #[test]
    fn grammar_constraint_rejects_unknown_start_and_symbols() {
        let missing_start = GrammarConstraint::new("S", HashMap::new());
        assert_eq!(missing_start, Err(SamplingError::InvalidGrammarConstraint));

        let invalid_symbol = GrammarConstraint::new(
            "S",
            HashMap::from([(
                "S".to_string(),
                vec![vec![GrammarSymbol::NonTerminal("MISSING".to_string())]],
            )]),
        );
        assert_eq!(invalid_symbol, Err(SamplingError::InvalidGrammarConstraint));
    }

    #[test]
    fn grammar_constraint_filters_candidate_tokens() {
        let grammar = two_digit_grammar();
        let token = sample_with_repetition_and_grammar(
            &[5.0, 4.0, 10.0],
            SamplingConfig::default(),
            0.4,
            &[],
            RepetitionPenaltyConfig::default(),
            &[0],
            Some(&grammar),
        )
        .expect("sampling should succeed");
        assert!(token <= 1);
    }

    #[test]
    fn grammar_constraint_errors_when_no_tokens_are_valid() {
        let grammar = two_digit_grammar();
        let err = sample_with_repetition_and_grammar(
            &[3.0, 2.0],
            SamplingConfig::default(),
            0.2,
            &[],
            RepetitionPenaltyConfig::default(),
            &[0, 1],
            Some(&grammar),
        );
        assert_eq!(err, Err(SamplingError::NoValidGrammarToken));
    }

    #[test]
    fn speculative_decode_accepts_all_draft_tokens_and_samples_one_more() {
        let result = speculative_decode(
            &[1, 2],
            &[vec![0.0, 10.0, 0.0], vec![0.0, 0.0, 10.0]],
            &[
                vec![0.0, 10.0, 0.0],
                vec![0.0, 0.0, 10.0],
                vec![10.0, 0.0, 0.0],
            ],
            SamplingConfig::default(),
            &[0.2, 0.3, 0.2],
        )
        .expect("speculative decode should succeed");

        assert_eq!(result.tokens, vec![1, 2, 0]);
        assert_eq!(result.accepted_draft_tokens, 2);
        assert!(!result.used_residual_fallback);
    }

    #[test]
    fn speculative_decode_rejects_and_samples_from_residual() {
        let result = speculative_decode(
            &[0],
            &[vec![10.0, 0.0, 0.0]],
            &[vec![0.0, 10.0, 0.0], vec![0.0, 0.0, 10.0]],
            SamplingConfig::default(),
            &[0.9, 0.1],
        )
        .expect("speculative decode should succeed");

        assert_eq!(result.tokens, vec![1]);
        assert_eq!(result.accepted_draft_tokens, 0);
        assert!(result.used_residual_fallback);
    }

    #[test]
    fn speculative_decode_rejects_invalid_lengths() {
        let err = speculative_decode(
            &[1],
            &[],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            SamplingConfig::default(),
            &[0.2, 0.3],
        );
        assert_eq!(err, Err(SamplingError::InvalidSpeculativeInputs));
    }

    #[test]
    fn beam_search_returns_best_sequence() {
        let result = beam_search(&[vec![2.0, 1.0], vec![0.1, 3.0], vec![4.0, 0.2]], 2, None)
            .expect("beam search should succeed");
        assert_eq!(result.tokens, vec![0, 1, 0]);
        assert!(result.score.is_finite());
    }

    #[test]
    fn beam_search_stops_when_all_beams_hit_eos() {
        let result = beam_search(
            &[vec![0.1, 3.0], vec![10.0, 0.0], vec![10.0, 0.0]],
            2,
            Some(1),
        )
        .expect("beam search should succeed");
        assert_eq!(result.tokens, vec![1]);
    }

    #[test]
    fn beam_search_rejects_invalid_inputs() {
        assert_eq!(
            beam_search(&[vec![1.0]], 0, None),
            Err(SamplingError::InvalidBeamWidth)
        );
        assert_eq!(
            beam_search(&[], 1, None),
            Err(SamplingError::InvalidBeamSearchInputs)
        );
        assert_eq!(
            beam_search(&[vec![]], 1, None),
            Err(SamplingError::InvalidBeamSearchInputs)
        );
    }
}
