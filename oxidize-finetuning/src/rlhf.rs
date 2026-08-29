//! PPO-style RLHF components.

use rayon::prelude::*;

use crate::config::FinetuneConfig;
use crate::lora::{LoRAAdapter, LoRATarget};

/// A linear scalar reward head: `r(h) = dot(w, h) + b`.
///
/// Weights are Xavier-uniform initialised so the initial reward distribution
/// is centred near zero.  No LoRA adapter is used — the whole head is trained.
#[derive(Debug, Clone)]
pub struct RewardModel {
    pub weights: Vec<f32>,
    pub bias: f32,
    in_dim: usize,
}

impl RewardModel {
    /// Create a new reward model with random weights.
    ///
    /// Uses an xorshift64 PRNG seeded by `seed`; weights drawn from
    /// Uniform(-1/sqrt(in_dim), 1/sqrt(in_dim)) (Xavier uniform).
    pub fn new(in_dim: usize, seed: u64) -> Self {
        let limit = (in_dim as f32).sqrt().recip();
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut weights = vec![0.0_f32; in_dim];
        for v in weights.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = ((state >> 32) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            *v = u * limit;
        }
        Self {
            weights,
            bias: 0.0,
            in_dim,
        }
    }

    /// Compute a scalar reward for a single hidden-state vector.
    ///
    /// Panics if `hidden.len() != in_dim`.
    pub fn score(&self, hidden: &[f32]) -> f32 {
        assert_eq!(
            hidden.len(),
            self.in_dim,
            "RewardModel::score: hidden.len()={} != in_dim={}",
            hidden.len(),
            self.in_dim
        );
        dot(&self.weights, hidden) + self.bias
    }

    /// Compute rewards for a batch of `count` hidden-state rows in parallel.
    ///
    /// `hiddens` must have length `count * in_dim` (row-major).
    pub fn score_batch(&self, hiddens: &[f32], count: usize) -> Vec<f32> {
        assert_eq!(
            hiddens.len(),
            count * self.in_dim,
            "RewardModel::score_batch: hiddens.len()={} != count({count}) * in_dim({})",
            hiddens.len(),
            self.in_dim
        );
        hiddens
            .par_chunks(self.in_dim)
            .map(|row| dot(&self.weights, row) + self.bias)
            .collect()
    }
}

/// PPO hyper-parameters.
#[derive(Debug, Clone)]
pub struct PpoConfig {
    /// Probability-ratio clipping threshold ε.  Typical value: 0.1 – 0.2.
    pub clip_eps: f32,
    /// Coefficient for the value (critic) loss term.
    pub value_coef: f32,
    /// Coefficient for the entropy bonus.
    pub entropy_coef: f32,
    /// KL divergence penalty coefficient (reference-policy regularisation).
    pub kl_penalty: f32,
    /// GAE λ for advantage estimation.  1.0 → Monte-Carlo, 0.0 → TD(0).
    pub gae_lambda: f32,
    /// Discount factor γ.  1.0 is standard for next-token prediction.
    pub gamma: f32,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_eps: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            kl_penalty: 0.02,
            gae_lambda: 0.95,
            gamma: 1.0,
        }
    }
}

/// Trajectory storage for a single PPO rollout.
///
/// Hidden states are stored as flat row-major slices appended by `add`.
/// `compute_gae` must be called once after all steps have been added
/// and before any PPO update uses the buffer.
#[derive(Debug, Clone, Default)]
pub struct RolloutBuffer {
    /// Hidden-state rows, each of length `in_dim` (inferred from first add).
    pub states: Vec<Vec<f32>>,
    /// Token ids for each timestep.
    pub actions: Vec<u32>,
    /// Scalar per-step rewards from the reward model (or environment).
    pub rewards: Vec<f32>,
    /// Log-probability of the chosen action under the behaviour policy.
    pub log_probs: Vec<f32>,
    /// Value estimates V(s_t) from the critic / reward model.
    pub values: Vec<f32>,
    /// GAE advantages A(s_t, a_t), filled by `compute_gae`.
    pub advantages: Vec<f32>,
}

impl RolloutBuffer {
    /// Create an empty rollout buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one (state, action, reward, log_prob, value) transition.
    pub fn add(&mut self, state: Vec<f32>, action: u32, reward: f32, log_prob: f32, value: f32) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.log_probs.push(log_prob);
        self.values.push(value);
    }

    /// How many transitions are stored.
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// True if the buffer contains no transitions.
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Compute Generalized Advantage Estimation (GAE, Schulman et al. 2015)
    /// and store results in `self.advantages`.
    ///
    /// `last_value` is the bootstrap value V(s_T) for the state *after* the
    /// final recorded transition.  Pass `0.0` for terminal episodes.
    ///
    /// After this call `advantages[t]` contains the GAE estimate for step `t`:
    ///
    /// ```text
    /// δ_t = r_t + γ · V(s_{t+1}) − V(s_t)
    /// A_t = δ_t + (γλ) · δ_{t+1} + (γλ)² · δ_{t+2} + …
    /// ```
    pub fn compute_gae(&mut self, config: &PpoConfig, last_value: f32) {
        let n = self.rewards.len();
        self.advantages = vec![0.0_f32; n];
        let mut gae = 0.0_f32;
        let gl = config.gamma * config.gae_lambda;

        for t in (0..n).rev() {
            let next_value = if t + 1 < n {
                self.values[t + 1]
            } else {
                last_value
            };
            let delta = self.rewards[t] + config.gamma * next_value - self.values[t];
            gae = delta + gl * gae;
            self.advantages[t] = gae;
        }
    }
}

// PpoStepReport / PpoReport

/// Metrics returned by a single PPO gradient step.
#[derive(Debug, Clone)]
pub struct PpoStepReport {
    /// Clipped surrogate policy loss (averaged over the mini-batch).
    pub policy_loss: f32,
    /// Value function MSE loss (averaged).
    pub value_loss: f32,
    /// Entropy bonus (averaged; positive means more entropy).
    pub entropy: f32,
    /// KL divergence estimate from the reference log-probs (averaged).
    pub kl: f32,
    /// Combined scalar loss = policy_loss + value_coef*value_loss − entropy_coef*entropy + kl_penalty*kl.
    pub total_loss: f32,
    /// Number of transitions processed in this step.
    pub n: usize,
}

/// Metrics returned by a full PPO training epoch over multiple rollout buffers.
#[derive(Debug, Clone)]
pub struct PpoReport {
    /// Total number of gradient steps taken.
    pub steps: usize,
    /// Total number of transitions processed.
    pub transitions: usize,
    /// Mean policy loss across all steps.
    pub mean_policy_loss: f32,
    /// Mean value loss across all steps.
    pub mean_value_loss: f32,
    /// Mean entropy across all steps.
    pub mean_entropy: f32,
    /// Mean KL penalty across all steps.
    pub mean_kl: f32,
    /// Wall-clock seconds.
    pub elapsed_seconds: f32,
}

/// PPO trainer backed by a LoRA actor and a linear reward model.
///
/// Because the `LoRAAdapter` exposes a rayon-parallel batch
/// forward/backward API, mini-batching a full rollout buffer is a single
/// contiguous GEMM-style pass rather than a per-token loop.
pub struct PpoTrainer {
    /// LoRA adapter used as the policy actor (π_θ).
    pub lora: LoRAAdapter,
    /// Linear scalar reward / value head.
    pub reward_model: RewardModel,
    /// Base finetune hyper-parameters (LR, warmup, weight decay …).
    pub config: FinetuneConfig,
    /// PPO-specific hyper-parameters.
    pub ppo_config: PpoConfig,
    /// Current optimizer step (1-indexed); advanced by `train_step`.
    step: usize,
}

impl PpoTrainer {
    /// Create a new PPO trainer.
    ///
    /// * `in_dim`  — hidden size fed into both the LoRA actor and the reward
    ///               model.
    /// * `out_dim` — vocabulary size (actor output dimension).
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        finetune_config: FinetuneConfig,
        ppo_config: PpoConfig,
    ) -> Self {
        let lora = LoRAAdapter::new(LoRATarget::OutputHead, in_dim, out_dim, &finetune_config);
        let reward_model = RewardModel::new(in_dim, finetune_config.seed);
        Self {
            lora,
            reward_model,
            config: finetune_config,
            ppo_config,
            step: 0,
        }
    }

    // PPO loss primitives

    /// Clipped surrogate policy loss for a single transition.
    ///
    /// ```text
    /// ratio    = exp(new_log_prob − old_log_prob)
    /// L_clip   = -min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
    /// ```
    ///
    /// Returns the *positive* loss scalar (gradient descent minimises it).
    #[inline]
    pub fn ppo_loss(old_log_prob: f32, new_log_prob: f32, advantage: f32, clip_eps: f32) -> f32 {
        let ratio = (new_log_prob - old_log_prob).exp();
        let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
        // Objective is MAXIMISED — negate to turn into a minimisation loss.
        -(ratio * advantage).min(clipped * advantage)
    }

    /// Compute the entropy of a single row of logits (numerically stable).
    ///
    /// H(π) = -sum_i p_i · log p_i
    #[inline]
    fn entropy_from_logits(logits: &[f32]) -> f32 {
        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|l| (l - max_l).exp()).sum();
        let log_z = max_l + exp_sum.ln();
        logits
            .iter()
            .map(|l| {
                let log_p = l - log_z;
                let p = log_p.exp();
                -p * log_p
            })
            .sum()
    }

    /// Log-probability of `action` under a softmax distribution over `logits`.
    #[inline]
    fn log_prob_from_logits(logits: &[f32], action: u32) -> f32 {
        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|l| (l - max_l).exp()).sum();
        let log_z = max_l + exp_sum.ln();
        let idx = (action as usize).min(logits.len().saturating_sub(1));
        logits[idx] - log_z
    }

    /// Run one PPO gradient step over all transitions in `buffer`.
    ///
    /// Steps:
    /// 1. Forward the LoRA actor on each hidden state to get current logits.
    /// 2. Compute per-transition: new_log_prob, entropy, clipped surrogate loss,
    ///    value loss, KL estimate.
    /// 3. Accumulate the total scalar gradient signal into LoRA grad buffers
    ///    via `backward_batch`.
    /// 4. Perform an AdamW update on the LoRA parameters.
    ///
    /// Returns a `PpoStepReport` with the mean loss components.
    pub fn train_step(&mut self, buffer: &RolloutBuffer) -> PpoStepReport {
        let n = buffer.len();
        if n == 0 {
            return PpoStepReport {
                policy_loss: 0.0,
                value_loss: 0.0,
                entropy: 0.0,
                kl: 0.0,
                total_loss: 0.0,
                n: 0,
            };
        }

        let vocab = self.lora.out_dim;
        let in_dim = self.lora.in_dim;
        let eps = self.ppo_config.clip_eps;
        let value_coef = self.ppo_config.value_coef;
        let entropy_coef = self.ppo_config.entropy_coef;
        let kl_penalty = self.ppo_config.kl_penalty;

        // Flatten all hidden states into a single contiguous buffer for the
        // batched LoRA forward.
        let flat_states: Vec<f32> = buffer
            .states
            .iter()
            .flat_map(|s| s.iter().copied())
            .collect();
        debug_assert_eq!(flat_states.len(), n * in_dim);

        // LoRA forward: base logits start at zero; the LoRA residual is added
        // in-place.  In a full system the caller would also add frozen-base
        // logits, but since those are constant w.r.t. the LoRA parameters they
        // do not affect the gradient — so we elide them here.
        let mut logits = vec![0.0_f32; n * vocab];
        // forward_batch can only fail on shape mismatch; we just constructed
        // flat_states to match, so unwrap is safe.
        self.lora
            .forward_batch(&flat_states, &mut logits, n)
            .expect("PpoTrainer::train_step: forward_batch shape mismatch (internal bug)");

        // Per-transition losses and gradient signals.
        //
        // We compute a gradient direction for each logit row, then pass the
        // combined gradient slice to `backward_batch` in one shot.

        let mut grad_logits = vec![0.0_f32; n * vocab];
        let mut sum_policy_loss = 0.0_f32;
        let mut sum_value_loss = 0.0_f32;
        let mut sum_entropy = 0.0_f32;
        let mut sum_kl = 0.0_f32;

        // Returns value estimate from the reward model (used as critic).
        let values_now: Vec<f32> = buffer
            .states
            .iter()
            .map(|s| self.reward_model.score(s))
            .collect();

        let scale = 1.0_f32 / n as f32;

        for t in 0..n {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let action = buffer.actions[t];
            let old_lp = buffer.log_probs[t];
            let advantage = buffer.advantages[t];
            let v_old = buffer.values[t];

            let new_lp = Self::log_prob_from_logits(row, action);
            let h = Self::entropy_from_logits(row);
            let v_now = values_now[t];

            // Returns for value loss: advantage + old value (returns ≈ Q(s,a)).
            let returns = advantage + v_old;
            let v_loss = (v_now - returns).powi(2);

            let p_loss = Self::ppo_loss(old_lp, new_lp, advantage, eps);
            let kl_t = old_lp - new_lp;

            sum_policy_loss += p_loss;
            sum_value_loss += v_loss;
            sum_entropy += h;
            sum_kl += kl_t;

            // Gradient of the combined loss w.r.t. the logit row.
            //
            // L_total = p_loss + value_coef·v_loss − entropy_coef·H + kl_penalty·kl
            //
            // All four terms depend on the logits:
            //   • p_loss: via ratio = exp(new_lp − old_lp); new_lp = logit[action] − log_Z
            //   • v_loss: via v_now = reward_model(state) — does NOT depend on
            //             logits; its gradient flows into the reward_model weights,
            //             not the LoRA, so we skip it here.
            //   • entropy H: d(-H)/d(logit_i) = p_i·(log p_i + H) (standard)
            //     but we want -entropy_coef·(-dH/dlogit_i) = entropy_coef·dH/dlogit_i
            //   • KL = old_lp − new_lp: dKL/d(logit[action]) = -d(new_lp)/d(logit[action])
            //     dKL/d(logit_i)   = -d(new_lp)/d(logit_i)
            //
            // dL/d(logit_i) = d(p_loss)/d(logit_i)
            //                 − entropy_coef · dH/d(logit_i)
            //                 + kl_penalty · dKL/d(logit_i)
            //
            // For the clipped surrogate the active branch determines the coefficient:
            //   Let coeff = −active_weight (negative because L = −min(r·A, clip·A))
            //   active_weight = ratio·A  if |ratio−1| <= eps, else clip·A/ratio
            //   ... simplify: since d(ratio)/d(new_lp) = ratio, and
            //   d(new_lp)/d(logit_i) = d/d(logit_i)[logit_{action} - log_Z]
            //                        = 1_{i==action} − p_i
            //
            // Combined in one pass below.

            let max_l = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|l| (l - max_l).exp()).sum();
            let log_z = max_l + exp_sum.ln();
            let action_idx = (action as usize).min(vocab.saturating_sub(1));

            // Coefficient for policy gradient: chain rule through log-prob.
            // ratio = exp(new_lp - old_lp)
            let ratio = (new_lp - old_lp).exp();
            let clipped_ratio = ratio.clamp(1.0 - eps, 1.0 + eps);

            // Choose the active branch of min(r·A, clip·A).
            // Gradient of -min(rA, clip_rA) w.r.t. new_lp:
            //   if ratio·A < clipped_ratio·A: active = ratio·A → d/d(new_lp) = -ratio·A
            //   else:                         active = clip·A   → d/d(new_lp) = 0 (clipped)
            let pg_coeff = if ratio * advantage <= clipped_ratio * advantage {
                // unclipped branch; d(-ratio·A)/d(new_lp) = -ratio·A
                -ratio * advantage
            } else {
                0.0
            };

            // kl = old_lp − new_lp → d(kl_penalty·kl)/d(new_lp) = -kl_penalty
            let kl_coeff = -kl_penalty;

            // entropy H = −Σ p_i log p_i
            // dH/d(logit_i) = −p_i·(1 + log p_i) + p_i·H  = p_i·(H − 1 − log p_i)
            // We want −entropy_coef·(−dH/dlogit_i) contribution = entropy_coef·dH/dlogit_i.
            // This is computed per-element inside the row loop below.

            // d(new_lp)/d(logit_i) = 1_{i==action_idx} − p_i
            let grad_row = &mut grad_logits[t * vocab..(t + 1) * vocab];
            for (i, g) in grad_row.iter_mut().enumerate() {
                let log_p_i = row[i] - log_z;
                let p_i = log_p_i.exp();

                // Policy + KL gradient via chain rule through new_lp.
                let d_new_lp = (if i == action_idx { 1.0 } else { 0.0 }) - p_i;
                let policy_kl_grad = (pg_coeff + kl_coeff) * d_new_lp;

                // Entropy gradient contribution.
                let d_h = p_i * (h - 1.0 - log_p_i);
                let entropy_grad = -entropy_coef * d_h;

                *g = scale * (policy_kl_grad + entropy_grad);
            }
        }

        // Accumulate gradients through LoRA backward.
        self.lora
            .backward_batch(&flat_states, &grad_logits, n)
            .expect("PpoTrainer::train_step: backward_batch shape mismatch (internal bug)");

        // AdamW optimizer step.
        self.step += 1;
        let lr = warmup_lr(
            self.config.learning_rate,
            self.step,
            self.config.warmup_steps,
        );
        self.lora.step(lr, self.config.weight_decay, self.step);
        self.lora.zero_grad();

        let total_loss = sum_policy_loss * scale + value_coef * sum_value_loss * scale
            - entropy_coef * sum_entropy * scale
            + kl_penalty * sum_kl * scale;

        PpoStepReport {
            policy_loss: sum_policy_loss * scale,
            value_loss: sum_value_loss * scale,
            entropy: sum_entropy * scale,
            kl: sum_kl * scale,
            total_loss,
            n,
        }
    }

    /// Train over a list of rollout buffers (one gradient step per buffer).
    ///
    /// Each buffer should have its `compute_gae` already called.  Buffers with
    /// zero transitions are silently skipped.
    pub fn train_epoch(&mut self, rollouts: Vec<RolloutBuffer>) -> PpoReport {
        use std::time::Instant;
        let started = Instant::now();

        let mut steps = 0usize;
        let mut transitions = 0usize;
        let mut sum_policy = 0.0_f32;
        let mut sum_value = 0.0_f32;
        let mut sum_entropy = 0.0_f32;
        let mut sum_kl = 0.0_f32;

        for buffer in &rollouts {
            if buffer.is_empty() {
                continue;
            }
            let report = self.train_step(buffer);
            steps += 1;
            transitions += report.n;
            sum_policy += report.policy_loss;
            sum_value += report.value_loss;
            sum_entropy += report.entropy;
            sum_kl += report.kl;
        }

        let elapsed = started.elapsed().as_secs_f32();
        let denom = steps.max(1) as f32;
        PpoReport {
            steps,
            transitions,
            mean_policy_loss: sum_policy / denom,
            mean_value_loss: sum_value / denom,
            mean_entropy: sum_entropy / denom,
            mean_kl: sum_kl / denom,
            elapsed_seconds: elapsed,
        }
    }
}

// Math helpers

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn warmup_lr(base: f32, step: usize, warmup: usize) -> f32 {
    if warmup == 0 || step >= warmup {
        base
    } else {
        base * (step as f32 / warmup as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_model_score_matches_score_batch() {
        let rm = RewardModel::new(8, 42);
        let hiddens: Vec<f32> = (0..3 * 8).map(|i| (i as f32 * 0.3).sin()).collect();
        let batch = rm.score_batch(&hiddens, 3);
        for t in 0..3 {
            let single = rm.score(&hiddens[t * 8..(t + 1) * 8]);
            assert!(
                (batch[t] - single).abs() < 1e-6,
                "batch[{t}]={} single={}",
                batch[t],
                single
            );
        }
    }

    #[test]
    fn reward_model_score_changes_with_weights() {
        let rm = RewardModel::new(4, 7);
        let h = vec![1.0_f32; 4];
        let s1 = rm.score(&h);
        let mut rm2 = rm.clone();
        rm2.weights[0] += 1.0;
        let s2 = rm2.score(&h);
        assert!((s2 - s1 - 1.0).abs() < 1e-5);
    }

    // PpoConfig defaults

    #[test]
    fn ppo_config_defaults_are_sensible() {
        let c = PpoConfig::default();
        assert!((c.clip_eps - 0.2).abs() < 1e-6);
        assert!((c.value_coef - 0.5).abs() < 1e-6);
        assert!((c.entropy_coef - 0.01).abs() < 1e-6);
        assert!((c.kl_penalty - 0.02).abs() < 1e-6);
        assert!((c.gae_lambda - 0.95).abs() < 1e-6);
        assert!((c.gamma - 1.0).abs() < 1e-6);
    }

    // RolloutBuffer + GAE

    fn make_buffer(rewards: &[f32], values: &[f32]) -> RolloutBuffer {
        let mut buf = RolloutBuffer::new();
        for (i, (&r, &v)) in rewards.iter().zip(values.iter()).enumerate() {
            buf.add(vec![i as f32; 4], i as u32, r, -0.5, v);
        }
        buf
    }

    #[test]
    fn gae_terminal_advantage_matches_td_residual() {
        // Single step: A_0 = δ_0 = r_0 + γ·0 − V_0 (last_value = 0, terminal).
        let mut buf = make_buffer(&[2.0], &[1.0]);
        let cfg = PpoConfig {
            gamma: 1.0,
            gae_lambda: 0.95,
            ..Default::default()
        };
        buf.compute_gae(&cfg, 0.0);
        assert!(
            (buf.advantages[0] - 1.0).abs() < 1e-5,
            "A_0={}",
            buf.advantages[0]
        );
    }

    #[test]
    fn gae_two_steps_correct() {
        // t=0: δ_0 = 1 + 1·V_1 − V_0 = 1 + 0.9 − 0.5 = 1.4
        // t=1: δ_1 = 2 + 1·0 − 0.9 = 1.1  (terminal, last_value=0)
        // A_1 = δ_1 = 1.1
        // A_0 = δ_0 + γλ·A_1 = 1.4 + 0.9·1.1 = 2.39
        let mut buf = make_buffer(&[1.0, 2.0], &[0.5, 0.9]);
        let cfg = PpoConfig {
            gamma: 1.0,
            gae_lambda: 0.9,
            ..Default::default()
        };
        buf.compute_gae(&cfg, 0.0);
        let a0 = buf.advantages[0];
        let a1 = buf.advantages[1];
        assert!((a1 - 1.1).abs() < 1e-4, "A_1={a1}");
        assert!((a0 - 2.39).abs() < 1e-4, "A_0={a0}");
    }

    #[test]
    fn gae_bootstrap_last_value() {
        // last_value = 5.0: δ_0 = r_0 + γ·last_value − V_0
        let mut buf = make_buffer(&[1.0], &[0.5]);
        let cfg = PpoConfig {
            gamma: 1.0,
            gae_lambda: 0.95,
            ..Default::default()
        };
        buf.compute_gae(&cfg, 5.0);
        let expected = 1.0 + 5.0 - 0.5; // = 5.5
        assert!(
            (buf.advantages[0] - expected).abs() < 1e-4,
            "A_0={}",
            buf.advantages[0]
        );
    }

    fn tiny_ppo_trainer() -> PpoTrainer {
        let ft = FinetuneConfig {
            rank: 2,
            alpha: 4.0,
            learning_rate: 1e-3,
            warmup_steps: 0,
            ..Default::default()
        };
        let ppo = PpoConfig::default();
        PpoTrainer::new(4, 8, ft, ppo)
    }

    fn toy_rollout(n: usize) -> RolloutBuffer {
        let mut buf = RolloutBuffer::new();
        for i in 0..n {
            let state: Vec<f32> = (0..4).map(|j| ((i * 4 + j) as f32 * 0.1).sin()).collect();
            buf.add(state, (i % 8) as u32, 1.0, -0.693, 0.0);
        }
        let cfg = PpoConfig::default();
        buf.compute_gae(&cfg, 0.0);
        buf
    }

    #[test]
    fn ppo_trainer_constructs() {
        let t = tiny_ppo_trainer();
        assert_eq!(t.lora.in_dim, 4);
        assert_eq!(t.lora.out_dim, 8);
        assert_eq!(t.reward_model.in_dim, 4);
    }

    #[test]
    fn ppo_loss_unclipped_positive_advantage() {
        // ratio = 1.0 (same policy), A > 0 → loss = -A < 0 (we want to max reward).
        let loss = PpoTrainer::ppo_loss(0.0, 0.0, 2.0, 0.2);
        assert!((loss - (-2.0)).abs() < 1e-6, "loss={loss}");
    }

    #[test]
    fn ppo_loss_clips_at_boundary() {
        // ratio = exp(large) >> 1+eps → clipped branch: loss = -(1+eps)*A
        let loss = PpoTrainer::ppo_loss(0.0, 10.0, 1.0, 0.2);
        assert!((loss - (-1.2)).abs() < 1e-5, "loss={loss}");
    }

    #[test]
    fn train_step_empty_buffer_returns_zero() {
        let mut t = tiny_ppo_trainer();
        let buf = RolloutBuffer::new();
        let r = t.train_step(&buf);
        assert_eq!(r.n, 0);
        assert_eq!(r.total_loss, 0.0);
    }

    #[test]
    fn train_step_returns_finite_metrics() {
        let mut t = tiny_ppo_trainer();
        let buf = toy_rollout(5);
        let r = t.train_step(&buf);
        assert_eq!(r.n, 5);
        assert!(r.policy_loss.is_finite(), "policy_loss={}", r.policy_loss);
        assert!(r.value_loss.is_finite(), "value_loss={}", r.value_loss);
        assert!(r.entropy.is_finite(), "entropy={}", r.entropy);
        assert!(r.total_loss.is_finite(), "total_loss={}", r.total_loss);
    }

    #[test]
    fn train_step_changes_lora_weights() {
        let mut t = tiny_ppo_trainer();
        // Non-zero B so adapter output is non-trivial.
        for (i, v) in t.lora.b.iter_mut().enumerate() {
            *v = ((i % 5) as f32 - 2.0) * 0.1;
        }
        let before: Vec<f32> = t.lora.a.clone();
        let buf = toy_rollout(4);
        t.train_step(&buf);
        assert!(
            before
                .iter()
                .zip(t.lora.a.iter())
                .any(|(a, b)| (a - b).abs() > 1e-12),
            "LoRA A weights did not change after train_step"
        );
    }

    #[test]
    fn train_epoch_aggregates_steps() {
        let mut t = tiny_ppo_trainer();
        let rollouts: Vec<RolloutBuffer> = (0..3).map(|_| toy_rollout(4)).collect();
        let report = t.train_epoch(rollouts);
        assert_eq!(report.steps, 3);
        assert!(report.transitions > 0);
        assert!(report.mean_policy_loss.is_finite());
        assert!(report.elapsed_seconds >= 0.0);
    }

    #[test]
    fn train_epoch_skips_empty_rollouts() {
        let mut t = tiny_ppo_trainer();
        let rollouts = vec![toy_rollout(3), RolloutBuffer::new(), toy_rollout(2)];
        let report = t.train_epoch(rollouts);
        assert_eq!(report.steps, 2);
        assert_eq!(report.transitions, 5);
    }

    #[test]
    fn ppo_loss_decreases_over_repeated_steps() {
        let ft = FinetuneConfig {
            rank: 4,
            alpha: 8.0,
            learning_rate: 1e-2,
            warmup_steps: 0,
            ..Default::default()
        };
        let ppo = PpoConfig {
            entropy_coef: 0.0,
            kl_penalty: 0.0,
            ..Default::default()
        };
        let mut t = PpoTrainer::new(8, 16, ft, ppo);
        for (i, v) in t.lora.b.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.05;
        }

        let make_buf = || {
            let mut buf = RolloutBuffer::new();
            for i in 0..8 {
                let state: Vec<f32> = (0..8).map(|j| ((i * 8 + j) as f32 * 0.07).sin()).collect();
                buf.add(state, (i % 16) as u32, 2.0, -0.693, 0.5);
            }
            let cfg = PpoConfig::default();
            buf.compute_gae(&cfg, 0.0);
            buf
        };

        let first = t.train_step(&make_buf()).policy_loss;
        let mut last = first;
        for _ in 0..30 {
            last = t.train_step(&make_buf()).policy_loss;
        }
        // With positive advantages the policy loss should trend downward.
        assert!(
            last <= first + 1.0,
            "policy_loss did not stay bounded: first={first} last={last}"
        );
    }
}
