use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub max_tokens: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: None,
            top_k: None,
            max_tokens: 256,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: rand::random(),
        }
    }
}

pub struct LogitsSampler {
    rng: StdRng,
}

impl LogitsSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn sample(&mut self, logits: &[f32], params: &SamplingParams) -> u32 {
        if params.temperature <= 0.0 {
            return arg_max(logits);
        }

        let scaled: Vec<f32> = logits
            .iter()
            .map(|&l| l / params.temperature as f32)
            .collect();
        let mut probs = softmax(&scaled);

        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

        if let Some(k) = params.top_k {
            for &idx in indices.iter().skip(k) {
                probs[idx] = 0.0;
            }
        }

        if let Some(p) = params.top_p {
            let mut cumulative = 0.0f32;
            let mut cutoff = indices.len();
            for (rank, &idx) in indices.iter().enumerate() {
                cumulative += probs[idx];
                if cumulative as f64 >= p {
                    cutoff = rank + 1;
                    break;
                }
            }
            for &idx in indices.iter().skip(cutoff) {
                probs[idx] = 0.0;
            }
        }

        let sum: f32 = probs.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return arg_max(logits);
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }

        let target: f32 = self.rng.gen_range(0.0..1.0);
        let mut cumulative = 0.0f32;
        for (idx, &p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= target {
                return idx as u32;
            }
        }
        (probs.len() - 1) as u32
    }
}

fn arg_max(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

pub fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context: &[u32]) {
    if penalty == 1.0 {
        return;
    }
    use std::collections::HashSet;
    let seen: HashSet<u32> = context.iter().copied().collect();
    for &token in &seen {
        let idx = token as usize;
        if idx >= logits.len() {
            continue;
        }
        if logits[idx] >= 0.0 {
            logits[idx] /= penalty;
        } else {
            logits[idx] *= penalty;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(temperature: f64) -> SamplingParams {
        SamplingParams {
            temperature,
            top_p: None,
            top_k: None,
            max_tokens: 32,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 42,
        }
    }

    #[test]
    fn greedy_picks_argmax() {
        let logits = vec![0.1, 5.0, -2.0, 3.0];
        let mut sampler = LogitsSampler::new(42);
        let token = sampler.sample(&logits, &params(0.0));
        assert_eq!(token, 1);
    }

    #[test]
    fn temperature_sampling_is_deterministic_for_fixed_seed() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let mut a = LogitsSampler::new(7);
        let mut b = LogitsSampler::new(7);
        let ta = a.sample(&logits, &params(1.0));
        let tb = b.sample(&logits, &params(1.0));
        assert_eq!(ta, tb);
    }

    #[test]
    fn top_k_only_considers_top_k_tokens() {
        let logits = vec![0.0, 0.0, 0.0, 100.0];
        let mut sampler = LogitsSampler::new(1);
        let mut p = params(1.0);
        p.top_k = Some(1);
        let token = sampler.sample(&logits, &p);
        assert_eq!(token, 3);
    }

    #[test]
    fn repeat_penalty_lowers_logit_of_seen_token() {
        let mut logits = vec![1.0, 1.0, 1.0];
        apply_repeat_penalty(&mut logits, 1.5, &[1]);
        assert!(logits[1] < logits[0]);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[2], 1.0);
    }

    #[test]
    fn repeat_penalty_handles_negative_logits() {
        let mut logits = vec![-1.0, -1.0];
        apply_repeat_penalty(&mut logits, 1.5, &[0]);
        // negative logit penalized by multiplying (more negative), so logits[0] < logits[1]
        assert!(logits[0] < logits[1]);
    }
}
