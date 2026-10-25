//! Decoy sampling, the zeta prior, stacking functions, cosine and ranking.

use super::Example;
use phistd::{cosine, mix64, Function};

/// Draws decoys in proportion to `weight^alpha` — frequency-proportional
/// negative sampling.
pub struct FrequencySampler {
    items: Vec<usize>,
    cumulative: Vec<f64>,
}

impl FrequencySampler {
    /// `weights[i]` is the weight of `items[i]`; `alpha` flattens (< 1) or
    /// sharpens (> 1) the distribution.
    pub fn new(items: Vec<usize>, weights: &[f64], alpha: f64) -> Self {
        assert_eq!(items.len(), weights.len());
        let mut cumulative = Vec::with_capacity(items.len());
        let mut acc = 0.0;
        for w in weights {
            acc += w.max(0.0).powf(alpha);
            cumulative.push(acc);
        }
        Self { items, cumulative }
    }

    pub fn draw(&self, target: usize, mut r: u64) -> usize {
        let total = *self.cumulative.last().unwrap_or(&1.0);
        for _ in 0..16 {
            let u = (mix64(r) >> 11) as f64 / (1u64 << 53) as f64 * total;
            let i = self.cumulative.partition_point(|c| *c <= u).min(self.items.len() - 1);
            let d = self.items[i];
            if d != target {
                return d;
            }
            r = mix64(r);
        }
        self.items[0]
    }
}

/// The Riemann-zeta spectral prior: mode `m ≥ 1` weighted by `m^{-s} / ζ(s)`
/// (mode 0 keeps weight 1). Larger `s` is a stronger smoothness prior. It is
/// the zeta function used as a spectral weighting and nothing deeper. Use it
/// with [`Fno::scale_modes`](super::Fno::scale_modes). For `s > 1`, `ζ(s)` is
/// summed over the first 200 000 terms; for `s ≤ 1` the un-normalised
/// `m^{-s}` is returned.
pub fn zeta_weights(modes: usize, s: f64) -> Vec<f64> {
    let norm = match s > 1.0 {
        true => (1..=200_000).map(|m| (m as f64).powf(-s)).sum::<f64>(),
        false => 1.0,
    };
    (0..modes).map(|m| if m == 0 { 1.0 } else { (m as f64).powf(-s) / norm }).collect()
}

/// Mean reciprocal rank of each example's target among `pool` under cosine
/// to the prediction, plus hit@10.
pub fn rank(table: &[Function], preds: &[Function], examples: &[Example], pool: &[usize]) -> (f64, f64) {
    let mut mrr = 0.0;
    let mut hit10 = 0usize;
    for (y, ex) in preds.iter().zip(examples) {
        let target = cosine(y, &table[ex.target]);
        let above = pool
            .iter()
            .filter(|&&w| w != ex.target)
            .filter(|&&w| cosine(y, &table[w]) > target)
            .count();
        mrr += 1.0 / (1 + above) as f64;
        if above < 10 {
            hit10 += 1;
        }
    }
    let n = examples.len().max(1) as f64;
    (mrr / n, hit10 as f64 / n)
}
