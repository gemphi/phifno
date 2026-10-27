//! The training epochs: hinge against a decoy, the mixture likelihood, full-table softmax.

use phistd::stack;
use super::{Example, Fno, MixtureExample};
use phistd::{Adam, MixtureTerms};
use phistd::{mix64, Function};

impl Fno {
    /// One epoch of hinge training. `table[i]` is the function of item `i`;
    /// decoys are drawn uniformly from `pool`. Returns the mean loss.
    pub fn train_epoch(
        &mut self,
        table: &[Function],
        examples: &[Example],
        pool: &[usize],
        batch: usize,
        adam: &mut Adam,
        seed: u64,
    ) -> f64 {
        let sampler = |target: usize, r: u64| phistd::decoy(pool, target, r);
        self.train_epoch_with(table, examples, &sampler, batch, adam, seed)
    }

    /// [`Fno::train_epoch`] with decoys drawn by `sampler(target, seed)`.
    ///
    /// A uniform decoy teaches the operator to beat a random item. A decoy
    /// drawn in proportion to frequency teaches it to beat the *likely* items —
    /// which is what a term mixed with a unigram has to do, since the unigram
    /// already knows the frequent ones. See [`FrequencySampler`].
    pub fn train_epoch_with(
        &mut self,
        table: &[Function],
        examples: &[Example],
        sampler: &dyn Fn(usize, u64) -> usize,
        batch: usize,
        adam: &mut Adam,
        seed: u64,
    ) -> f64 {
        let params = self.params();
        let s = self.config.space;
        let mut total = 0.0;
        let mut batches = 0usize;
        let mut order: Vec<usize> = (0..examples.len()).collect();
        order.sort_by_key(|i| mix64(*i as u64 ^ seed));
        for chunk in order.chunks(batch) {
            self.tape.truncate(self.keep);
            let ctx: Vec<Function> = chunk.iter().map(|&i| examples[i].input.clone()).collect();
            let tgt: Vec<Function> = chunk.iter().map(|&i| table[examples[i].target].clone()).collect();
            let dec: Vec<Function> = chunk
                .iter()
                .enumerate()
                .map(|(j, &i)| {
                    let r = mix64(seed ^ (i as u64) << 20 ^ j as u64);
                    table[sampler(examples[i].target, r)].clone()
                })
                .collect();
            let x = self.tape.constant(stack(&ctx, s));
            let t = self.tape.constant(stack(&tgt, s));
            let d = self.tape.constant(stack(&dec, s));
            let y = self.forward(x);
            let sp = self.tape.cosine(y, t);
            let sn = self.tape.cosine(y, d);
            let loss = self.tape.hinge(sp, sn, self.config.margin);
            total += self.tape.value(loss).get(0, 0, 0).re;
            batches += 1;
            self.tape.backward(loss);
            adam.step(&mut self.tape, &params);
            self.mask_modes();
        }
        self.tape.truncate(self.keep);
        total / batches.max(1) as f64
    }

    /// One epoch on the mixture likelihood: the operator's cosine against the
    /// whole table goes through `softmax(β·s)`, is mixed with each position's
    /// other component at `gamma`, and the negative log-likelihood of the true
    /// target is minimised. Every position's `k`, `coef` and `other` are held
    /// constant, so the operator learns to put mass exactly where the other
    /// components leave it missing. Returns the mean loss (a perplexity is
    /// `exp` of it).
    // The mixture's two knobs (gamma, beta) sit beside the same five arguments
    // as `train_epoch_softmax`; a struct would split one call shape in two.
    #[allow(clippy::too_many_arguments)]
    pub fn train_epoch_mixture(
        &mut self,
        table: &[Function],
        examples: &[MixtureExample],
        gamma: f64,
        beta: f64,
        batch: usize,
        adam: &mut Adam,
        seed: u64,
    ) -> f64 {
        let params = self.params();
        let s = self.config.space;
        let mut total = 0.0;
        let mut batches = 0usize;
        let mut order: Vec<usize> = (0..examples.len()).collect();
        order.sort_by_key(|i| mix64(*i as u64 ^ seed));
        let table_t = stack(table, s);
        for chunk in order.chunks(batch) {
            self.tape.truncate(self.keep);
            let ctx: Vec<Function> = chunk.iter().map(|&i| examples[i].input.clone()).collect();
            let x = self.tape.constant(stack(&ctx, s));
            let t = self.tape.constant(table_t.clone());
            let y = self.forward(x);
            let scores = self.tape.cosine_table(y, t);
            let terms = MixtureTerms {
                targets: chunk.iter().map(|&i| examples[i].target).collect(),
                k: chunk.iter().map(|&i| examples[i].k).collect(),
                coef: chunk.iter().map(|&i| examples[i].coef).collect(),
                gamma,
                other: chunk.iter().map(|&i| examples[i].other).collect(),
                beta,
            };
            let loss = self.tape.mixture_nll(scores, terms);
            total += self.tape.value(loss).get(0, 0, 0).re;
            batches += 1;
            self.tape.backward(loss);
            adam.step(&mut self.tape, &params);
            self.mask_modes();
        }
        self.tape.truncate(self.keep);
        total / batches.max(1) as f64
    }

    /// One epoch of softmax training: each example's output is scored by
    /// cosine against every row of `table` (`cosine_table`) and pulled toward
    /// its target row by `cross_entropy` at inverse temperature `beta`; Adam
    /// steps after each batch of `batch`, in an order shuffled by `seed`.
    /// Returns the mean loss per batch, whose exponential is the
    /// operator's perplexity over the table.
    pub fn train_epoch_softmax(
        &mut self,
        table: &[Function],
        examples: &[Example],
        beta: f64,
        batch: usize,
        adam: &mut Adam,
        seed: u64,
    ) -> f64 {
        let params = self.params();
        let space = self.config.space;
        let mut total = 0.0;
        let mut batches = 0usize;
        let mut order: Vec<usize> = (0..examples.len()).collect();
        order.sort_by_key(|i| mix64(*i as u64 ^ seed));
        let table_t = stack(table, space);
        for chunk in order.chunks(batch) {
            self.reset();
            let inputs: Vec<Function> = chunk.iter().map(|&i| examples[i].input.clone()).collect();
            let targets: Vec<usize> = chunk.iter().map(|&i| examples[i].target).collect();
            let x = self.tape.constant(stack(&inputs, space));
            let t = self.tape.constant(table_t.clone());
            let y = self.forward(x);
            let scores = self.tape.cosine_table(y, t);
            let loss = self.tape.cross_entropy(scores, &targets, beta);
            total += self.tape.value(loss).data[0].re;
            batches += 1;
            self.tape.backward(loss);
            adam.step(&mut self.tape, &params);
            self.mask_modes();
        }
        self.reset();
        total / batches.max(1) as f64
    }
}
