//! The Fourier neural operator: construction, forward, apply, masks (resolution transfer is resolution.rs).

use super::{FnoConfig, Layer, Tie};
use phistd::{Tape, Tensor, Var};
use phistd::{c64, mix64, Function};
use phistd::{stack, unstack};
use super::save::APPLY_CHUNK;


/// The operator: its parameters live on the tape's first nodes.
pub struct Fno {
    pub config: FnoConfig,
    pub tape: Tape,
    pub(super) lift: Var,
    pub(super) lift_b: Var,
    pub(super) layers: Vec<Layer>,
    pub(super) proj: Var,
    pub(super) keep: usize,
    pub(super) mask: Vec<bool>,
}

impl Fno {
    pub fn new(config: FnoConfig, seed: u64) -> Self {
        match config.tie {
            Tie::Circle | Tie::Phase => assert!(config.space.is_power_of_two(), "space must be a power of two"),
            Tie::Sphere | Tie::Anisotropic => {
                let d = (config.space as f64).sqrt() as usize;
                assert_eq!(d * d, config.space, "sphere: space must be a perfect square");
            }
        }
        let mut tape = Tape::new();
        let s = config.space;
        let wd = config.width;
        let seed = mix64(seed);
        let lift = tape.param(Tensor::randn([1, wd, 1], 1.0, seed ^ 1));
        let lift_b = tape.param(Tensor::zeros([1, wd, 1]));
        // On the circle `R` is one weight per signed frequency and the mask
        // keeps the low ones; on the sphere it is one weight per degree and
        // the mask keeps the low degrees.
        let (slots, mask): (usize, Vec<bool>) = match config.tie.groups(s) {
            None => (s, (0..s).map(|m| m < config.modes || m + config.modes >= s).collect()),
            Some(g) => {
                // `modes` counts degrees kept, under either sphere tie.
                let groups = g.iter().max().map(|m| m + 1).unwrap_or(0);
                let deg = |group: usize| match config.tie {
                    Tie::Anisotropic => (0..).find(|l| (l + 1) * (l + 2) / 2 > group).unwrap_or(0),
                    _ => group,
                };
                (groups, (0..groups).map(|gi| deg(gi) < config.modes).collect())
            }
        };
        let mut layers = Vec::with_capacity(config.layers);
        for l in 0..config.layers {
            let mut r = Tensor::randn([wd, wd, slots], 1.0 / wd as f64, seed ^ (10 + l as u64));
            for i in 0..wd {
                for o in 0..wd {
                    for (m, keep) in mask.iter().enumerate() {
                        if !keep {
                            r.set(i, o, m, c64::new(0.0, 0.0));
                        }
                    }
                }
            }
            let r = tape.param(r);
            let w = tape.param(Tensor::randn([wd, wd, 1], 1.0 / (wd as f64).sqrt(), seed ^ (20 + l as u64)));
            let b = tape.param(Tensor::zeros([1, wd, 1]));
            layers.push(Layer { r, w, b });
        }
        let proj = tape.param(Tensor::randn([wd, 1, 1], 1.0 / (wd as f64).sqrt(), seed ^ 99));
        let keep = tape.len();
        Self { config, tape, lift, lift_b, layers, proj, keep, mask }
    }

    pub fn params(&self) -> Vec<Var> {
        let mut v = vec![self.lift, self.lift_b];
        for l in &self.layers {
            v.extend([l.r, l.w, l.b]);
        }
        v.push(self.proj);
        v
    }

    /// Registers an extra parameter on the operator's tape — a table of
    /// functions, for joint training — so it survives `reset`.
    pub fn attach_param(&mut self, t: Tensor) -> Var {
        self.tape.truncate(self.keep);
        let v = self.tape.param(t);
        self.keep = self.tape.len();
        v
    }

    /// Clears everything on the tape after the parameters.
    pub fn reset(&mut self) {
        self.tape.truncate(self.keep);
    }

    pub fn parameter_count(&self) -> usize {
        self.params().iter().map(|p| self.tape.value(*p).len()).sum()
    }

    /// Forward pass on a `[b,1,space]` input already on the tape.
    pub fn forward(&mut self, x: Var) -> Var {
        let p = self.params();
        super::graph::forward(&mut self.tape, &self.config, &p, x)
    }

    /// Applies the operator to a batch of functions, off the tape.
    ///
    /// Runs in chunks of [`APPLY_CHUNK`] functions: every op allocates its
    /// output, so one huge batch keeps a dozen huge tensors alive and pays
    /// for the memory (a 4,096-function batch was 2.5 s; in chunks it is
    /// under 0.5 s on the same machine).
    pub fn apply(&mut self, inputs: &[Function]) -> Vec<Function> {
        let mut out = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(APPLY_CHUNK) {
            self.tape.truncate(self.keep);
            let x = self.tape.constant(stack(chunk, self.config.space));
            let y = self.forward(x);
            out.extend(unstack(self.tape.value(y)));
        }
        self.tape.truncate(self.keep);
        out
    }

}
