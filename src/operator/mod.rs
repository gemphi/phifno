//! The Fourier neural operator on a circle.
//!
//! An operator maps a function on a circle to another. This is the Fourier
//! neural operator of Li et al. (2020) in its one-dimensional periodic form:
//! a stack of layers
//!
//! ```text
//!   h ← σ( F⁻¹( R · F(h) )  +  W h + b )
//! ```
//!
//! where `F` is the unitary FFT along the sample axis, `R` is a learned complex
//! `width × width` matrix **per Fourier mode** (a filter in the spectral
//! domain — equivalently a learned circular convolution with width-mixing),
//! `W` is a pointwise linear map across the width, and `σ` is GELU on the real
//! and imaginary parts. A lift takes the single input function to `width`
//! features and a projection brings it back.
//!
//! Keeping only the low `modes` is the regulariser: the operator is forced to
//! be smooth over the circle and the exceptions have to earn their
//! high-frequency coefficients. It is also what makes the operator
//! resolution-free — the same `R` applies at any power-of-two sample count.
//!
//! Training is by hinge on cosine similarity against a sampled decoy, which is
//! a ranking objective: given an input function, make the right target the
//! nearest of a table. `modes = 0` removes the spectral path and is the
//! pointwise control, which cannot move information between samples.

mod fno;
mod graph;
mod modes;
mod module;
mod resolution;
mod sampler;
mod save;
mod symmetry;
mod train;
#[cfg(test)]
mod tests;

pub use fno::Fno;
pub use module::{Module, Sequential};
pub use phistd::{cosine, stack, unstack};
pub use sampler::{rank, zeta_weights, FrequencySampler};
pub use save::APPLY_CHUNK;

use phistd::Var;
use phistd::Function;
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FnoConfig {
    /// Samples on the circle. Must be a power of two.
    pub space: usize,
    /// Feature width after the lift.
    pub width: usize,
    /// Number of Fourier layers.
    pub layers: usize,
    /// Kept low modes on each side of zero frequency (`0` disables the
    /// spectral path entirely — the pointwise control).
    pub modes: usize,
    /// Hinge margin on cosine similarity.
    pub margin: f64,
    pub lr: f64,
    /// How the spectral path reads the samples: as a circle (FFT, one
    /// weight per frequency) or as a sphere (the samples *are* spherical
    /// harmonic coefficients, one weight per degree, tied across orders).
    #[serde(default)]
    pub tie: Tie,
}

/// The geometry of the spectral path, and the symmetry it keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Tie {
    /// FFT over the sample axis; `modes` low frequencies kept; a weight per
    /// kept frequency. The operator of docs/76–78.
    #[default]
    Circle,
    /// No FFT: sample `k` is the real spherical harmonic `Y_l^m` with
    /// `l = ⌊√k⌋`, and one complex weight per degree `l` is shared by its
    /// `2l+1` orders — an operator that commutes with rotations of the
    /// sphere, as in the spherical FNO / FourCastNet 3. `space` must be a
    /// perfect square (64 = degrees 0..8); `modes` is the number of degrees
    /// kept.
    Sphere,
    /// The sphere with one weight per `(l, |m|)`: the cosine and sine
    /// orders of one `|m|` share a weight, different `|m|` do not. Between
    /// the isotropic sphere (8 groups) and the free circle (64): 36 groups
    /// for `l < 8`. Not rotation-equivariant; it keeps the reflection
    /// symmetry `φ → −φ` only.
    Anisotropic,
    /// The circle read as a wave (docs/154): the same FFT path, no biases,
    /// and the phase-preserving activation `z · tanh|z| / |z|` in place of
    /// GELU on the real and imaginary parts. Every step is then
    /// complex-linear or acts on modulus alone, so the operator commutes
    /// with a global phase: `F(e^{iφ}x) = e^{iφ}F(x)`.
    Phase,
}

impl Tie {
    /// The group of each sample under this tie; `None` for the circle
    /// (plain or read as a wave).
    pub fn groups(self, space: usize) -> Option<Vec<usize>> {
        let degree = |k: usize| (k as f64).sqrt().floor() as usize;
        match self {
            Tie::Circle | Tie::Phase => None,
            Tie::Sphere => Some((0..space).map(degree).collect()),
            Tie::Anisotropic => Some(
                (0..space)
                    .map(|k| {
                        let l = degree(k);
                        let am = (k as isize - (l * l + l) as isize).unsigned_abs();
                        // groups of lower degrees: Σ_{j<l}(j+1) = l(l+1)/2
                        l * (l + 1) / 2 + am
                    })
                    .collect(),
            ),
        }
    }

    /// How many weight slots `space` samples hold under this tie.
    pub fn degrees(self, space: usize) -> usize {
        self.groups(space).and_then(|g| g.iter().max().map(|m| m + 1)).unwrap_or(space)
    }
}

impl FnoConfig {
    /// A sensible default for a circle of `space` samples.
    pub fn for_space(space: usize) -> Self {
        Self { space, width: 8, layers: 2, modes: (space / 5).max(1), margin: 0.2, lr: 2e-3, tie: Tie::Circle }
    }

    /// The sphere: 64 samples read as degrees 0..8 (phifno), width 8, two
    /// layers, every degree kept, one weight per degree (1,192 parameters).
    pub fn sphere() -> Self {
        Self { space: crate::HARMONICS, width: 8, layers: 2, modes: crate::DEGREES, margin: 0.2, lr: 2e-3, tie: Tie::Sphere }
    }

    /// One weight per `(l, |m|)`: the tie that beats the circle on Austen.
    pub fn anisotropic() -> Self {
        Self { tie: Tie::Anisotropic, ..Self::sphere() }
    }

    /// The circle operator of the sphere's width and depth, for comparison.
    pub fn circle64() -> Self {
        Self { modes: 12, tie: Tie::Circle, ..Self::sphere() }
    }
}

/// A training example: an input function and the index of its target in the
/// table the operator ranks against.
#[derive(Debug, Clone)]
pub struct Example {
    pub input: Function,
    pub target: usize,
}

/// A training example for the mixture objective: the input, the target's
/// index in the table, and the constants of the mixture at that position —
/// `k + coef · (γ · scored + (1−γ) · other)`.
#[derive(Debug, Clone)]
pub struct MixtureExample {
    pub input: Function,
    pub target: usize,
    pub k: f64,
    pub coef: f64,
    pub other: f64,
}

pub(super) struct Layer {
    pub(super) r: Var,
    pub(super) w: Var,
    pub(super) b: Var,
}
