//! # phifno 1.0.0 — the spherical Fourier neural operator, for words
//!
//! In Phient's world, a word (w) is a seed: what it relates to, under what
//! conditions (c). The Phical paradigm, X ⇒ Y, drives and composes Phient's
//! Phasor Language Models: Phinet, Phiano, PhiTon. Phical stores no point: a
//! word is 64 relations, each a complex phasor on a closed circle with a
//! count; meaning is a side effect of pairs of words moving under a condition.
//!
//! phinet is the parent engine — tensors, tape, the operator on the circle —
//! and phifno is one of its children (with phical, phinum, phiqlm: PML, Phi
//! machine learning). It reads the same 64 channels a different way: as the
//! 64 real spherical harmonics `Y_l^m`, `l < 8`, so a word is a band-limited
//! function on the sphere and an operator that treats every rotation of the
//! sphere alike is **one complex weight per degree** — eight weights per
//! layer channel pair instead of sixty-four. That is the construction of the
//! spherical FNO and FourCastNet 3 (NVIDIA torch-harmonics / makani), which
//! forecast the atmosphere in exactly this representation; here the sphere
//! is a word's.
//!
//! What it buys, measured (phiano `bin/operator_sweep`, Austen, hinge, 3
//! epochs, pool 1,228): **12.8× chance validation / 13.1× test with 1,192
//! parameters**, against the circle operator's 14.7× / 14.6× with 8,360 —
//! seven times fewer parameters for 88% of the ranking, in 89 s against 142.
//! On the synthetic degree-filter task (phical's `filter` example, and
//! `degree_filter_task` in `operator/tests/sphere.rs`) the sphere tie
//! beats the circle outright, because the relation *is* per degree. And the
//! **anisotropic tie** (`Tie::Anisotropic`, one weight per `(l, |m|)`, 36
//! slots, 4,776 parameters) beats the circle on Austen itself: 16.0× / 15.6×
//! against 14.7× / 14.6×, in 85 s. `anisotropic()` is that configuration.
//!
//! The operator that reads this representation lives HERE too (docs/173):
//! phifno is Phient's one Fourier address — the harmonics ([`sphere`]), the
//! Fourier neural operator ([`operator`]: `Fno`, `FnoConfig`, `Tie` for the
//! circle, sphere and anisotropic ties), the heat flow exact in Fourier
//! modes ([`heat`]), and the FFT kernels re-exported from the engine
//! ([`fft`]). phical keeps the exact geometry of the relation circle and
//! re-exports these, so `phical::operator` still reads.
//!
//! ```
//! use phifno::Word;
//! use phistd::unit_random;
//!
//! let w = Word::new(unit_random(64, 3));
//! assert!((w.energy().iter().sum::<f64>() - 1.0).abs() < 1e-9);
//! assert_eq!(w.at_degree(3).coeffs.len(), 64);
//! ```

pub mod heat;
pub mod operator;
pub mod sphere;

/// The FFT kernels, re-exported from the engine so every Fourier thing has
/// the one address; the tensor autograd in phistd is their home of record.
pub mod fft {
    pub use phistd::{fft2_inplace, fft_inplace};
}

pub use operator::{rank, zeta_weights, Example, Fno, FnoConfig, FrequencySampler, MixtureExample, Module, Sequential, Tie};

pub use sphere::{degree_energy, degree_gate, degree_of, index_of, order_of, real_harmonic, Grid, Sphere, DEGREES, HARMONICS};
use phistd::{c64, Function};

/// A word as a function on the sphere: its 64 channels as coefficients.
#[derive(Clone, Debug)]
pub struct Word {
    pub coeffs: Function,
}

impl Word {
    pub fn new(coeffs: Function) -> Self {
        assert_eq!(coeffs.len(), HARMONICS, "a word is 64 coefficients");
        Self { coeffs }
    }

    /// Energy per degree — the word's spectrum by scale, rotation-invariant.
    pub fn energy(&self) -> [f64; DEGREES] {
        degree_energy(&self.coeffs)
    }

    /// The word seen at a coarser resolution: degrees `≤ max_degree` only.
    pub fn at_degree(&self, max_degree: usize) -> Word {
        Word { coeffs: degree_gate(&self.coeffs, max_degree) }
    }

    /// The word sampled on the sphere's minimal exact grid (8 × 16).
    pub fn on_grid(&self, sphere: &Sphere) -> Vec<c64> {
        sphere.synthesis(&self.coeffs)
    }

    /// Cosine between two words at a resolution — how much of their
    /// agreement lives at the coarse scales.
    pub fn agreement_at(&self, other: &Word, max_degree: usize) -> f64 {
        phistd::cosine(&self.at_degree(max_degree).coeffs, &other.at_degree(max_degree).coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phistd::unit_random;

    #[test]
    fn degree_gate_is_monotone_in_agreement_with_itself() {
        let w = Word::new(unit_random(64, 11));
        let full = w.agreement_at(&w, 7);
        assert!((full - 1.0).abs() < 1e-9);
        // With itself, every truncation still agrees perfectly on what remains.
        for l in 0..8 {
            assert!((w.agreement_at(&w, l) - 1.0).abs() < 1e-9);
        }
        let e = w.energy();
        assert!((e.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
}
