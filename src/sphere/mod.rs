//! The sphere: a word's sixty-four channels as the sixty-four real spherical
//! harmonics of degree `l < 8`.
//!
//! In Phient's world, a word (w) is a seed: what it relates to, under what
//! conditions (c). Phical stores no point: a word is 64 relations, each a
//! phasor on a closed circle. Sixty-four is not a round number
//! by accident: the real spherical harmonics `Y_l^m` with `0 ≤ l < 8` and
//! `−l ≤ m ≤ l` number `Σ(2l+1) = 8² = 64`. So the same 64 channels that are
//! a Fourier series on the circle are also a function on the sphere,
//! band-limited at degree 7 — the representation FourCastNet 3 and the
//! spherical Fourier neural operator (torch-harmonics, NVIDIA) forecast the
//! atmosphere in, where a learned weight per *degree* `l` is an operator that
//! commutes with every rotation of the globe.
//!
//! Channel `k` ↔ `(l, m)` by `k = l² + l + m`, the standard packing. `l` is
//! the *scale* of the relation (0 = the word's mean, 7 = its finest detail);
//! `m` is its orientation. A hexagram octave (`phistd::hexagram`) groups the
//! circle's modes by scale; the degree does the same on the sphere.
//!
//! `Sphere` samples a Gauss–Legendre × equiangular grid and gives the
//! analysis and synthesis matrices between 64 coefficients and the grid; the
//! two are exact inverses on the band-limited space (tested). `degree_gate`
//! keeps degrees `≤ L` — a resolution change on the sphere, the counterpart
//! of `Fno::at_resolution` on the circle.

mod basis;
mod grid;
mod transform;

pub use basis::{degree_of, index_of, order_of, real_harmonic, DEGREES, HARMONICS};
pub use grid::{gauss_legendre, Grid};
pub use transform::Sphere;

use phistd::c64;

/// Keep only degrees `≤ max_degree` of a 64-coefficient function: the
/// function seen at a coarser resolution on the sphere.
pub fn degree_gate(coeffs: &[c64], max_degree: usize) -> Vec<c64> {
    coeffs
        .iter()
        .enumerate()
        .map(|(k, &c)| if degree_of(k) <= max_degree { c } else { c64::new(0.0, 0.0) })
        .collect()
}

/// Energy per degree: how much of the function lives at each scale.
pub fn degree_energy(coeffs: &[c64]) -> [f64; DEGREES] {
    let mut e = [0.0; DEGREES];
    for (k, c) in coeffs.iter().enumerate().take(HARMONICS) {
        e[degree_of(k)] += c.norm_sqr();
    }
    e
}

#[cfg(test)]
mod tests;
