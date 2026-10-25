//! Analysis and synthesis between 64 coefficients and a grid.

use super::basis::{real_harmonic, HARMONICS};
use super::grid::Grid;
use phistd::c64;

/// The transform pair for one grid: `synth[i][k] = Y_k(x_i)` and
/// `anal[k][i] = w_i · Y_k(x_i)`, so `anal · synth = I` on the band-limited
/// space whenever the quadrature is exact — `nlat ≥ 8`, `nlon ≥ 16`.
pub struct Sphere {
    pub grid: Grid,
    synth: Vec<f64>,
}

impl Sphere {
    /// The smallest grid on which the 64 harmonics are exactly resolved.
    pub fn minimal() -> Self {
        Self::new(8, 16)
    }

    pub fn new(nlat: usize, nlon: usize) -> Self {
        let grid = Grid::new(nlat, nlon);
        let mut synth = vec![0.0; grid.len() * HARMONICS];
        for i in 0..grid.len() {
            let (theta, phi, _) = grid.at(i);
            for k in 0..HARMONICS {
                synth[i * HARMONICS + k] = real_harmonic(k, theta, phi);
            }
        }
        Self { grid, synth }
    }

    /// The function on the grid: `f(x_i) = Σ_k c_k Y_k(x_i)`.
    pub fn synthesis(&self, coeffs: &[c64]) -> Vec<c64> {
        (0..self.grid.len())
            .map(|i| {
                let row = &self.synth[i * HARMONICS..(i + 1) * HARMONICS];
                coeffs.iter().zip(row).map(|(c, y)| c * y).sum()
            })
            .collect()
    }

    /// The coefficients of a function on the grid: `c_k = Σ_i w_i f(x_i) Y_k(x_i)`.
    pub fn analysis(&self, values: &[c64]) -> Vec<c64> {
        let mut out = vec![c64::new(0.0, 0.0); HARMONICS];
        for i in 0..self.grid.len() {
            let (_, _, w) = self.grid.at(i);
            let row = &self.synth[i * HARMONICS..(i + 1) * HARMONICS];
            for k in 0..HARMONICS {
                out[k] += values[i] * (w * row[k]);
            }
        }
        out
    }

    /// Rotate a band-limited function about the pole by `angle` — on the
    /// sphere a rotation mixes only within each degree, and about the pole it
    /// is a phase on `m`. Tested against synthesis on the rotated grid.
    pub fn rotate_about_pole(&self, coeffs: &[c64], angle: f64) -> Vec<c64> {
        use super::basis::{degree_of, index_of, order_of};
        let mut out = vec![c64::new(0.0, 0.0); HARMONICS];
        for k in 0..HARMONICS {
            let (l, m) = (degree_of(k), order_of(k));
            if m == 0 {
                out[k] = coeffs[k];
                continue;
            }
            let (kc, ks) = (index_of(l, m.abs()), index_of(l, -m.abs()));
            let (a, b) = (coeffs[kc], coeffs[ks]);
            let (c, s) = ((m.abs() as f64 * angle).cos(), (m.abs() as f64 * angle).sin());
            // cos(m(φ−α)) = cos mφ cos mα + sin mφ sin mα, and likewise sine.
            out[kc] = a * c + b * s;
            out[ks] = b * c - a * s;
        }
        out
    }
}
