//! Laplace–Beltrami heat flow: diffusion on the circle, exact in Fourier
//! modes, and on the sphere, exact in spherical harmonics and band-limited
//! to the [`crate::sphere`] basis' eight degrees. Used to smooth a persona's
//! distribution over sectors before comparing it, and to define "nearby"
//! by overlap after diffusion.

use crate::sphere::{degree_of, Sphere};
use phistd::c64;
use std::f64::consts::TAU;

/// Heat flow of a real function sampled at `n` equally spaced points on the
/// circle for time `t`: mode `k` decays by `exp(−k² t)`. Exact (the circle's
/// eigenfunctions are the Fourier modes), O(n²) by direct summation, which
/// is what a 64-bin histogram wants.
pub fn circle(f: &[f64], t: f64) -> Vec<f64> {
    let n = f.len();
    if n == 0 {
        return Vec::new();
    }
    let half = n / 2;
    // Real DFT coefficients.
    let mut a = vec![0.0f64; half + 1];
    let mut b = vec![0.0f64; half + 1];
    for k in 0..=half {
        for (j, v) in f.iter().enumerate() {
            let th = TAU * (k * j) as f64 / n as f64;
            a[k] += v * th.cos();
            b[k] += v * th.sin();
        }
    }
    let mut out = vec![0.0f64; n];
    for (j, o) in out.iter_mut().enumerate() {
        let mut s = a[0] / n as f64;
        for k in 1..=half {
            let decay = (-((k * k) as f64) * t).exp();
            let th = TAU * (k * j) as f64 / n as f64;
            let w = if k == half && n.is_multiple_of(2) { 1.0 } else { 2.0 };
            s += w * decay * (a[k] * th.cos() + b[k] * th.sin()) / n as f64;
        }
        *o = s;
    }
    out
}

/// Heat flow on the sphere, on harmonic coefficients: `Y_l^m` is an
/// eigenfunction of the Laplace–Beltrami operator with eigenvalue
/// `−l(l+1)`, so coefficient `k` (degree `l`) decays by `exp(−l(l+1) t)`.
/// Exact. Degree 0 — the mean — never moves.
pub fn sphere_coeffs(coeffs: &[c64], t: f64) -> Vec<c64> {
    coeffs
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let l = degree_of(k) as f64;
            c * (-l * (l + 1.0) * t).exp()
        })
        .collect()
}

/// Heat flow on the sphere, on values at `grid`'s points: analysed into the
/// band-limited basis, decayed degree by degree, synthesised back. Exact
/// for band-limited values; anything above the basis' top degree is
/// dropped by the analysis.
pub fn sphere(grid: &Sphere, values: &[c64], t: f64) -> Vec<c64> {
    grid.synthesis(&sphere_coeffs(&grid.analysis(values), t))
}

/// Overlap of two non-negative distributions after diffusing both for `t`:
/// the Bhattacharyya coefficient `Σ √(p_i q_i)`, in [0, 1].
pub fn overlap_after(p: &[f64], q: &[f64], t: f64) -> f64 {
    let (dp, dq) = (circle(p, t), circle(q, t));
    let norm = |v: &[f64]| {
        let s: f64 = v.iter().map(|x| x.max(0.0)).sum::<f64>().max(1e-300);
        v.iter().map(|x| x.max(0.0) / s).collect::<Vec<_>>()
    };
    let (dp, dq) = (norm(&dp), norm(&dq));
    dp.iter().zip(&dq).map(|(a, b)| (a * b).sqrt()).sum::<f64>().clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_preserves_mass_and_flattens() {
        let mut f = vec![0.0f64; 64];
        f[10] = 1.0;
        let g = circle(&f, 0.0);
        assert!(g.iter().zip(&f).all(|(a, b)| (a - b).abs() < 1e-9), "t = 0 is the identity");
        let h = circle(&f, 0.05);
        assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-9, "mass is conserved");
        assert!(h[10] < 1.0 && h[11] > 0.0 && h[9] > 0.0, "mass spreads to the neighbours");
        let flat = circle(&f, 100.0);
        assert!(flat.iter().all(|v| (v - 1.0 / 64.0).abs() < 1e-6), "long times flatten");
    }

    #[test]
    fn heat_on_the_sphere_decays_each_degree_and_keeps_the_mean() {
        use crate::sphere::HARMONICS;
        let coeffs: Vec<c64> = (0..HARMONICS).map(|k| c64::new(1.0 + k as f64 * 0.1, 0.0)).collect();
        let t = 0.03;
        let hot = sphere_coeffs(&coeffs, t);
        for (k, (a, b)) in hot.iter().zip(&coeffs).enumerate() {
            let l = degree_of(k) as f64;
            assert!((a - b * (-l * (l + 1.0) * t).exp()).norm() < 1e-12, "coefficient {k}");
        }
        assert_eq!(hot[0], coeffs[0], "the mean does not move");
        let grid = Sphere::minimal();
        let values = grid.synthesis(&coeffs);
        let same = sphere(&grid, &values, 0.0);
        assert!(same.iter().zip(&values).all(|(a, b)| (a - b).norm() < 1e-9), "t = 0 is the identity on band-limited values");
        let flat = sphere(&grid, &values, 100.0);
        let mean = grid.synthesis(&crate::sphere::degree_gate(&coeffs, 0));
        assert!(flat.iter().zip(&mean).all(|(a, b)| (a - b).norm() < 1e-9), "long times leave only the mean");
    }

    #[test]
    fn overlap_grows_with_diffusion_for_nearby_peaks() {
        let (mut p, mut q) = (vec![0.0f64; 64], vec![0.0f64; 64]);
        p[10] = 1.0;
        q[12] = 1.0;
        assert!(overlap_after(&p, &q, 0.0) < 1e-6, "disjoint before diffusion");
        assert!(overlap_after(&p, &q, 0.2) > 0.5, "nearby after");
        let mut far = vec![0.0f64; 64];
        far[42] = 1.0;
        assert!(overlap_after(&p, &far, 0.2) < overlap_after(&p, &q, 0.2));
    }
}
