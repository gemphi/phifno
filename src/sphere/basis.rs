//! Real spherical harmonics up to degree 7, evaluated point by point.

use std::f64::consts::PI;

/// Degrees `0..DEGREES`; `DEGREES²` harmonics.
pub const DEGREES: usize = 8;
/// The sixty-four.
pub const HARMONICS: usize = DEGREES * DEGREES;

/// Channel `k` of the pair `(l, m)`: `l² + l + m`.
pub fn index_of(l: usize, m: isize) -> usize {
    (l * l) as usize + (l as isize + m) as usize
}

/// The degree of channel `k`: the largest `l` with `l² ≤ k`.
pub fn degree_of(k: usize) -> usize {
    (k as f64).sqrt().floor() as usize
}

/// The order of channel `k`, in `−l..=l`.
pub fn order_of(k: usize) -> isize {
    let l = degree_of(k);
    k as isize - (l * l + l) as isize
}

/// Associated Legendre `P_l^m(x)` for `m ≥ 0`, without the Condon–Shortley
/// phase, by the standard recurrences.
fn legendre(l: usize, m: usize, x: f64) -> f64 {
    // P_m^m
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = ((1.0 - x) * (1.0 + x)).max(0.0).sqrt();
        let mut fact = 1.0;
        for _ in 0..m {
            pmm *= fact * somx2;
            fact += 2.0;
        }
    }
    if l == m {
        return pmm;
    }
    // P_{m+1}^m
    let mut pmmp1 = x * (2 * m + 1) as f64 * pmm;
    if l == m + 1 {
        return pmmp1;
    }
    let mut pll = 0.0;
    for ll in (m + 2)..=l {
        pll = (x * (2 * ll - 1) as f64 * pmmp1 - (ll + m - 1) as f64 * pmm) / (ll - m) as f64;
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

fn factorial_ratio(l: usize, m: usize) -> f64 {
    // (l − m)! / (l + m)!
    let mut r = 1.0;
    for i in (l - m + 1)..=(l + m) {
        r /= i as f64;
    }
    r
}

/// The orthonormal real harmonic `Y_l^m(θ, φ)` — colatitude `θ`, longitude
/// `φ` — for channel `k`. `m < 0` takes the sine, `m > 0` the cosine, and the
/// set is orthonormal on the unit sphere.
pub fn real_harmonic(k: usize, theta: f64, phi: f64) -> f64 {
    let l = degree_of(k);
    let m = order_of(k);
    let am = m.unsigned_abs();
    let norm = ((2 * l + 1) as f64 / (4.0 * PI) * factorial_ratio(l, am)).sqrt();
    let p = legendre(l, am, theta.cos());
    match m {
        0 => norm * p,
        m if m > 0 => std::f64::consts::SQRT_2 * norm * p * (am as f64 * phi).cos(),
        _ => std::f64::consts::SQRT_2 * norm * p * (am as f64 * phi).sin(),
    }
}
