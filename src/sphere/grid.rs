//! A quadrature grid on the sphere: Gauss–Legendre colatitudes, equiangular
//! longitudes — the grid torch-harmonics calls `legendre-gauss`.

use std::f64::consts::PI;

/// Gauss–Legendre nodes and weights on `[-1, 1]`, by Newton on the Legendre
/// polynomial. Exact for polynomials of degree `< 2n`.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    (nodes, weights)
}

/// The sample points of the grid, with the quadrature weight of each.
#[derive(Clone, Debug)]
pub struct Grid {
    pub nlat: usize,
    pub nlon: usize,
    /// Colatitude of each latitude row.
    pub theta: Vec<f64>,
    /// Longitude of each column.
    pub phi: Vec<f64>,
    /// Quadrature weight of each row (includes the `2π / nlon` of the column).
    pub weight: Vec<f64>,
}

impl Grid {
    pub fn new(nlat: usize, nlon: usize) -> Self {
        let (x, w) = gauss_legendre(nlat);
        let theta = x.iter().map(|c| c.acos()).collect();
        let weight = w.iter().map(|w| w * 2.0 * PI / nlon as f64).collect();
        let phi = (0..nlon).map(|j| 2.0 * PI * j as f64 / nlon as f64).collect();
        Self { nlat, nlon, theta, phi, weight }
    }

    pub fn len(&self) -> usize {
        self.nlat * self.nlon
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row-major `(row, column)` of sample `i`.
    pub fn at(&self, i: usize) -> (f64, f64, f64) {
        let (r, c) = (i / self.nlon, i % self.nlon);
        (self.theta[r], self.phi[c], self.weight[r])
    }
}
