use super::*;
use phistd::c64;

fn random_coeffs(seed: u64) -> Vec<c64> {
    let mut s = seed;
    (0..HARMONICS)
        .map(|_| {
            s = phistd::mix64(s ^ 0xA5);
            let a = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            s = phistd::mix64(s ^ 0x5A);
            let b = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            c64::new(a, b)
        })
        .collect()
}

#[test]
fn indexing_packs_sixty_four() {
    let mut seen = vec![false; HARMONICS];
    for l in 0..DEGREES {
        for m in -(l as isize)..=(l as isize) {
            let k = index_of(l, m);
            assert_eq!((degree_of(k), order_of(k)), (l, m));
            seen[k] = true;
        }
    }
    assert!(seen.iter().all(|s| *s));
}

#[test]
fn harmonics_are_orthonormal_on_the_grid() {
    let sphere = Sphere::minimal();
    for a in 0..HARMONICS {
        for b in 0..HARMONICS {
            let mut dot = 0.0;
            for i in 0..sphere.grid.len() {
                let (t, p, w) = sphere.grid.at(i);
                dot += w * real_harmonic(a, t, p) * real_harmonic(b, t, p);
            }
            let want = if a == b { 1.0 } else { 0.0 };
            assert!((dot - want).abs() < 1e-9, "Y{a}·Y{b} = {dot}");
        }
    }
}

#[test]
fn analysis_inverts_synthesis() {
    let sphere = Sphere::minimal();
    let c = random_coeffs(3);
    let back = sphere.analysis(&sphere.synthesis(&c));
    for k in 0..HARMONICS {
        assert!((back[k] - c[k]).norm() < 1e-9, "k={k}: {} vs {}", back[k], c[k]);
    }
}

#[test]
fn rotation_about_the_pole_matches_a_shifted_grid() {
    let sphere = Sphere::new(8, 32);
    let c = random_coeffs(9);
    let angle = 2.0 * std::f64::consts::PI * 3.0 / 32.0; // three columns
    let rotated = sphere.rotate_about_pole(&c, angle);
    let f = sphere.synthesis(&c);
    let g = sphere.synthesis(&rotated);
    for r in 0..8 {
        for col in 0..32 {
            let shifted = f[r * 32 + (col + 3) % 32];
            assert!((g[r * 32 + col] - shifted).norm() < 1e-9);
        }
    }
    // Degree energy is invariant under rotation.
    let (e0, e1) = (degree_energy(&c), degree_energy(&rotated));
    for l in 0..DEGREES {
        assert!((e0[l] - e1[l]).abs() < 1e-9);
    }
}

#[test]
fn degree_gate_is_a_projection() {
    let c = random_coeffs(4);
    let g = degree_gate(&c, 3);
    assert_eq!(degree_gate(&g, 3), g);
    assert_eq!(degree_energy(&g)[4..].iter().sum::<f64>(), 0.0);
    assert!(degree_energy(&g)[..4].iter().sum::<f64>() > 0.0);
}
