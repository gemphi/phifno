//! The sphere tie: the gradient of `tie`, and an operator that learns a
//! rotation of the sphere.

use super::super::*;
use crate::sphere::{Sphere, HARMONICS};
use phistd::{Adam, Tape, Tensor};
use phistd::{c64, unit_random, Function};

#[test]
fn tie_gradient_matches_finite_differences() {
    let groups: Vec<usize> = (0..16).map(|k| (k as f64).sqrt().floor() as usize).collect();
    let build = |r: Tensor| -> (Tape, Var, Var) {
        let mut tape = Tape::new();
        let x = tape.constant(Tensor::randn([2, 3, 16], 1.0, 5));
        let target = tape.constant(Tensor::randn([2, 3, 16], 1.0, 6));
        let r = tape.param(r);
        let tied = tape.tie(r, &groups);
        let y = tape.mode_mix(x, tied);
        let d = tape.add(y, target);
        let dd = tape.mul(d, d);
        let loss = tape.mean_real(dd);
        (tape, loss, r)
    };
    let r0 = Tensor::randn([3, 3, 4], 0.5, 7);
    let (mut tape, loss, r) = build(r0.clone());
    let base = tape.value(loss).get(0, 0, 0).re;
    tape.backward(loss);
    let g = tape.grad(r).unwrap().clone();
    let eps = 1e-6;
    for k in 0..g.len() {
        for part in 0..2 {
            let mut rb = r0.clone();
            if part == 0 { rb.data[k].re += eps } else { rb.data[k].im += eps }
            let (tape2, loss2, _) = build(rb);
            let fd = (tape2.value(loss2).get(0, 0, 0).re - base) / eps;
            let an = if part == 0 { g.data[k].re } else { g.data[k].im };
            assert!((fd - an).abs() < 1e-4 * (1.0 + an.abs()), "elem {k} part {part}: {fd:.6e} vs {an:.6e}");
        }
    }
}

/// Targets are the inputs filtered by degree — each scale `l` scaled by its
/// own gain, the sphere's analogue of a smoothing or sharpening kernel. An
/// operator that commutes with rotation is exactly one weight per degree,
/// so the sphere tie represents this with eight weights; on the circle the
/// same 64 numbers read as samples of a periodic signal and the filter is
/// nothing an FFT diagonalises. (A *rotation* of the sphere is the opposite
/// case — a phase per order — and is deliberately not the task.)
fn degree_filter_task() -> (Vec<Function>, Vec<Function>, Vec<Example>, Vec<usize>) {
    let gains = [1.0, 0.8, -0.5, 0.3, 1.2, -1.0, 0.5, 0.1];
    let inputs: Vec<Function> = (0..60).map(|i| unit_random(HARMONICS, i as u64)).collect();
    let targets: Vec<Function> = inputs
        .iter()
        .map(|f| f.iter().enumerate().map(|(k, c)| c * gains[crate::sphere::degree_of(k)]).collect())
        .collect();
    let examples: Vec<Example> = (0..60).map(|i| Example { input: inputs[i].clone(), target: i }).collect();
    (inputs, targets, examples, (0..60).collect())
}

#[test]
fn sphere_operator_learns_a_degree_filter() {
    let (inputs, targets, examples, pool) = degree_filter_task();
    let cfg = FnoConfig { space: 64, width: 4, layers: 1, modes: 8, margin: 0.5, lr: 1e-2, tie: Tie::Sphere };
    let mut fno = Fno::new(cfg, 1);
    assert_eq!(fno.tape.value(fno.layers[0].r).shape, [4, 4, 8]);
    let mut adam = Adam::new(cfg.lr);
    for epoch in 0..60 {
        fno.train_epoch(&targets, &examples, &pool, 20, &mut adam, epoch);
    }
    let (mrr, _) = rank(&targets, &fno.apply(&inputs), &examples, &pool);
    assert!(mrr > 0.9, "sphere MRR {mrr}");
}

#[test]
fn the_circle_finds_the_degree_filter_harder() {
    let (inputs, targets, examples, pool) = degree_filter_task();
    let (id, _) = rank(&targets, &inputs, &examples, &pool);
    let mut best = 0.0f64;
    for tie in [Tie::Circle, Tie::Sphere] {
        let cfg = FnoConfig { space: 64, width: 4, layers: 1, modes: 8, margin: 0.5, lr: 1e-2, tie };
        let mut fno = Fno::new(cfg, 1);
        let mut adam = Adam::new(cfg.lr);
        for epoch in 0..30 {
            fno.train_epoch(&targets, &examples, &pool, 20, &mut adam, epoch);
        }
        let (mrr, _) = rank(&targets, &fno.apply(&inputs), &examples, &pool);
        if tie == Tie::Sphere {
            assert!(mrr > best, "sphere {mrr} should beat circle {best} (identity {id})");
        }
        best = best.max(mrr);
    }
    let _ = (Sphere::minimal(), c64::new(0.0, 0.0));
}

#[test]
fn anisotropic_groups_count_thirty_six() {
    let g = Tie::Anisotropic.groups(64).unwrap();
    assert_eq!(g.iter().max().copied(), Some(35));
    // k=0 → group 0; k=1 (l=1,m=-1) → 2; k=2 (l=1,m=0) → 1; k=3 (l=1,m=1) → 2
    assert_eq!(&g[..4], &[0, 2, 1, 2]);
    let cfg = FnoConfig { space: 64, width: 4, layers: 1, modes: 8, margin: 0.5, lr: 1e-2, tie: Tie::Anisotropic };
    let fno = Fno::new(cfg, 1);
    assert_eq!(fno.tape.value(fno.layers[0].r).shape, [4, 4, 36]);
}

#[test]
fn at_resolution_on_the_sphere_keeps_the_shared_degrees() {
    // Groups are numbered degree first, so the groups two resolutions share
    // carry over, the new ones start at zero, and `space` is a perfect
    // square (36 is not a power of two).
    for (cfg, at64, at16) in [(FnoConfig::sphere(), 8, 4), (FnoConfig::anisotropic(), 36, 10)] {
        let fno = Fno::new(cfg, 3);
        let mut small = fno.at_resolution(16);
        assert_eq!((small.config.space, small.config.modes), (16, 4));
        let big_r = fno.tape.value(fno.params()[2]).clone();
        let small_r = small.tape.value(small.params()[2]).clone();
        assert_eq!((big_r.shape[2], small_r.shape[2]), (at64, at16), "{:?}", cfg.tie);
        for g in 0..at16 {
            assert_eq!(small_r.get(0, 1, g), big_r.get(0, 1, g), "group {g} carries over");
        }
        let back = small.at_resolution(64);
        let back_r = back.tape.value(back.params()[2]);
        for g in 0..at64 {
            let want = match g < at16 {
                true => big_r.get(0, 1, g),
                false => c64::new(0.0, 0.0),
            };
            assert_eq!(back_r.get(0, 1, g), want, "group {g} on the way back");
        }
        assert_eq!(small.apply(&[unit_random(16, 5)])[0].len(), 16, "the moved operator runs");
        assert_eq!(fno.at_resolution(36).config.modes, 6);
    }
}
