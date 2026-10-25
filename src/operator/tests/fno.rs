//! The operator learns the shift, the control cannot, softmax learns it too, resolution transfer, Sequential.

use super::super::*;
use phistd::Adam;
use phistd::{unit_random, Function};

    fn table(n: usize, space: usize) -> Vec<Function> {
        (0..n).map(|i| unit_random(space, i as u64 + 1)).collect()
    }

    /// The operator must learn a fixed relation the identity cannot express:
    /// a circular shift by one sample, which is exactly a spectral
    /// multiplication.
    #[test]
    fn test_learns_a_circular_shift_the_identity_cannot() {
        let space = 64;
        let inputs = table(60, space);
        let targets: Vec<Function> = inputs
            .iter()
            .map(|f| {
                let mut g = f.clone();
                g.rotate_right(1);
                g
            })
            .collect();
        let examples: Vec<Example> =
            (0..inputs.len()).map(|i| Example { input: inputs[i].clone(), target: i }).collect();
        let pool: Vec<usize> = (0..inputs.len()).collect();
        let (mrr_id, _) = rank(&targets, &inputs, &examples, &pool);

        let cfg = FnoConfig { space, width: 4, layers: 1, modes: 32, margin: 0.5, lr: 1e-2, tie: Tie::Circle };
        let mut fno = Fno::new(cfg, 1);
        let mut adam = Adam::new(cfg.lr);
        for epoch in 0..60 {
            fno.train_epoch(&targets, &examples, &pool, 20, &mut adam, epoch as u64);
        }
        let preds = fno.apply(&inputs);
        let (mrr, _) = rank(&targets, &preds, &examples, &pool);
        assert!(mrr > 0.9, "operator mrr {mrr:.3} (identity {mrr_id:.3})");
        assert!(mrr > mrr_id + 0.3, "must beat the identity: {mrr:.3} vs {mrr_id:.3}");
    }

    fn shift_task(space: usize, n: usize, by: usize) -> (Vec<Function>, Vec<Function>, Vec<Example>, Vec<usize>) {
        let inputs = table(n, space);
        let targets: Vec<Function> = inputs.iter().map(|f| { let mut g = f.clone(); g.rotate_right(by); g }).collect();
        let examples: Vec<Example> = (0..n).map(|i| Example { input: inputs[i].clone(), target: i }).collect();
        (inputs, targets, examples, (0..n).collect())
    }

    #[test]
    fn test_softmax_epoch_learns_the_shift_too() {
        let (inputs, targets, examples, pool) = shift_task(64, 60, 1);
        let cfg = FnoConfig { space: 64, width: 4, layers: 1, modes: 32, margin: 0.5, lr: 1e-2, tie: Tie::Circle };
        let mut fno = Fno::new(cfg, 1);
        let mut adam = Adam::new(cfg.lr).with_decay(1e-3);
        let mut last = f64::INFINITY;
        for epoch in 0..40 {
            last = fno.train_epoch_softmax(&targets, &examples, 8.0, 20, &mut adam, epoch);
        }
        let (mrr, _) = rank(&targets, &fno.apply(&inputs), &examples, &pool);
        assert!(mrr > 0.9, "softmax mrr {mrr:.3}, loss {last:.3}");
        assert!(last.exp() < 5.0, "perplexity over 60 items {:.2}", last.exp());
    }


    #[test]
    fn test_at_resolution_keeps_low_modes_and_transfers_the_shift() {
        // Train rotate-by-one at 32 samples; at 64 samples the same map of the
        // circle is rotate-by-two, and the transferred operator must do it.
        let (inputs32, targets32, examples32, pool32) = shift_task(32, 60, 1);
        let cfg = FnoConfig { space: 32, width: 4, layers: 1, modes: 8, margin: 0.5, lr: 1e-2, tie: Tie::Circle };
        let mut fno = Fno::new(cfg, 1);
        let mut adam = Adam::new(cfg.lr);
        for epoch in 0..80 {
            fno.train_epoch(&targets32, &examples32, &pool32, 20, &mut adam, epoch);
        }
        let (mrr32, _) = rank(&targets32, &fno.apply(&inputs32), &examples32, &pool32);
        assert!(mrr32 > 0.8, "at 32: {mrr32:.3}");
        // band-limited inputs (modes ≤ 8) so the map is representable at both resolutions
        let inputs64: Vec<Function> = inputs32.iter().map(|f| phistd::phasor::resample(f, phistd::phasor::Resolution::P64)).collect();
        let targets64: Vec<Function> = inputs64.iter().map(|f| { let mut g = f.clone(); g.rotate_right(2); g }).collect();
        let examples64: Vec<Example> = (0..60).map(|i| Example { input: inputs64[i].clone(), target: i }).collect();
        let mut fno64 = fno.at_resolution(64);
        assert_eq!(fno64.config.space, 64);
        assert_eq!(fno64.parameter_count(), fno.parameter_count() + 4 * 4 * (64 - 32));
        let (mrr64, _) = rank(&targets64, &fno64.apply(&inputs64), &examples64, &pool32);
        let (mrr_id, _) = rank(&targets64, &inputs64, &examples64, &pool32);
        assert!(mrr64 > mrr_id + 0.2, "transferred {mrr64:.3} vs identity {mrr_id:.3}");
    }

    #[test]
    fn test_sequential_composes_two_operators_on_one_tape() {
        let cfg = FnoConfig { space: 16, width: 3, layers: 1, modes: 4, margin: 0.2, lr: 1e-2, tie: Tie::Circle };
        let (a, b) = (Fno::new(cfg, 1), Fno::new(cfg, 2));
        let na = a.parameter_count();
        let mut seq = Sequential::new(a, b);
        assert_eq!(seq.parameter_count(), 2 * na);
        let x = seq.tape().constant(stack(&[unit_random(16, 7)], 16));
        let y = seq.forward(x);
        assert_eq!(seq.tape().value(y).shape, [1, 1, 16]);
        let l = seq.tape().mean_real(y);
        seq.tape().backward(l);
        let params = seq.parameters();
        assert!(params.iter().all(|p| seq.first.tape.grad(*p).is_some()), "gradient reaches both operators");
    }

    /// The pointwise control (no spectral path) cannot learn the shift.
    #[test]
    fn test_pointwise_control_cannot_move_information_between_samples() {
        let space = 64;
        let inputs = table(40, space);
        let targets: Vec<Function> = inputs
            .iter()
            .map(|f| {
                let mut g = f.clone();
                g.rotate_right(7);
                g
            })
            .collect();
        let examples: Vec<Example> =
            (0..inputs.len()).map(|i| Example { input: inputs[i].clone(), target: i }).collect();
        let pool: Vec<usize> = (0..inputs.len()).collect();
        let cfg = FnoConfig { space, width: 4, layers: 1, modes: 0, margin: 0.5, lr: 1e-2, tie: Tie::Circle };
        let mut fno = Fno::new(cfg, 1);
        let mut adam = Adam::new(cfg.lr);
        for epoch in 0..40 {
            fno.train_epoch(&targets, &examples, &pool, 20, &mut adam, epoch as u64);
        }
        let preds = fno.apply(&inputs);
        let (mrr, _) = rank(&targets, &preds, &examples, &pool);
        assert!(mrr < 0.3, "pointwise control should fail a shift: mrr {mrr:.3}");
    }

    /// The same operator applies at a different resolution.
    #[test]
    fn test_resolution_free_construction_and_roundtrip() {
        let cfg = FnoConfig { space: 32, width: 3, layers: 2, modes: 4, margin: 0.2, lr: 1e-3, tie: Tie::Circle };
        let mut a = Fno::new(cfg, 42);
        let bytes = a.to_bytes();
        let mut b = Fno::from_bytes(&bytes).expect("roundtrip");
        let x = vec![unit_random(32, 5)];
        let ya = a.apply(&x);
        let yb = b.apply(&x);
        for (p, q) in ya[0].iter().zip(&yb[0]) {
            assert!((p - q).norm() < 1e-12);
        }
        let mut c = Fno::new(FnoConfig::for_space(128), 1);
        assert_eq!(c.apply(&[unit_random(128, 9)])[0].len(), 128);
    }
