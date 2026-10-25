//! Save and load: v2 with optimiser, legacy v1, compact f32.

use super::super::*;
use super::super::save::{SavedParam, SavedV1};
use phistd::AdamState;
use phistd::Adam;
use phistd::{unit_random, Function};

    fn table(n: usize, space: usize) -> Vec<Function> {
        (0..n).map(|i| unit_random(space, i as u64 + 1)).collect()
    }

    /// The operator must learn a fixed relation the identity cannot express:
    /// a circular shift by one sample, which is exactly a spectral
    fn shift_task(space: usize, n: usize, by: usize) -> (Vec<Function>, Vec<Function>, Vec<Example>, Vec<usize>) {
        let inputs = table(n, space);
        let targets: Vec<Function> = inputs.iter().map(|f| { let mut g = f.clone(); g.rotate_right(by); g }).collect();
        let examples: Vec<Example> = (0..n).map(|i| Example { input: inputs[i].clone(), target: i }).collect();
        (inputs, targets, examples, (0..n).collect())
    }


    #[test]
    fn test_save_load_v2_with_optimiser_and_legacy_v1() {
        let cfg = FnoConfig { space: 16, width: 3, layers: 2, modes: 4, margin: 0.2, lr: 1e-2, tie: Tie::Circle };
        let mut fno = Fno::new(cfg, 5);
        let (inputs, targets, examples, pool) = shift_task(16, 20, 1);
        let mut adam = Adam::new(cfg.lr).with_decay(0.01);
        fno.train_epoch(&targets, &examples, &pool, 10, &mut adam, 0);
        let bytes = fno.to_bytes_with(&adam);
        assert!(bytes.starts_with(b"PHINET"));
        let (again, adam2) = Fno::from_bytes_with(&bytes).unwrap();
        let adam2 = adam2.expect("optimiser saved");
        assert_eq!(adam2.steps(), adam.steps());
        assert_eq!(adam2.weight_decay, 0.01);
        let mut again = again;
        for (a, b) in fno.apply(&inputs).iter().zip(again.apply(&inputs)) {
            for (x, y) in a.iter().zip(&b) { assert!((x - y).norm() < 1e-12); }
        }
        // a pre-version file (no header) still loads
        let legacy = bincode::serialize(&SavedV1 { config: cfg, params: fno.saved_params() }).unwrap();
        let old = Fno::from_bytes(&legacy).unwrap();
        assert_eq!(old.parameter_count(), fno.parameter_count());
        assert!(Fno::from_bytes(b"PHINET\x09\x00junk").is_none(), "unknown version is refused");
    }

    #[test]
    fn test_compact_save_halves_the_file_and_barely_moves_the_output() {
        let cfg = FnoConfig { space: 64, width: 8, layers: 2, modes: 12, margin: 0.2, lr: 2e-3, tie: Tie::Circle };
        let mut fno = Fno::new(cfg, 3);
        let full = fno.to_bytes();
        let compact = fno.to_bytes_compact();
        assert!(compact.len() * 2 <= full.len() + 512, "{} vs {}", compact.len(), full.len());
        let mut again = Fno::from_bytes(&compact).unwrap();
        let inputs = table(50, 64);
        let (a, b) = (fno.apply(&inputs), again.apply(&inputs));
        let worst = a.iter().zip(&b).map(|(x, y)| 1.0 - cosine(x, y)).fold(0.0, f64::max);
        assert!(worst < 1e-10, "round-trip drift {worst:e}");
    }


#[test]
fn a_file_saved_before_tie_existed_still_loads_as_a_circle() {
    // Serialise the pre-`tie` layout by hand and load it through the public path.
    use serde::Serialize;
    #[derive(Serialize)]
    struct OldConfig { space: usize, width: usize, layers: usize, modes: usize, margin: f64, lr: f64 }
    #[derive(Serialize)]
    struct OldSaved { config: OldConfig, params: Vec<SavedParam>, adam: Option<AdamState> }
    let cfg = FnoConfig { space: 16, width: 3, layers: 1, modes: 4, margin: 0.2, lr: 1e-2, tie: Tie::Circle };
    let fno = Fno::new(cfg, 5);
    let old = OldSaved {
        config: OldConfig { space: 16, width: 3, layers: 1, modes: 4, margin: 0.2, lr: 1e-2 },
        params: fno.saved_params(),
        adam: None,
    };
    let mut bytes = b"PHINET".to_vec();
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend(bincode::serialize(&old).unwrap());
    let back = Fno::from_bytes(&bytes).expect("old layout loads");
    assert_eq!(back.config.tie, Tie::Circle);
    assert_eq!(back.parameter_count(), fno.parameter_count());
}
