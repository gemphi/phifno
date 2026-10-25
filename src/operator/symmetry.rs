//! How far an operator is from the two symmetries a wave reading would
//! want (docs/153): commuting with a global phase, `F(e^{iφ}x) = e^{iφ}F(x)`,
//! and with conjugation, `F(x̄) = conj F(x)`. An operator with both reads a
//! phase-shifted wave as the same wave shifted, and the conjugate wave as
//! the time-reversed one.

use super::Fno;
use phistd::{c64, Function};

impl Fno {
    /// The mean over `inputs` of `‖F(e^{iφ}x) − e^{iφ}F(x)‖ / ‖F(x)‖` and of
    /// `‖F(x̄) − conj F(x)‖ / ‖F(x)‖`: `(phase, conjugation)`, both 0 for an
    /// operator with the symmetry.
    pub fn defects(&mut self, inputs: &[Function], phi: f64) -> (f64, f64) {
        let r = c64::from_polar(1.0, phi);
        let (mut phase, mut conj) = (0.0, 0.0);
        for x in inputs {
            let ys = self.apply(&[x.clone(), x.iter().map(|z| z * r).collect(), x.iter().map(|z| z.conj()).collect()]);
            let size = ys[0].iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt().max(1e-12);
            let gap = |a: &Function, b: &dyn Fn(&c64) -> c64| a.iter().zip(&ys[0]).map(|(p, q)| (p - b(q)).norm_sqr()).sum::<f64>().sqrt() / size;
            phase += gap(&ys[1], &|q| q * r);
            conj += gap(&ys[2], &|q| q.conj());
        }
        let n = inputs.len().max(1) as f64;
        (phase / n, conj / n)
    }
}

#[cfg(test)]
mod tests {
    use phistd::{unit_random};
    use crate::operator::{Fno, FnoConfig, Tie};

    #[test]
    fn the_wave_tie_commutes_with_a_global_phase_and_the_circle_does_not() {
        let inputs: Vec<_> = (0..8).map(|i| unit_random(32, i)).collect();
        let cfg = FnoConfig { space: 32, width: 6, layers: 2, modes: 6, margin: 0.3, lr: 1e-2, tie: Tie::Phase };
        let mut wave = Fno::new(cfg, 7);
        let (phase, _) = wave.defects(&inputs, 1.0);
        assert!(phase < 1e-12, "Tie::Phase: F(e^{{iφ}}x) = e^{{iφ}}F(x), defect {phase}");
        let mut circle = Fno::new(FnoConfig { tie: Tie::Circle, ..cfg }, 7);
        let (phase, conj) = circle.defects(&inputs, 1.0);
        assert!(phase > 0.01 && conj > 0.01, "the circle keeps neither: {phase} {conj}");
        let back = Fno::from_bytes(&wave.to_bytes()).expect("reads back");
        assert_eq!(back.config.tie, Tie::Phase, "the tie survives a save");
    }
}
