//! The spectral mask and the per-mode prior applied as projections after a step.

use super::Fno;
use phistd::c64;

impl Fno {
    /// Re-zeroes the masked modes, which gradient steps otherwise fill.
    pub fn mask_modes(&mut self) {
        for l in &self.layers {
            let mut r = self.tape.value(l.r).clone();
            for i in 0..r.shape[0] {
                for o in 0..r.shape[1] {
                    for m in 0..r.shape[2] {
                        if !self.mask[m] {
                            r.set(i, o, m, c64::new(0.0, 0.0));
                        }
                    }
                }
            }
            self.tape.set_value(l.r, r);
        }
    }

    /// Scales every layer's spectral weights by a per-mode factor — a
    /// spectral prior applied as a projection after each optimiser step.
    /// Modes are indexed by frequency: `weights[m]` multiplies mode `m`,
    /// with negative frequencies (the upper half of the FFT) sharing the
    /// weight of their positive partner.
    pub fn scale_modes(&mut self, weights: &[f64]) {
        let space = self.config.space;
        for l in &self.layers {
            let mut r = self.tape.value(l.r).clone();
            for i in 0..r.shape[0] {
                for o in 0..r.shape[1] {
                    for m in 0..r.shape[2] {
                        let f = if m <= space / 2 { m } else { space - m };
                        let w = weights.get(f).copied().unwrap_or(0.0);
                        let v = r.get(i, o, m);
                        r.set(i, o, m, v * w);
                    }
                }
            }
            self.tape.set_value(l.r, r);
        }
    }
}
