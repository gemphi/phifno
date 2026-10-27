//! The same operator at another resolution: on the circle another number of
//! samples, on the sphere another number of degrees (the counterpart of
//! `sphere::degree_gate`).

use super::Fno;
use phistd::Tensor;

impl Fno {
    /// The same operator at `space` samples. Lift, pointwise weights, biases
    /// and projection do not depend on `space`; only `R` is remapped.
    ///
    /// On the circle (`Tie::Circle`, `Tie::Phase`, `space` a power of two)
    /// `R` is stored per signed frequency, and the kept low frequencies exist
    /// at every power-of-two `space`, so they are copied across and the rest
    /// stays zero — what the FNO literature means by resolution invariance;
    /// phical's `ranges` example measures how far it holds.
    ///
    /// On the sphere (`Tie::Sphere`, `Tie::Anisotropic`, `space` a perfect
    /// square) `R` is one weight per group, and groups are numbered degree
    /// first — the same number is the same `l` (or `(l, |m|)`) at every size
    /// — so the groups both resolutions have are copied and the new ones
    /// start at zero. `modes` counts degrees there, capped at the new count.
    pub fn at_resolution(&self, space: usize) -> Fno {
        let mut cfg = self.config;
        cfg.space = space;
        cfg.modes = match cfg.tie.groups(space) {
            None => cfg.modes.min(space / 2),
            Some(_) => cfg.modes.min((space as f64).sqrt() as usize),
        };
        let mut out = Fno::new(cfg, 0);
        let (old_s, wd, layers) = (self.config.space, self.config.width, self.config.layers);
        let sphere = self.config.tie.groups(old_s).is_some();
        let src = self.params();
        let dst = out.params();
        for (i, (sp, dp)) in src.iter().zip(&dst).enumerate() {
            let sv = self.tape.value(*sp).clone();
            // The parameters run lift, lift bias, then (R, W, b) per layer, then the projection.
            let is_r = i >= 2 && i < 2 + 3 * layers && (i - 2) % 3 == 0;
            let r = match (is_r, sphere) {
                (false, _) => sv,
                (true, false) => circle(&sv, wd, old_s, space),
                (true, true) => {
                    let slots = out.tape.value(*dp).shape[2];
                    let mut r = Tensor::zeros([wd, wd, slots]);
                    for a in 0..wd {
                        for b in 0..wd {
                            for g in 0..slots.min(sv.shape[2]) {
                                r.set(a, b, g, sv.get(a, b, g));
                            }
                        }
                    }
                    r
                }
            };
            out.tape.set_value(*dp, r);
        }
        out.mask_modes();
        out
    }
}

/// `R` from `old_s` to `space` samples on the circle, frequency by signed
/// frequency.
fn circle(sv: &Tensor, wd: usize, old_s: usize, space: usize) -> Tensor {
    let mut r = Tensor::zeros([wd, wd, space]);
    for a in 0..wd {
        for b in 0..wd {
            for m in 0..space {
                let f: i64 = match m <= space / 2 {
                    true => m as i64,
                    false => m as i64 - space as i64,
                };
                if f.unsigned_abs() as usize > old_s / 2 {
                    continue;
                }
                let om = match f >= 0 {
                    true => f as usize,
                    false => (old_s as i64 + f) as usize,
                };
                r.set(a, b, m, sv.get(a, b, om));
            }
        }
    }
    r
}
