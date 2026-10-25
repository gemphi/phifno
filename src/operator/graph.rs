//! The operator's forward graph, written once for every module that runs it
//! (`Fno::forward`, and `Sequential`'s second operator on the first one's
//! tape): `h ← σ(F⁻¹(R · F(h)) + W h + b)` per layer, between a lift and a
//! projection. The parameters come in `Fno::params` order: lift, lift bias,
//! then `R, W, b` per layer, then the projection.

use super::{FnoConfig, Tie};
use phistd::{Tape, Var};

/// The forward pass of an operator of `config` with parameters `p` on `t`.
/// Under `Tie::Phase` the biases are left out and `σ` is the
/// phase-preserving `squash` instead of GELU on the real and imaginary
/// parts (docs/154).
pub(super) fn forward(t: &mut Tape, config: &FnoConfig, p: &[Var], x: Var) -> Var {
    let wave = config.tie == Tie::Phase;
    let act = |t: &mut Tape, v: Var| if wave { t.squash(v) } else { t.gelu(v) };
    let bias = |t: &mut Tape, v: Var, b: Var| if wave { v } else { t.add_bias(v, b) };
    let h = t.linear(x, p[0]);
    let h = bias(t, h, p[1]);
    let mut h = act(t, h);
    let groups = config.tie.groups(config.space);
    for l in 0..config.layers {
        let (r, w, b) = (p[2 + 3 * l], p[3 + 3 * l], p[4 + 3 * l]);
        let sum = t.linear(h, w);
        let mut sum = bias(t, sum, b);
        if config.modes > 0 {
            let hs = match &groups {
                None => {
                    let hf = t.fft(h);
                    let hm = t.mode_mix(hf, r);
                    t.ifft(hm)
                }
                Some(groups) => {
                    // The samples are already coefficients: no transform,
                    // one weight per degree tied across its orders.
                    let r = t.tie(r, groups);
                    t.mode_mix(h, r)
                }
            };
            sum = t.add(sum, hs);
        }
        h = if l + 1 < config.layers { act(t, sum) } else { sum };
    }
    t.linear(h, p[2 + 3 * config.layers])
}
