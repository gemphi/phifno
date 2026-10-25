//! Dump the 64 real harmonics on the minimal grid as TSV, for a parity
//! check against an independent implementation (scipy.special.sph_harm).
use phifno::sphere::{real_harmonic, Sphere, HARMONICS};
fn main() {
    let s = Sphere::minimal();
    println!("i\ttheta\tphi\tweight\t{}", (0..HARMONICS).map(|k| format!("Y{k}")).collect::<Vec<_>>().join("\t"));
    for i in 0..s.grid.len() {
        let (t, p, w) = s.grid.at(i);
        let ys: Vec<String> = (0..HARMONICS).map(|k| format!("{:.15e}", real_harmonic(k, t, p))).collect();
        println!("{i}\t{t:.15e}\t{p:.15e}\t{w:.15e}\t{}", ys.join("\t"));
    }
}
