"""Parity of phinet::sphere::real_harmonic against scipy.special.sph_harm_y (scipy >= 1.15).

Run:  cargo run --release --example sphere_dump > sphere.tsv && python3 examples/sphere_parity.py sphere.tsv
Real harmonics from the complex ones (Condon–Shortley phase removed):
  m = 0 : Y_l^0
  m > 0 : sqrt2 * (-1)^m * Re Y_l^m
  m < 0 : sqrt2 * (-1)^m * Im Y_l^|m|
"""
import sys, numpy as np
from scipy.special import sph_harm_y
rows = np.loadtxt(sys.argv[1], skiprows=1)
theta, phi, w = rows[:, 1], rows[:, 2], rows[:, 3]
ys = rows[:, 4:]
worst = 0.0
for k in range(64):
    l = int(np.floor(np.sqrt(k))); m = k - l * l - l
    z = sph_harm_y(l, abs(m), theta, phi)        # scipy >= 1.15: (degree, order, polar, azimuth)
    if m == 0:
        ref = z.real
    elif m > 0:
        ref = np.sqrt(2) * (-1) ** m * z.real
    else:
        ref = np.sqrt(2) * (-1) ** m * z.imag
    worst = max(worst, np.max(np.abs(ys[:, k] - ref)))
# quadrature orthonormality with the grid's own weights
G = (ys * w[:, None]).T @ ys
orth = np.max(np.abs(G - np.eye(64)))
print(f"max |phinet - scipy| over 64 harmonics x {len(theta)} points: {worst:.3e}")
print(f"max |<Y_a,Y_b> - delta| under the grid quadrature: {orth:.3e}")
assert worst < 1e-12 and orth < 1e-9
print("parity: ok")
