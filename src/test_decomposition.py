"""Unit tests for the decomposition math, and nothing else (README convention).

Covers exactly the properties of notes/theory.md S3.3:
  1. orthogonality      <G_rad, G_tan> = 0
  2. Pythagoras          ||G_rad||^2 + ||G_tan||^2 = ||G||^2, rho in [0,1]
  3. subspace scale-invariance under W -> alpha*W
  4. nesting             rho_row >= rho_global
  5. purity              G prop W => rho = 1; row-orthogonal G => rho = 0

Run: python src/test_decomposition.py
Exact properties, exact tolerances: float64, atol 1e-10.
"""

import sys

import torch

from common import (
    decompose_global,
    decompose_row,
    logit_radial_step,
    rho_global,
    rho_global_joint,
    rho_row,
)

torch.set_default_dtype(torch.float64)
ATOL = 1e-10
FAILURES = []


def check(name, cond):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAILURES.append(name)


def random_pairs(n=20, K=10, d=64):
    gen = torch.Generator().manual_seed(0)
    for _ in range(n):
        yield torch.randn(K, d, generator=gen), torch.randn(K, d, generator=gen)


print("1. orthogonality")
for W, G in random_pairs():
    for dec in (decompose_global, decompose_row):
        r, t = dec(G, W)
        check(f"{dec.__name__}: <rad,tan>=0", abs(float((r * t).sum())) < ATOL)

print("2. Pythagoras and range")
for W, G in random_pairs():
    for dec, rho in ((decompose_global, rho_global), (decompose_row, rho_row)):
        r, t = dec(G, W)
        pyth = abs(float(r.norm() ** 2 + t.norm() ** 2 - G.norm() ** 2))
        check(f"{dec.__name__}: pythagoras", pyth < ATOL)
        v = rho(G, W)
        check(f"{rho.__name__}: in [0,1]", -ATOL <= v <= 1 + ATOL)

print("3. subspace scale-invariance under W -> alpha*W")
for W, G in random_pairs(n=10):
    for alpha in (0.01, 3.7, 250.0):
        for dec, rho in ((decompose_global, rho_global), (decompose_row, rho_row)):
            r1, t1 = dec(G, W)
            r2, t2 = dec(G, alpha * W)
            check(f"{dec.__name__}: tan invariant (a={alpha})",
                  float((t1 - t2).abs().max()) < ATOL)
            check(f"{rho.__name__}: rho invariant (a={alpha})",
                  abs(rho(G, W) - rho(G, alpha * W)) < ATOL)

print("4. nesting rho_row >= rho_global")
for W, G in random_pairs(n=50):
    check("nesting", rho_row(G, W) >= rho_global(G, W) - ATOL)

print("5. purity")
for W, _ in random_pairs(n=5):
    check("G=cW: rho_global=1", abs(rho_global(2.5 * W, W) - 1) < ATOL)
    check("G=cW: rho_row=1", abs(rho_row(2.5 * W, W) - 1) < ATOL)
    # build G with every row orthogonal to the matching row of W
    G = torch.randn(*W.shape, generator=torch.Generator().manual_seed(1))
    Wn = W / W.norm(dim=1, keepdim=True)
    G = G - (G * Wn).sum(dim=1, keepdim=True) * Wn
    check("row-orthogonal G: rho_row=0", rho_row(G, W) < ATOL)
    check("row-orthogonal G: rho_global=0", rho_global(G, W) < ATOL)

print("6. joint (W,b) projection (sensitivity arm)")
for W, G in random_pairs(n=5):
    b = torch.randn(W.shape[0], generator=torch.Generator().manual_seed(2))
    # gradient proportional to (W, b) is pure scale: rho = 1
    check("G=(cW,cb): rho_joint=1",
          abs(rho_global_joint(3.0 * W, W, 3.0 * b, b) - 1) < ATOL)

print("7. logit-level estimator sanity")
z = torch.randn(32, 10, generator=torch.Generator().manual_seed(3))
rho_l, _, _ = logit_radial_step(z, 1.5 * z)  # pure scaling of logits
check("pure z-scaling: rho_logit=1", abs(rho_l - 1) < ATOL)
dz = torch.randn(32, 10, generator=torch.Generator().manual_seed(4))
zh = z / z.norm(dim=1, keepdim=True)
dz = dz - (zh * dz).sum(dim=1, keepdim=True) * zh  # tangential displacement
rho_l, _, _ = logit_radial_step(z, z + dz)
check("tangential dz: rho_logit=0", rho_l < ATOL)

if FAILURES:
    print(f"\n{len(FAILURES)} FAILURES")
    sys.exit(1)
print("\nall tests passed")
