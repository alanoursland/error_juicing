"""Regenerate the E4 figure from committed metrics (results/e4/*.json).

The outward-facing scatter (P10): final test ECE vs integrated radial motion
R = sum of per-epoch radial-weighted logit motion (theory.md S6), one point per
run, colored by arm. Secondary panel: ECE vs the LogitNorm-literature metric
integral d||z||. Prints the Spearman rank correlation P10 is judged on.

Smoke runs (config.smoke == true) are excluded.

Usage: python src/fig_e4.py
"""

import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


points = defaultdict(list)  # arm -> [(R, dz_integral, ece, acc)]
for p in sorted(glob.glob("results/e4/e4_*.json")):
    d = json.load(open(p))
    if d["config"].get("smoke"):
        continue
    m = d["metrics"]
    # skip the first increment: it is dominated by the init transient
    R = float(np.sum(m["ep_R_increment"][1:]))
    dzn = float(np.sum(m["ep_dznorm"][1:]))
    points[d["config"]["arm"]].append(
        (R, dzn, m["ep_test_ece"][-1], m["ep_test_acc"][-1]))
assert points, "no (non-smoke) E4 metrics found; run src/run_e4_all.sh first"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for i, (arm, pts) in enumerate(sorted(points.items())):
    pts = np.array(pts)
    ax1.scatter(pts[:, 0], pts[:, 2], label=arm, color=f"C{i}", s=36)
    ax2.scatter(pts[:, 1], pts[:, 2], label=arm, color=f"C{i}", s=36)
ax1.set_xlabel("integrated radial logit motion R")
ax1.set_ylabel("final test ECE")
ax1.set_title("A. Calibration error vs radial motion (headline)")
ax1.legend(fontsize=8)
ax2.set_xlabel(r"$\int d\|z\|$ (LogitNorm-literature metric)")
ax2.set_ylabel("final test ECE")
ax2.set_title("B. Same, secondary metric")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig("reports/fig_e4_fixes.png", dpi=180)
print("wrote reports/fig_e4_fixes.png")

# P10 statistic: arm-mean ranks (averaged over seeds), then Spearman
arms = sorted(points)
Rm = [np.mean([p[0] for p in points[a]]) for a in arms]
Em = [np.mean([p[2] for p in points[a]]) for a in arms]
print(f"\n{'arm':<18}{'R':<14}{'ECE':<10}{'acc'}")
for a in arms:
    pts = np.array(points[a])
    print(f"{a:<18}{pts[:,0].mean():<14.2f}{pts[:,2].mean():<10.4f}"
          f"{pts[:,3].mean():.4f}")
print(f"\nSpearman(R, ECE) over arm means: {spearman(Rm, Em):+.3f}  "
      f"(P10 threshold: >= +0.8)")
all_pts = np.array([p for a in arms for p in points[a]])
print(f"Spearman over individual runs:   "
      f"{spearman(all_pts[:, 0], all_pts[:, 2]):+.3f}")
