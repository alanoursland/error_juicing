"""Regenerate the E4 figure from committed metrics (results/e4/*.json).

Panels:
  A: final test ECE vs integrated radial motion R (log x) -- the registered
     P10 scatter. The relation is U-shaped, not monotone (see report).
  B: ECE vs post-transient R (epochs >= 40), removing the logitnorm
     init-transient artifact.
  C: ECE vs final probe ||z|| -- the scale the deployed softmax actually sees.

Prints the Spearman statistics P10/P11c are judged on, both as registered and
post-transient. Smoke runs excluded.

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


points = defaultdict(list)  # arm -> [(R, R_post, znorm_end, ece, acc)]
for p in sorted(glob.glob("results/e4/e4_*.json")):
    d = json.load(open(p))
    if d["config"].get("smoke"):
        continue
    m = d["metrics"]
    R = float(np.sum(m["ep_R_increment"][1:]))     # registered: skip init step
    R_post = float(np.sum(m["ep_R_increment"][40:]))  # post-transient
    points[d["config"]["arm"]].append(
        (R, R_post, m["ep_z_norm"][-1], m["ep_test_ece"][-1],
         m["ep_test_acc"][-1]))
assert points, "no (non-smoke) E4 metrics found; run src/run_e4_all.py first"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for i, (arm, pts) in enumerate(sorted(points.items())):
    pts = np.array(pts)
    for ax, col in zip(axes, [0, 1, 2]):
        ax.scatter(pts[:, col], pts[:, 3], label=arm, color=f"C{i}", s=36)
axes[0].set_xlabel("integrated radial logit motion R (registered)")
axes[1].set_xlabel(r"post-transient R (epochs $\geq$ 40)")
axes[2].set_xlabel(r"final probe $\|z\|$")
for ax, t in zip(axes, ["A. P10 scatter (as registered)",
                        "B. Transient removed",
                        "C. Deployed confidence scale"]):
    ax.set_ylabel("final test ECE (raw logits)")
    ax.set_title(t)
    ax.set_xscale("log")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("reports/fig_e4_fixes.png", dpi=180)
print("wrote reports/fig_e4_fixes.png")

arms = sorted(points)
print(f"\n{'arm':<18}{'R':<12}{'R_post40':<12}{'|z| end':<10}{'ECE':<9}{'acc'}")
for a in arms:
    pts = np.array(points[a])
    print(f"{a:<18}{pts[:,0].mean():<12.1f}{pts[:,1].mean():<12.1f}"
          f"{pts[:,2].mean():<10.1f}{pts[:,3].mean():<9.4f}{pts[:,4].mean():.4f}")

for label, col in [("R (registered)", 0), ("R post-transient", 1),
                   ("final |z|", 2)]:
    Rm = [np.mean([p[col] for p in points[a]]) for a in arms]
    Em = [np.mean([p[3] for p in points[a]]) for a in arms]
    sub = [a for a in arms if a != "temperature"]
    Rs = [np.mean([p[col] for p in points[a]]) for a in sub]
    Es = [np.mean([p[3] for p in points[a]]) for a in sub]
    print(f"Spearman(ECE, {label:<17}): all arms {spearman(Rm, Em):+.3f}   "
          f"ex-temperature {spearman(Rs, Es):+.3f}")
