"""E4 deployment-corrected calibration figure (results/e4/ece_corrected.json).

Panels:
  A: deployed ECE vs signed log T* (post-hoc optimal temperature). Sign
     separates over-confident (log T* > 0) from under-confident arms; the
     two-sidedness of P10's U-shape, on its proper axis.
  B: deployed ECE vs ECE after post-hoc temperature scaling, per arm. The
     gap to the diagonal is the scale-attributable share of miscalibration;
     the height of ece@T* is the scale-irreparable (shape) residual.

Usage: python src/fig_e4_corrected.py
"""

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

recs = [r for r in json.load(open("results/e4/ece_corrected.json"))]
by_arm = defaultdict(list)
for r in recs:
    by_arm[r["arm"]].append(r)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

for i, (arm, rs) in enumerate(sorted(by_arm.items())):
    x = [r["log_Tstar"] for r in rs]
    y = [r["ece_deployed"] for r in rs]
    ax1.scatter(x, y, color=f"C{i}", s=36, label=arm)
ax1.axvline(0, color="gray", lw=0.8, ls=":")
ax1.text(0.05, 0.93, "overconfident →", transform=ax1.transAxes, fontsize=8)
ax1.text(0.05, 0.05, "← underconfident: label smoothing only", fontsize=8,
         transform=ax1.transAxes)
ax1.set_xlabel(r"signed $\log T^{*}$ (post-hoc optimal temperature)")
ax1.set_ylabel("deployed test ECE")
ax1.set_title("A. Two-sided miscalibration on the scale axis")
ax1.legend(fontsize=7)

arms = sorted(by_arm)
xpos = np.arange(len(arms))
dep = [np.mean([r["ece_deployed"] for r in by_arm[a]]) for a in arms]
fix = [np.mean([r["ece_at_Tstar"] for r in by_arm[a]]) for a in arms]
ax2.bar(xpos - 0.2, dep, 0.4, label="deployed ECE", color="C0")
ax2.bar(xpos + 0.2, fix, 0.4, label=r"ECE at post-hoc $T^{*}$", color="C1")
for x, d, f in zip(xpos, dep, fix):
    ax2.text(x, max(d, f) + 0.002, f"{100*(d-f)/d:.0f}%", ha="center",
             fontsize=7)
ax2.set_xticks(xpos, [a.replace("_", "\n") for a in arms], fontsize=7)
ax2.set_ylabel("test ECE")
ax2.set_title("B. One global scalar repairs most of every arm\n"
              "(percent shown: scale-attributable share)")
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig("reports/fig_e4_corrected.png", dpi=180)
fig.savefig("reports/fig_e4_corrected.pdf")
print("wrote reports/fig_e4_corrected.{png,pdf}")

print(f"\n{'arm':<17}{'ece_depl':<10}{'ece@T*':<9}{'removable%':<12}"
      f"{'T_learned':<11}{'T*':<7}{'sign'}")
for a in arms:
    rs = by_arm[a]
    f = lambda k: np.mean([r[k] for r in rs])
    rem = 100 * (f("ece_deployed") - f("ece_at_Tstar")) / f("ece_deployed")
    side = "under" if f("T_star") < 1 else "over"
    print(f"{a:<17}{f('ece_deployed'):<10.4f}{f('ece_at_Tstar'):<9.4f}"
          f"{rem:<12.0f}{f('T_learned'):<11.3f}{f('T_star'):<7.2f}{side}")
