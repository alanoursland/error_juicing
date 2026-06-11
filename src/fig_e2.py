"""Regenerate E2 figures from committed metrics (results/e2/*.json).

Outputs reports/fig_e2_intervention.png:
  panel A: final train loss by arm x optimizer (the structural floor and the
           Adam-SGD gap)
  panel B: train-loss curves (mean over seeds)
  panel C: final test ECE by arm x optimizer
  panel D: ||W||_F trajectories (constraint sanity check)

Usage: python src/fig_e2.py
"""

import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ["baseline", "global", "row", "full"]
OPTS = ["sgd", "adam"]

runs = defaultdict(list)  # (arm, opt) -> [metrics]
for p in sorted(glob.glob("results/e2/e2_*.json")):
    d = json.load(open(p))
    runs[(d["config"]["arm"], d["config"]["optimizer"])].append(d["metrics"])
assert runs, "no E2 metrics found"
ARMS = [a for a in ARMS if any(k[0] == a for k in runs)]


def agg(arm, opt, key):
    return np.array([m[key] for m in runs[(arm, opt)]])


fig, axes = plt.subplots(2, 2, figsize=(11, 8))
(axA, axB), (axC, axD) = axes

x = np.arange(len(ARMS))
width = 0.35
for i, opt in enumerate(OPTS):
    finals = [agg(a, opt, "ep_train_loss")[:, -1] for a in ARMS]
    axA.bar(x + (i - 0.5) * width, [f.mean() for f in finals], width,
            yerr=[f.std() for f in finals], label=opt, capsize=3)
axA.set_yscale("log")
axA.set_xticks(x, ARMS)
axA.set_ylabel("final train loss (log)")
axA.set_title("A. Structural floor and the Adam-SGD gap")
axA.legend()

PAIR_COLORS = [f"C{i}" for i in range(len(ARMS) * len(OPTS))]
for (arm, opt), color in zip([(a, o) for a in ARMS for o in OPTS], PAIR_COLORS):
    curves = agg(arm, opt, "ep_train_loss")
    axB.plot(curves.mean(0), color=color, lw=1.4, label=f"{arm}/{opt}")
axB.set_yscale("log")
axB.set_xlabel("epoch")
axB.set_ylabel("train loss (log)")
axB.set_title("B. Loss curves")
axB.legend(fontsize=8)

for i, opt in enumerate(OPTS):
    finals = [agg(a, opt, "ep_test_ece")[:, -1] for a in ARMS]
    axC.bar(x + (i - 0.5) * width, [f.mean() for f in finals], width,
            yerr=[f.std() for f in finals], label=opt, capsize=3)
axC.set_xticks(x, ARMS)
axC.set_ylabel("final test ECE")
axC.set_title("C. Calibration")
axC.legend()

for (arm, opt), color in zip([(a, o) for a in ARMS for o in OPTS], PAIR_COLORS):
    wn = agg(arm, opt, "w_norm")
    axD.plot(wn.mean(0), color=color, lw=1.2, label=f"{arm}/{opt}")
axD.set_xlabel("step")
axD.set_ylabel(r"$\|W\|_F$")
axD.set_title("D. Constraint check: head norm")
axD.legend(fontsize=8)

fig.tight_layout()
fig.savefig("reports/fig_e2_intervention.png", dpi=180)
fig.savefig("reports/fig_e2_intervention.pdf")
print("wrote reports/fig_e2_intervention.png")

# console summary for the lab report
print(f"\n{'arm':<10}{'opt':<7}{'train loss':<14}{'test acc':<12}{'ECE':<10}")
for arm in ARMS:
    for opt in OPTS:
        tl = agg(arm, opt, "ep_train_loss")[:, -1]
        ac = agg(arm, opt, "ep_test_acc")[:, -1]
        ec = agg(arm, opt, "ep_test_ece")[:, -1]
        print(f"{arm:<10}{opt:<7}{tl.mean():.4f}+-{tl.std():.4f}  "
              f"{ac.mean():.4f}+-{ac.std():.4f}  {ec.mean():.4f}+-{ec.std():.4f}")
for arm in ARMS:
    gap = (agg(arm, "sgd", "ep_train_loss")[:, -1].mean()
           - agg(arm, "adam", "ep_train_loss")[:, -1].mean())
    print(f"SGD-Adam final-loss gap, {arm}: {gap:+.4f}")
