"""Regenerate E3 figures from committed metrics (results/e3/*.json).

Outputs reports/fig_e3_optimizer.png:
  panel A: epoch-displacement rho_global per optimizer (mean over seeds and
           lrs) -- the step-budget allocation plot
  panel B: last-quarter mean rho_disp per optimizer x lr (bar)
  panel C: ||W||_F growth per optimizer

Usage: python src/fig_e3.py
"""

import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

runs = defaultdict(list)  # (optimizer, lr) -> [metrics]
for p in sorted(glob.glob("results/e3/e3_*.json")):
    d = json.load(open(p))
    runs[(d["config"]["optimizer"], d["config"]["lr"])].append(d["metrics"])
assert runs, "no E3 metrics found"

OPT_COLOR = {"sgd": "C0", "adam": "C1", "adamw": "C2"}

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14, 4.2))

for (opt, lr), ms in sorted(runs.items()):
    rd = np.array([m["ep_rho_disp_global"] for m in ms])
    axA.plot(rd.mean(0), color=OPT_COLOR[opt], lw=1.4,
             ls="-" if lr == max(l for o, l in runs if o == opt) else "--",
             label=f"{opt} lr={lr:g}")
axA.set_xlabel("epoch")
axA.set_ylabel(r"$\rho_{disp}$ (epoch displacement, global)")
axA.set_title("A. Radial share of realized travel")
axA.legend(fontsize=8)

labels, means, errs, colors = [], [], [], []
for (opt, lr), ms in sorted(runs.items()):
    rd = np.array([m["ep_rho_disp_global"] for m in ms])
    q = rd.shape[1] // 4
    vals = rd[:, -q:].mean(1)
    labels.append(f"{opt}\n{lr:g}")
    means.append(vals.mean())
    errs.append(vals.std())
    colors.append(OPT_COLOR[opt])
axB.bar(range(len(labels)), means, yerr=errs, color=colors, capsize=3)
axB.set_xticks(range(len(labels)), labels, fontsize=8)
axB.set_ylabel(r"mean $\rho_{disp}$, last 25%")
axB.set_title("B. Last-quarter radial allocation")

for (opt, lr), ms in sorted(runs.items()):
    wn = np.array([m["w_norm"] for m in ms])
    axC.plot(wn.mean(0), color=OPT_COLOR[opt], lw=1.0,
             ls="-" if lr == max(l for o, l in runs if o == opt) else "--",
             label=f"{opt} lr={lr:g}")
axC.set_xlabel("step")
axC.set_ylabel(r"$\|W\|_F$")
axC.set_title("C. Head norm growth")
axC.legend(fontsize=8)

fig.tight_layout()
fig.savefig("reports/fig_e3_optimizer.png", dpi=180)
print("wrote reports/fig_e3_optimizer.png")

print(f"\n{'opt':<7}{'lr':<9}{'rho_disp lastq':<17}{'rho_step lastq':<17}"
      f"{'|W| end':<9}{'test acc'}")
for (opt, lr), ms in sorted(runs.items()):
    rd = np.array([m["ep_rho_disp_global"] for m in ms])
    rs = np.array([m["rho_step_global"] for m in ms])
    q = rd.shape[1] // 4
    qs = rs.shape[1] // 4
    wn = np.array([m["w_norm"][-1] for m in ms])
    ac = np.array([m["ep_test_acc"][-1] for m in ms])
    print(f"{opt:<7}{lr:<9g}{rd[:, -q:].mean():.4f}+-{rd[:, -q:].mean(1).std():.4f}  "
          f"{rs[:, -qs:].mean():.5f}        {wn.mean():<9.2f}{ac.mean():.4f}")
