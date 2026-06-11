"""Regenerate E2b (SAE testbed) figures from committed metrics (results/e2b/).

Outputs reports/fig_e2b_sae.png:
  panel A: loss curves, baseline vs constrained, per optimizer (the anomaly
           and its removal)
  panel B: ||(W,b)|| trajectories
  panel C: rho_disp on (W,b) per optimizer (directed pursuit test, P12b)
  panel D: loss components for baseline adam (which term diverges)

Usage: python src/fig_e2b.py
"""

import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

runs = defaultdict(list)
for p in sorted(glob.glob("results/e2b/e2b_*.json")):
    d = json.load(open(p))
    runs[(d["config"]["arm"], d["config"]["optimizer"])].append(d)
assert runs, "no E2b metrics found"


def agg(arm, opt, key):
    return np.array([d["metrics"][key] for d in runs[(arm, opt)]])


STYLE = {("baseline", "sgd"): ("C0", "-"), ("baseline", "adam"): ("C1", "-"),
         ("constrained", "sgd"): ("C0", "--"), ("constrained", "adam"): ("C1", "--")}

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
(axA, axB), (axC, axD) = axes

for (arm, opt), (c, ls) in STYLE.items():
    if (arm, opt) not in runs:
        continue
    y = agg(arm, opt, "ep_loss")
    axA.plot(y.mean(0), color=c, ls=ls, lw=1.5, label=f"{arm}/{opt}")
    axA.fill_between(range(y.shape[1]), y.mean(0) - y.std(0),
                     y.mean(0) + y.std(0), color=c, alpha=0.15, lw=0)
axA.set_xlabel("epoch")
axA.set_ylabel("total loss (per-batch sum)")
axA.set_title("A. The 2601 anomaly and its removal")
axA.legend(fontsize=8)

for (arm, opt), (c, ls) in STYLE.items():
    if (arm, opt) not in runs:
        continue
    y = agg(arm, opt, "ep_wb_norm")
    axB.plot(y.mean(0), color=c, ls=ls, lw=1.5, label=f"{arm}/{opt}")
axB.set_xlabel("epoch")
axB.set_ylabel(r"$\|(W,b)\|$")
axB.set_yscale("log")
axB.set_title("B. The scale degree of freedom")
axB.legend(fontsize=8)

for opt, c in [("sgd", "C0"), ("adam", "C1")]:
    y = agg("baseline", opt, "ep_rho_disp_joint")
    axC.plot(y.mean(0), color=c, lw=1.5, label=f"baseline/{opt}")
axC.set_xlabel("epoch")
axC.set_ylabel(r"$\rho_{disp}$ on $(W,b)$")
axC.set_title("C. Directed radial pursuit (P12b)")
axC.legend(fontsize=8)

for key, c in [("ep_lse", "C0"), ("ep_var", "C1"), ("ep_tc", "C2")]:
    y = agg("baseline", "adam", key)
    axD.plot(y.mean(0), color=c, lw=1.5, label=key[3:])
axD.set_xlabel("epoch")
axD.set_ylabel("loss component (baseline/adam)")
axD.set_title("D. Which term diverges")
axD.legend(fontsize=8)

fig.tight_layout()
fig.savefig("reports/fig_e2b_sae.png", dpi=180)
fig.savefig("reports/fig_e2b_sae.pdf")
print("wrote reports/fig_e2b_sae.png")

print(f"\n{'arm':<13}{'opt':<7}{'final loss':<18}{'|Wb| end':<12}"
      f"{'rho_disp lastq':<16}{'probe acc'}")
gaps = {}
for arm in ["baseline", "constrained"]:
    for opt in ["sgd", "adam"]:
        if (arm, opt) not in runs:
            continue
        fl = agg(arm, opt, "ep_loss")[:, -1]
        wn = agg(arm, opt, "ep_wb_norm")[:, -1]
        rd = agg(arm, opt, "ep_rho_disp_joint")
        q = rd.shape[1] // 4
        pa = np.array([d["probe_acc"] for d in runs[(arm, opt)]])
        gaps[(arm, opt)] = fl.mean()
        print(f"{arm:<13}{opt:<7}{fl.mean():9.1f}+-{fl.std():5.1f}  "
              f"{wn.mean():<12.1f}{rd[:, -q:].mean():<16.4f}{pa.mean():.4f}")
for arm in ["baseline", "constrained"]:
    if (arm, "sgd") in gaps and (arm, "adam") in gaps:
        print(f"SGD-Adam final-loss gap, {arm}: "
              f"{gaps[(arm,'sgd')] - gaps[(arm,'adam')]:+.1f}")
