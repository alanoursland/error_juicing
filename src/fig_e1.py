"""Regenerate E1 figures from committed metrics (results/e1/*.json).

Never retrains (README convention). Outputs:
  reports/fig_e1_thesis.png      panel A: the three rho estimators + train error
                                 panel B: ||W|| growth and direction drift (log-t)
  reports/fig_e1_estimators.png  per-seed estimator comparison

Usage: python src/fig_e1.py
"""

import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import smooth

runs = []
for p in sorted(glob.glob("results/e1/e1_seed*.json")):
    d = json.load(open(p))
    if not d["config"].get("head_bias"):
        runs.append(d)
assert runs, "no E1 metrics found; run src/e1_decomposition.py first"

epochs = len(runs[0]["metrics"]["ep_train_err"])
steps_per_epoch = len(runs[0]["metrics"]["rho_global"]) // epochs
ep = np.arange(epochs)


def stack(key):
    return np.array([r["metrics"][key] for r in runs])


def band(ax, x, ys, label, color, ls="-"):
    mean, std = ys.mean(0), ys.std(0)
    ax.plot(x, mean, color=color, ls=ls, label=label, lw=1.8)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, lw=0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# --- Panel A: radial fraction, three estimators, with train error overlay ---
band(ax1, ep, stack("ep_rho_disp_global"), r"$\rho_{disp}$ (epoch displacement)", "C0")
band(ax1, ep, stack("ep_rho_disp_row") - stack("ep_rho_disp_global"),
     r"$\rho_{row}-\rho_{global}$ (disp.)", "C4")
band(ax1, ep, stack("ep_rho_full_global"), r"$\rho_{full}$ (full-batch grad)", "C1")
ps = stack("rho_global").reshape(len(runs), epochs, steps_per_epoch).mean(2)
band(ax1, ep, ps, r"$\rho_{step}$ (per-step minibatch)", "C2")
ax1.axhline(1 / 2560, color="gray", lw=0.8, ls=":", label="chance (1/Kd)")
err = stack("ep_train_err")
sep = np.array([np.argmax(e == 0) for e in err]).mean()
ax1.axvline(sep, color="k", lw=0.8, ls="--", label="train error = 0")
axe = ax1.twinx()
axe.plot(ep, err.mean(0), color="crimson", lw=1.2, alpha=0.7)
axe.set_ylabel("train error", color="crimson")
axe.set_yscale("log")
ax1.set_xlabel("epoch")
ax1.set_ylabel(r"radial fraction $\rho_{global}$")
ax1.set_ylim(-0.02, 1.0)
ax1.legend(fontsize=7, loc="center right")
ax1.set_title("A. Radial fraction: three estimators (MNIST MLP, SGD)")

# --- Panel B: norm growth + direction drift on log-t ---
wn = stack("w_norm")[:, ::steps_per_epoch]
band(ax2, ep, wn, r"$\|W\|_F$", "C0")
ax2.set_xlabel("epoch")
ax2.set_ylabel(r"$\|W\|_F$", color="C0")
ax2.set_title("B. Norm grows; direction converges (drift, log axes)")
axd = ax2.twinx()
drift_steps = np.array(runs[0]["metrics"]["drift_step"]) / steps_per_epoch
dr = stack("drift")
axd.plot(drift_steps, dr.mean(0), color="C3", lw=1.5,
         label=r"$\|\hat W(t)-\hat W(t-k)\|$")
axd.fill_between(drift_steps, dr.mean(0) - dr.std(0), dr.mean(0) + dr.std(0),
                 color="C3", alpha=0.18, lw=0)
axd.set_yscale("log")
axd.set_xscale("log")
axd.set_ylabel("direction drift (log)", color="C3")
ax2.set_xscale("log")

fig.tight_layout()
fig.savefig("reports/fig_e1_thesis.png", dpi=180)
print("wrote reports/fig_e1_thesis.png")

# --- supplementary: per-seed estimator curves ---
fig2, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.4), sharey=True)
for ax, r in zip(np.atleast_1d(axes), runs):
    mm = r["metrics"]
    ax.plot(ep, mm["ep_rho_disp_global"], "C0", label=r"$\rho_{disp}$")
    ax.plot(ep, mm["ep_rho_full_global"], "C1", lw=0.9, label=r"$\rho_{full}$")
    ax.plot(ep, smooth(np.array(mm["rho_global"]).reshape(epochs, -1).mean(1), 5),
            "C2", lw=0.9, label=r"$\rho_{step}$")
    ax.axvline(np.argmax(np.array(mm["ep_train_err"]) == 0), color="k",
               lw=0.8, ls="--")
    ax.set_title(f"seed {r['config']['seed']}", fontsize=9)
    ax.set_xlabel("epoch")
np.atleast_1d(axes)[0].set_ylabel(r"$\rho_{global}$")
np.atleast_1d(axes)[0].legend(fontsize=8)
fig2.tight_layout()
fig2.savefig("reports/fig_e1_estimators.png", dpi=180)
print("wrote reports/fig_e1_estimators.png")
