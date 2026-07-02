"""Shared utilities for the error-juicing experiments.

Implements exactly what notes/theory.md specifies:
  - radial/tangential decomposition, global (S3.1) and per-row (S3.2)
  - rho_grad / rho_step are the same functions applied to G or delta-W (S4)
  - logit-level radial fraction estimator (S6)
  - ECE, models, data, seeding, metric I/O

Research code, not a library: plain functions, no classes beyond nn.Module.
"""

import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Decomposition (theory.md S3). G is a gradient or an update at parameter W.
# ---------------------------------------------------------------------------


def decompose_global(G, W):
    """Project G onto span{W} (1-D, Frobenius). Returns (G_rad, G_tan)."""
    Wf, Gf = W.flatten(), G.flatten()
    Wn = Wf / Wf.norm()
    G_rad = ((Gf @ Wn) * Wn).view_as(G)
    return G_rad, G - G_rad


def decompose_row(G, W):
    """Project G onto span{e_j w_j^T} (K-D, per-row). Returns (G_rad, G_tan)."""
    Wn = W / W.norm(dim=1, keepdim=True)
    G_rad = (G * Wn).sum(dim=1, keepdim=True) * Wn
    return G_rad, G - G_rad


def rho_global(G, W):
    """Radial fraction ||proj_span{W} G||^2 / ||G||^2."""
    Wf, Gf = W.detach().flatten(), G.detach().flatten()
    den = Gf @ Gf
    if den == 0:
        return 0.0
    c = (Gf @ Wf) / Wf.norm()
    return float(c * c / den)


def rho_row(G, W):
    """Radial fraction onto the per-row subspace. rho_row >= rho_global (S3.3.4)."""
    G, W = G.detach(), W.detach()
    den = (G * G).sum()
    if den == 0:
        return 0.0
    Wn = W / W.norm(dim=1, keepdim=True)
    c = (G * Wn).sum(dim=1)
    return float((c * c).sum() / den)


def rho_global_joint(G, W, gb, b):
    """Sensitivity arm only (theory.md S1): joint (W, b) scaling generates
    z -> alpha*z when the head has a bias; project the concatenated gradient
    onto the concatenated parameter."""
    p = torch.cat([W.detach().flatten(), b.detach().flatten()])
    g = torch.cat([G.detach().flatten(), gb.detach().flatten()])
    den = g @ g
    if den == 0:
        return 0.0
    c = (g @ p) / p.norm()
    return float(c * c / den)


# ---------------------------------------------------------------------------
# Logit-level estimator (theory.md S6)
# ---------------------------------------------------------------------------


def logit_radial_step(z_prev, z_curr):
    """Per-example radial fraction of dz between consecutive logged points,
    and the radial-weighted / unweighted norm motion.

    Returns (mean rho_logit, mean rho-weighted ||dz||, mean d||z||) over the
    probe batch. Mean-per-step estimator (stabler; theory.md S6).
    """
    dz = z_curr - z_prev
    dz_norm = dz.norm(dim=1)
    keep = dz_norm > 0
    if keep.sum() == 0:
        return 0.0, 0.0, 0.0
    z_hat = z_prev / z_prev.norm(dim=1, keepdim=True)
    coef = (z_hat * dz).sum(dim=1)
    rho = (coef**2 / dz_norm**2)[keep]
    weighted = (rho * dz_norm[keep]).mean()
    dznorm = (z_curr.norm(dim=1) - z_prev.norm(dim=1)).mean()
    return float(rho.mean()), float(weighted), float(dznorm)


def responsibility_entropy(logits):
    """Mean per-example softmax entropy (nats)."""
    logp = F.log_softmax(logits, dim=1)
    return float(-(logp.exp() * logp).sum(dim=1).mean())


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def ece(logits, labels, n_bins=15):
    """Standard expected calibration error, equal-width confidence bins."""
    conf, pred = F.softmax(logits, dim=1).max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += float(m.float().mean() * (correct[m].mean() - conf[m].mean()).abs())
    return total


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Small MLP. Head is bias-free by default (theory.md S1)."""

    def __init__(self, hidden=256, num_classes=10, head_bias=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, hidden), nn.ReLU()
        )
        self.head = nn.Linear(hidden, num_classes, bias=head_bias)

    def forward(self, x):
        return self.head(self.body(x))


# ---------------------------------------------------------------------------
# Data: MNIST fully in memory (fast on CPU; 60k x 784 floats ~ 188 MB)
# ---------------------------------------------------------------------------

# _DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_DATA_DIR = "E:/ml_datasets"


def mnist_tensors(train=True):
    import torchvision

    ds = torchvision.datasets.MNIST(_DATA_DIR, train=train, download=True)
    x = ds.data.float().div_(255.0).sub_(0.1307).div_(0.3081).unsqueeze(1)
    y = ds.targets.clone()
    return x, y


# ---------------------------------------------------------------------------
# Constraints (E2/E4 interventions): post-step projection, NOT weight-norm
# reparametrization (theory.md / README E2: remove the radial DOF from the
# iterate without changing the gradient geometry).
# ---------------------------------------------------------------------------


@torch.no_grad()
def project_rows_unit(W, value=1.0):
    W.mul_(value / W.norm(dim=1, keepdim=True))


@torch.no_grad()
def project_global_norm(W, value):
    W.mul_(value / W.norm())


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def r5(x):
    """Round for compact json metrics (figures don't need more)."""
    return float(f"{x:.5g}")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def smooth(xs, k):
    """Centered moving average, same length (for figure scripts)."""
    xs = np.asarray(xs, dtype=float)
    if k <= 1:
        return xs
    kernel = np.ones(k) / k
    return np.convolve(xs, kernel, mode="same")
