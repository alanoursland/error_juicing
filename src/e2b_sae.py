"""E2b -- the intervention on the 2601 decoder-free SAE (second testbed).

Reimplements the core of arXiv:2601.06478's encoder_study minimally and
faithfully (model, losses, and hyperparameters follow
alanoursland/implicit_em_in_neural_activations, encoder_study/src/{model,
losses}.py; raw [0,1] MNIST pixels, no normalization, matching their data.py).

Testbed: Encoder 784->64 (Linear WITH bias) + ReLU; loss = LSE + variance +
decorrelation, all lambda = 1.0. The 2601 anomaly: Adam keeps lowering the
loss for as long as it runs while SGD converges by ~epoch 70, at identical
probe accuracy.

Arms:
  baseline      unconstrained (should reproduce the anomaly, P12a)
  constrained   ||(W, b)|| jointly projected to its init value after every
                step. Joint (W, b) scaling is the exact generator of a -> alpha*a
                for the ReLU encoder (theory.md S1 logic, applied to the
                pre-activation scale DOF the variance term leaves open).

Predictions addressed: P7 (on its intended testbed), P12.

Usage:
  python src/e2b_sae.py --arm baseline --optimizer adam --seed 1 --out results/e2b/
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import r5, rho_global, save_json, set_seed

_DATA = None


def mnist_raw():
    """MNIST as raw [0,1] flattened pixels (matches encoder_study data.py)."""
    global _DATA
    if _DATA is None:
        import torchvision

        tr = torchvision.datasets.MNIST("data", train=True, download=True)
        te = torchvision.datasets.MNIST("data", train=False, download=True)
        _DATA = (tr.data.float().div(255.0).flatten(1), tr.targets.clone(),
                 te.data.float().div(255.0).flatten(1), te.targets.clone())
    return _DATA


def combined_loss(a):
    """LSE + variance + decorrelation, lambdas = 1.0 (encoder_study losses.py).

    Returns (total, lse, var, tc); LSE is the per-batch SUM, matching theirs.
    """
    lse = -torch.logsumexp(-a, dim=1).sum()
    var = -(a.var(dim=0) + 1e-6).log().sum()
    ac = a - a.mean(dim=0, keepdim=True)
    cov = (ac.T @ ac) / (a.shape[0] - 1)
    std = a.std(dim=0) + 1e-8
    corr = cov / (std[:, None] * std[None, :])
    off = corr - torch.eye(a.shape[1])
    tc = (off * (1 - torch.eye(a.shape[1]))).pow(2).sum()
    return lse + var + tc, lse, var, tc


def probe_accuracy(feats_tr, ytr, feats_te, yte, epochs=20):
    """Linear probe on frozen activations (feature-quality metric of 2601)."""
    probe = nn.Linear(feats_tr.shape[1], 10)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(epochs):
        for i in range(0, len(feats_tr), 1024):
            opt.zero_grad()
            F.cross_entropy(probe(feats_tr[i:i + 1024]), ytr[i:i + 1024]).backward()
            opt.step()
    with torch.no_grad():
        return float((probe(feats_te).argmax(1) == yte).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "constrained"], required=True)
    ap.add_argument("--optimizer", choices=["sgd", "adam"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 0.01 sgd / 1e-3 adam (2601 optima)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", default="results/e2b/")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    lr = args.lr if args.lr is not None else (0.01 if args.optimizer == "sgd" else 1e-3)

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    xtr, ytr, xte, yte = mnist_raw()
    n = len(xtr)

    enc = nn.Linear(784, args.hidden)  # with bias, matching encoder_study
    opt = (torch.optim.SGD(enc.parameters(), lr=lr) if args.optimizer == "sgd"
           else torch.optim.Adam(enc.parameters(), lr=lr))

    def wb():
        return torch.cat([enc.weight.detach().flatten(), enc.bias.detach()])

    norm0 = float(wb().norm())

    @torch.no_grad()
    def project():
        s = norm0 / float(wb().norm())
        enc.weight.mul_(s)
        enc.bias.mul_(s)

    m = {k: [] for k in [
        "ep_loss", "ep_lse", "ep_var", "ep_tc",      # mean per-batch components
        "ep_wb_norm", "ep_rho_disp_joint",           # scale DOF tracking
    ]}
    t0 = time.time()

    for epoch in range(args.epochs):
        wb_start = wb().clone()
        perm = torch.randperm(n)
        sums = torch.zeros(4)
        nb = 0
        for i in range(0, n - args.batch + 1, args.batch):
            x = xtr[perm[i:i + args.batch]]
            opt.zero_grad()
            a = F.relu(enc(x))
            total, lse, var, tc = combined_loss(a)
            total.backward()
            opt.step()
            if args.arm == "constrained":
                project()
            sums += torch.tensor([float(total.detach()), float(lse.detach()),
                                  float(var.detach()), float(tc.detach())])
            nb += 1
        for k, v in zip(["ep_loss", "ep_lse", "ep_var", "ep_tc"], sums / nb):
            m[k].append(r5(float(v)))
        d = (wb() - wb_start)[None, :]
        m["ep_rho_disp_joint"].append(r5(rho_global(d, wb_start[None, :])))
        m["ep_wb_norm"].append(r5(float(wb().norm())))
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"{args.arm}/{args.optimizer} seed {args.seed} ep {epoch:3d}  "
                  f"loss {m['ep_loss'][-1]:9.1f}  |Wb| {m['ep_wb_norm'][-1]:7.2f}  "
                  f"rho_disp {m['ep_rho_disp_joint'][-1]:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    with torch.no_grad():
        ftr = F.relu(enc(xtr))
        fte = F.relu(enc(xte))
    acc = probe_accuracy(ftr, ytr, fte, yte)
    print(f"probe accuracy: {acc:.4f}")

    tag = f"e2b_{args.arm}_{args.optimizer}_seed{args.seed}"
    save_json(f"{args.out.rstrip('/')}/{tag}.json",
              {"config": {**vars(args), "lr": lr}, "metrics": m,
               "probe_acc": acc})
    print(f"saved {tag}.json  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
