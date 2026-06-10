"""E2 -- the intervention (causal proof).

Same setup as E1. Two intervention arms mirroring the rho_row/rho_global
distinction (README), plus baseline, crossed with {SGD, Adam}:

  baseline   no constraint
  row        rows of W projected to unit norm after every step
  global     ||W||_F projected to sqrt(K) after every step (fixed global
             temperature; sqrt(K) matches the row arm's total Frobenius
             budget so the two arms differ only in per-row freedom)
  full       row constraint on the head PLUS the body layer's (W1, b1)
             jointly projected to its init norm. Added after the first grid
             (amendment A2): head-only constraints leak -- the homogeneous
             body re-creates the scale DOF, and Adam exploits it to reach
             lower-than-baseline loss. This arm closes the leak.

Both are post-step projections, NOT weight-norm reparametrization: the radial
DOF is removed from the iterate without changing the gradient geometry
(theory.md / README E2 -- state this in the report; optimization reviewers ask).

Predictions addressed: P6, P7, P8.

The second testbed (2601 decoder-free SAE) lives in a separate repo and is
deliberately absent here; this script is the primary CE-classifier testbed.

Usage:
  python src/e2_intervention.py --arm row --optimizer adam --seed 0 --out results/e2/
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F

from common import (
    MLP,
    ece,
    mnist_tensors,
    project_global_norm,
    project_rows_unit,
    r5,
    responsibility_entropy,
    rho_global,
    rho_row,
    save_json,
    set_seed,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "row", "global", "full"],
                    required=True)
    ap.add_argument("--optimizer", choices=["sgd", "adam"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 0.1 for sgd, 1e-3 for adam")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", default="results/e2/")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    lr = args.lr if args.lr is not None else (0.1 if args.optimizer == "sgd" else 1e-3)

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    xtr, ytr = mnist_tensors(train=True)
    xte, yte = mnist_tensors(train=False)
    n = len(xtr)

    model = MLP(hidden=args.hidden)
    opt = (torch.optim.SGD(model.parameters(), lr=lr) if args.optimizer == "sgd"
           else torch.optim.Adam(model.parameters(), lr=lr))
    W = model.head.weight
    K = W.shape[0]

    W1 = model.body[1].weight
    b1 = model.body[1].bias

    def body_norm():
        return float((W1.detach().norm() ** 2 + b1.detach().norm() ** 2) ** 0.5)

    body_norm0 = body_norm()

    @torch.no_grad()
    def project_body():
        s = body_norm0 / body_norm()
        W1.mul_(s)
        b1.mul_(s)

    def apply_constraint():
        if args.arm in ("row", "full"):
            project_rows_unit(W.data)
        elif args.arm == "global":
            project_global_norm(W.data, math.sqrt(K))
        if args.arm == "full":
            project_body()

    # apply the constraint at init too, so the iterate starts on the manifold
    apply_constraint()

    m = {k: [] for k in [
        "rho_global", "rho_row", "w_norm", "loss",                       # per step
        "ep_train_loss", "ep_train_err", "ep_test_acc", "ep_test_ece",
        "ep_entropy",
    ]}
    t0 = time.time()

    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            loss.backward()
            m["rho_global"].append(r5(rho_global(W.grad, W)))
            m["rho_row"].append(r5(rho_row(W.grad, W)))
            m["w_norm"].append(r5(float(W.detach().norm())))
            m["loss"].append(r5(float(loss)))
            opt.step()
            apply_constraint()

        model.eval()
        with torch.no_grad():
            ztr = torch.cat([model(xtr[j:j + 4096]) for j in range(0, n, 4096)])
            zte = torch.cat([model(xte[j:j + 4096]) for j in range(0, len(xte), 4096)])
        m["ep_train_loss"].append(r5(float(F.cross_entropy(ztr, ytr))))
        m["ep_train_err"].append(r5(float((ztr.argmax(1) != ytr).float().mean())))
        m["ep_test_acc"].append(r5(float((zte.argmax(1) == yte).float().mean())))
        m["ep_test_ece"].append(r5(ece(zte, yte)))
        m["ep_entropy"].append(r5(responsibility_entropy(ztr)))
        print(f"{args.arm}/{args.optimizer} seed {args.seed} epoch {epoch:3d}  "
              f"loss {m['ep_train_loss'][-1]:.4f}  acc {m['ep_test_acc'][-1]:.4f}  "
              f"ece {m['ep_test_ece'][-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    tag = f"e2_{args.arm}_{args.optimizer}_seed{args.seed}"
    save_json(f"{args.out.rstrip('/')}/{tag}.json",
              {"config": {**vars(args), "lr": lr}, "metrics": m})
    print(f"saved {tag}.json  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
