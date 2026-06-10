"""E3 -- optimizer asymmetry (novel result).

SGD vs Adam vs AdamW, same MLP architecture, lr grid. Logs the radial fraction
of the *realized step* delta-W (rho_step, theory.md S4) -- the claim is about
step budget, and for adaptive optimizers rho_step != rho_grad. rho_grad is
logged alongside for reference.

Prediction addressed: P9 (time-averaged post-separation rho_step(Adam) >
1.5 x rho_step(SGD), robust across the lr grid).

Usage:
  python src/e3_optimizer.py --optimizer adam --lr 1e-3 --seed 0 --out results/e3/
"""

import argparse
import time

import torch
import torch.nn.functional as F

from common import (
    MLP,
    mnist_tensors,
    r5,
    rho_global,
    rho_row,
    save_json,
    set_seed,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", default="results/e3/")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    xtr, ytr = mnist_tensors(train=True)
    xte, yte = mnist_tensors(train=False)
    n = len(xtr)

    model = MLP(hidden=args.hidden)
    opt = {
        "sgd": lambda p: torch.optim.SGD(p, lr=args.lr),
        "adam": lambda p: torch.optim.Adam(p, lr=args.lr),
        "adamw": lambda p: torch.optim.AdamW(p, lr=args.lr),  # default wd 0.01
    }[args.optimizer](model.parameters())
    W = model.head.weight

    m = {k: [] for k in [
        "rho_step_global", "rho_step_row", "rho_grad_global",            # per step
        "w_norm", "loss",
        "ep_train_err", "ep_test_acc",                                   # per epoch
        # aggregate estimator (amendment A1): rho of the epoch displacement.
        # Per-step minibatch rho is noise-dominated; the optimizer comparison
        # is about where the *accumulated* step budget goes.
        "ep_rho_disp_global", "ep_rho_disp_row",
    ]}
    t0 = time.time()

    for epoch in range(args.epochs):
        W_start = W.detach().clone()
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            loss.backward()
            W_before = W.detach().clone()
            m["rho_grad_global"].append(r5(rho_global(W.grad, W)))
            opt.step()
            dW = W.detach() - W_before
            # rho_step decomposes the realized update at the pre-step iterate
            m["rho_step_global"].append(r5(rho_global(dW, W_before)))
            m["rho_step_row"].append(r5(rho_row(dW, W_before)))
            m["w_norm"].append(r5(float(W.detach().norm())))
            m["loss"].append(r5(float(loss)))

        dWe = W.detach() - W_start
        m["ep_rho_disp_global"].append(r5(rho_global(dWe, W_start)))
        m["ep_rho_disp_row"].append(r5(rho_row(dWe, W_start)))
        model.eval()
        with torch.no_grad():
            ztr = torch.cat([model(xtr[j:j + 4096]) for j in range(0, n, 4096)])
            zte = torch.cat([model(xte[j:j + 4096]) for j in range(0, len(xte), 4096)])
        m["ep_train_err"].append(r5(float((ztr.argmax(1) != ytr).float().mean())))
        m["ep_test_acc"].append(r5(float((zte.argmax(1) == yte).float().mean())))
        print(f"{args.optimizer} lr={args.lr} seed {args.seed} epoch {epoch:3d}  "
              f"err {m['ep_train_err'][-1]:.5f}  "
              f"rho_disp_g {m['ep_rho_disp_global'][-1]:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    tag = f"e3_{args.optimizer}_lr{args.lr:g}_seed{args.seed}"
    save_json(f"{args.out.rstrip('/')}/{tag}.json",
              {"config": vars(args), "metrics": m})
    print(f"saved {tag}.json  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
