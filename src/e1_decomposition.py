"""E1 -- the decomposition (thesis figure).

MNIST, small MLP + bias-free linear head, standard CE, plain SGD (so
rho_grad = rho_step, theory.md S4).

Logged per step: rho_global, rho_row, ||W||_F, train batch loss.
Logged every --drift-k steps: normalized direction drift ||What(t)-What(t-k)||_F,
per-row norms.
Logged per epoch (eval checkpoints): train loss/error on the full train set,
test accuracy, responsibility entropy on the train set (theory.md S6).

Predictions addressed: P1-P5 (notes/predictions.md).

Sensitivity arm (theory.md S1): --head-bias trains with a biased head and logs
the joint (W,b) rho_global alongside.

Usage:
  python src/e1_decomposition.py --seed 0 --epochs 80 --out results/e1/
"""

import argparse
import time

import torch
import torch.nn.functional as F

from common import (
    MLP,
    mnist_tensors,
    r5,
    responsibility_entropy,
    rho_global,
    rho_global_joint,
    rho_row,
    save_json,
    set_seed,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--drift-k", type=int, default=469)  # ~1 epoch of steps
    ap.add_argument("--out", default="results/e1/")
    ap.add_argument("--head-bias", action="store_true",
                    help="sensitivity arm: biased head + joint (W,b) projection")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    xtr, ytr = mnist_tensors(train=True)
    xte, yte = mnist_tensors(train=False)
    n = len(xtr)

    model = MLP(hidden=args.hidden, head_bias=args.head_bias)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    W = model.head.weight

    m = {k: [] for k in [
        "rho_global", "rho_row", "rho_joint", "w_norm", "loss",          # per step
        "drift_step", "drift", "row_norms",                              # per drift-k
        "ep_train_loss", "ep_train_err", "ep_test_acc", "ep_entropy",    # per epoch
        # per epoch, the aggregate estimators (amendment A1 in predictions.md):
        # rho of the full-train-set gradient (expected loss reduction is
        # first-order in this, not in the minibatch gradient), and rho of the
        # epoch displacement W_end - W_start (what training actually moved).
        "ep_rho_full_global", "ep_rho_full_row",
        "ep_rho_disp_global", "ep_rho_disp_row",
    ]}
    what_prev, step = None, 0
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

            G = W.grad
            m["rho_global"].append(r5(rho_global(G, W)))
            m["rho_row"].append(r5(rho_row(G, W)))
            if args.head_bias:
                m["rho_joint"].append(r5(rho_global_joint(
                    G, W, model.head.bias.grad, model.head.bias)))
            m["w_norm"].append(r5(float(W.norm())))
            m["loss"].append(r5(float(loss)))

            opt.step()

            if step % args.drift_k == 0:
                what = (W / W.norm()).detach().clone()
                if what_prev is not None:
                    m["drift_step"].append(step)
                    m["drift"].append(r5(float((what - what_prev).norm())))
                what_prev = what
                m["row_norms"].append([r5(float(x)) for x in W.norm(dim=1)])
            step += 1

        model.eval()
        # full-train-set head gradient in closed form: G = (P - Y)^T H / n
        G_full = torch.zeros_like(W.detach())
        ztr_chunks = []
        with torch.no_grad():
            for j in range(0, n, 4096):
                h = model.body(xtr[j:j + 4096])
                z = model.head(h)
                ztr_chunks.append(z)
                p = z.softmax(1)
                p[torch.arange(len(z)), ytr[j:j + 4096]] -= 1.0
                G_full += p.T @ h
            G_full /= n
            ztr = torch.cat(ztr_chunks)
            zte = torch.cat([model(xte[j:j + 4096]) for j in range(0, len(xte), 4096)])
        m["ep_rho_full_global"].append(r5(rho_global(G_full, W)))
        m["ep_rho_full_row"].append(r5(rho_row(G_full, W)))
        dW = W.detach() - W_start
        m["ep_rho_disp_global"].append(r5(rho_global(dW, W_start)))
        m["ep_rho_disp_row"].append(r5(rho_row(dW, W_start)))
        m["ep_train_loss"].append(r5(float(F.cross_entropy(ztr, ytr))))
        m["ep_train_err"].append(r5(float((ztr.argmax(1) != ytr).float().mean())))
        m["ep_test_acc"].append(r5(float((zte.argmax(1) == yte).float().mean())))
        m["ep_entropy"].append(r5(responsibility_entropy(ztr)))
        print(f"seed {args.seed} epoch {epoch:3d}  "
              f"train_err {m['ep_train_err'][-1]:.5f}  "
              f"rho_full_g {m['ep_rho_full_global'][-1]:.3f}  "
              f"rho_disp_g {m['ep_rho_disp_global'][-1]:.3f}  "
              f"|W| {m['w_norm'][-1]:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    tag = f"e1_seed{args.seed}" + ("_bias" if args.head_bias else "")
    save_json(f"{args.out.rstrip('/')}/{tag}.json",
              {"config": vars(args), "metrics": m})
    print(f"saved {tag}.json  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
