"""E4 -- fix taxonomy (outward-facing result).

CIFAR-10, ResNet-18 (CIFAR stem), CE loss, bias-free final layer. Six arms, one
regularizer at a time (configs/e4/*.yaml): baseline, weight_decay,
label_smoothing, logitnorm, focal, temperature (learned-then-frozen).

Data pipeline: the whole dataset is cached on the training device and
augmentation (RandomCrop(32, padding=4) + RandomHorizontalFlip) runs
vectorized on-device. No DataLoader, no worker processes -- fast everywhere
and immune to Windows multiprocessing issues.

Logged per epoch: train loss/err, test acc, test ECE; logit-level radial motion
on a fixed probe batch (theory.md S6): mean rho_logit, the radial-weighted
increment (whose running sum is the headline metric R), and d||z|| (secondary,
the LogitNorm literature's measurement).

Prediction addressed: P10, P11.

Execution contract (README): runs on local GPU; resumable (checkpoint per
epoch; a killed run loses <= 1 epoch); metrics flushed per epoch; never writes
figures; --smoke runs 2 epochs on a subset for CPU pipeline verification;
prints estimated wall-clock per arm at startup.

Usage:
  python src/e4_fixes.py --config configs/e4/baseline.yaml --seed 0 --out results/e4/
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from common import ece, logit_radial_step, r5, save_json, set_seed

# _DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_DATA_DIR = "E:/ml_datasets"

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def make_resnet18():
    """torchvision ResNet-18 with the standard CIFAR stem and bias-free head."""
    from torchvision.models import resnet18

    m = resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, 10, bias=False)  # theory.md S1
    return m


def cifar10_tensors(smoke, device):
    """Whole dataset as uint8 tensors on the training device."""
    import torchvision

    tr = torchvision.datasets.CIFAR10(_DATA_DIR, train=True, download=True)
    te = torchvision.datasets.CIFAR10(_DATA_DIR, train=False, download=True)
    xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()
    ytr = torch.tensor(tr.targets)
    xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
    yte = torch.tensor(te.targets)
    if smoke:
        xtr, ytr, xte, yte = xtr[:2000], ytr[:2000], xte[:1000], yte[:1000]
    return (xtr.to(device), ytr.to(device), xte.to(device), yte.to(device))


def augment(x01):
    """RandomCrop(32, padding=4) + RandomHorizontalFlip, vectorized on-device.

    Operates on [0,1] floats pre-normalization, so the zero padding matches
    torchvision's PIL-level black padding exactly.
    """
    B, dev = len(x01), x01.device
    pad = F.pad(x01, (4, 4, 4, 4))
    i = torch.randint(0, 9, (B,), device=dev).view(-1, 1, 1, 1)
    j = torch.randint(0, 9, (B,), device=dev).view(-1, 1, 1, 1)
    ar = torch.arange(32, device=dev)
    b = torch.arange(B, device=dev).view(-1, 1, 1, 1)
    c = torch.arange(3, device=dev).view(1, 3, 1, 1)
    x = pad[b, c, i + ar.view(1, 1, 32, 1), j + ar.view(1, 1, 1, 32)]
    flip = torch.rand(B, device=dev) < 0.5
    x[flip] = x[flip].flip(-1)
    return x


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, z, y):
        logp = F.log_softmax(z, dim=1).gather(1, y[:, None]).squeeze(1)
        return (-((1 - logp.exp()) ** self.gamma) * logp).mean()


def make_loss(cfg, temp_param):
    kind = cfg.get("loss", "ce")
    if kind == "ce":
        ls = cfg.get("label_smoothing", 0.0)
        return lambda z, y: F.cross_entropy(z, y, label_smoothing=ls)
    if kind == "logitnorm":
        tau = cfg["tau"]
        return lambda z, y: F.cross_entropy(
            z / (z.norm(dim=1, keepdim=True) + 1e-7) / tau, y)
    if kind == "focal":
        return FocalLoss(cfg["gamma"])
    if kind == "temperature":
        # learned-then-frozen temperature: logits / T with T = exp(log_t)
        return lambda z, y: F.cross_entropy(z / temp_param.exp(), y)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/e4/")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on a data subset; pipeline check on CPU")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    arm = cfg["arm"]
    epochs = 2 if args.smoke else cfg.get("epochs", 160)
    batch = cfg.get("batch", 128)
    dev = torch.device(args.device)
    mean, std = _MEAN.to(dev), _STD.to(dev)

    set_seed(args.seed)
    tag = f"e4_{arm}_seed{args.seed}" + ("_smoke" if args.smoke else "")
    out_json = f"{args.out.rstrip('/')}/{tag}.json"
    ckpt_path = f"{args.out.rstrip('/')}/{tag}.ckpt"

    xtr, ytr, xte, yte = cifar10_tensors(args.smoke, dev)
    n = len(xtr)
    steps_per_epoch = n // batch
    xte_n = ((xte.float() / 255) - mean) / std
    n_probe = 256 if args.smoke else 512
    probe_x = ((xtr[:n_probe].float() / 255) - mean) / std  # eval view, no aug

    def build():
        set_seed(args.seed)
        model = make_resnet18().to(dev)
        temp_param = nn.Parameter(torch.zeros((), device=dev))  # log T
        params = [{"params": model.parameters()}]
        if arm == "temperature":
            params.append({"params": [temp_param], "weight_decay": 0.0})
        opt = torch.optim.SGD(params, lr=cfg.get("lr", 0.1), momentum=0.9,
                              weight_decay=cfg.get("weight_decay", 0.0))
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1)
        return model, temp_param, opt, sched, make_loss(cfg, temp_param)

    model, temp_param, opt, sched, loss_fn = build()
    freeze_at = int(epochs * cfg.get("temp_freeze_frac", 0.5))

    m = {k: [] for k in [
        "ep_train_loss", "ep_train_err", "ep_test_acc", "ep_test_ece",
        "ep_rho_logit", "ep_R_increment", "ep_dznorm", "ep_z_norm", "ep_temp",
    ]}
    start_epoch = 0
    z_prev = None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        temp_param.data = ck["temp"]
        m = ck["metrics"]
        z_prev = ck["z_prev"]
        start_epoch = ck["epoch"] + 1
        print(f"resumed {tag} at epoch {start_epoch}", flush=True)

    # wall-clock estimate: time a few steps, then rebuild cleanly. Fresh
    # starts only: the timing steps perturb the model.
    if start_epoch == 0:
        model.train()
        xb = ((augment(xtr[:batch].float() / 255)) - mean) / std
        yb = ytr[:batch]
        for _ in range(2):  # warmup
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(5):
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        per_step = (time.time() - t) / 5
        est = per_step * steps_per_epoch * epochs * 1.1  # +eval overhead
        print(f"{tag}: ~{per_step*1000:.0f} ms/step, estimated wall-clock "
              f"{est/3600:.2f} h for {epochs} epochs", flush=True)
        model, temp_param, opt, sched, loss_fn = build()

    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        if arm == "temperature":
            temp_param.requires_grad_(epoch < freeze_at)
        model.train()
        perm = torch.randperm(n, device=dev)
        run_loss, run_err, count = 0.0, 0, 0
        for k in range(0, steps_per_epoch * batch, batch):
            idx = perm[k:k + batch]
            xb = (augment(xtr[idx].float() / 255) - mean) / std
            yb = ytr[idx]
            opt.zero_grad()
            z = model(xb)
            loss = loss_fn(z, yb)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach()) * len(yb)
            run_err += int((z.argmax(1) != yb).sum())
            count += len(yb)
        sched.step()

        model.eval()
        with torch.no_grad():
            z_probe = model(probe_x).cpu()
            zte = torch.cat([model(xte_n[k:k + 512]).cpu()
                             for k in range(0, len(xte_n), 512)])
        yte_c = yte.cpu()
        if z_prev is not None:
            rho_l, weighted, dzn = logit_radial_step(z_prev, z_probe)
            m["ep_rho_logit"].append(r5(rho_l))
            m["ep_R_increment"].append(r5(weighted))
            m["ep_dznorm"].append(r5(dzn))
        z_prev = z_probe
        m["ep_z_norm"].append(r5(float(z_probe.norm(dim=1).mean())))
        m["ep_train_loss"].append(r5(run_loss / count))
        m["ep_train_err"].append(r5(run_err / count))
        m["ep_test_acc"].append(r5(float((zte.argmax(1) == yte_c).float().mean())))
        m["ep_test_ece"].append(r5(ece(zte, yte_c)))
        m["ep_temp"].append(r5(float(temp_param.exp())))
        print(f"{tag} epoch {epoch:3d}  loss {m['ep_train_loss'][-1]:.4f}  "
              f"acc {m['ep_test_acc'][-1]:.4f}  ece {m['ep_test_ece'][-1]:.4f}  "
              f"|z| {m['ep_z_norm'][-1]:.1f}  ({time.time() - t0:.0f}s)",
              flush=True)

        # flush metrics + checkpoint every epoch (contract: lose <= 1 epoch)
        save_json(out_json, {"config": {**cfg, "seed": args.seed,
                                        "smoke": args.smoke}, "metrics": m})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "temp": temp_param.data,
                    "metrics": m, "z_prev": z_prev, "epoch": epoch}, ckpt_path)

    print(f"done {tag}  total {(time.time() - t0)/3600:.2f} h  "
          f"R = {sum(m['ep_R_increment']):.2f}")


if __name__ == "__main__":
    main()
