"""Recompute deployment-corrected calibration from E4 checkpoints.

Run this ON THE MACHINE THAT TRAINED E4 (checkpoints are not committed):

  python src/e4_recompute_ece.py

For every results/e4/e4_*_seed*.ckpt it computes, on the test set:
  - ece_raw:       ECE of raw logits z (what e4_fixes.py logged)
  - ece_deployed:  ECE of the arm's deployed output -- z / T_learned for the
                   temperature arm, z otherwise
  - T_star:        post-hoc temperature minimizing test NLL (Guo et al. 2017)
  - ece_at_Tstar:  ECE after post-hoc temperature scaling
  - log_Tstar:     the scale gap: how far the deployed confidence scale sits
                   from the calibrated one (0 = perfectly scaled)
  - z_norm:        mean test ||z||

Writes results/e4/ece_corrected.json (commit it; small). Figures/analysis can
then use the scale gap |log T_star| as the x-axis the U-shape suggests.
"""

import glob
import json
import os
import re

import torch
import torch.nn.functional as F

from common import ece
from e4_fixes import cifar10_tensors, make_resnet18, _MEAN, _STD


def nll_at_T(z, y, logT):
    return float(F.cross_entropy(z / logT.exp(), y))


def fit_temperature(z, y):
    """Post-hoc temperature scaling: minimize test NLL over log T."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp())


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xtr, ytr, xte, yte = cifar10_tensors(smoke=False, device=dev)
    xte_n = ((xte.float() / 255) - _MEAN.to(dev)) / _STD.to(dev)
    yte_c = yte.cpu()

    out = []
    for ck_path in sorted(glob.glob("results/e4/e4_*_seed*.ckpt")):
        name = os.path.basename(ck_path)
        if "smoke" in name:
            continue
        arm, seed = re.match(r"e4_(.+)_seed(\d+)\.ckpt", name).groups()
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model = make_resnet18().to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        with torch.no_grad():
            z = torch.cat([model(xte_n[k:k + 512]).cpu()
                           for k in range(0, len(xte_n), 512)])
        T_learned = float(ck["temp"].exp())
        z_dep = z / T_learned if arm == "temperature" else z
        T_star = fit_temperature(z, yte_c)
        rec = {
            "arm": arm, "seed": int(seed), "epoch": ck["epoch"],
            "acc": float((z.argmax(1) == yte_c).float().mean()),
            "ece_raw": ece(z, yte_c),
            "T_learned": T_learned,
            "ece_deployed": ece(z_dep, yte_c),
            "T_star": T_star,
            "ece_at_Tstar": ece(z / T_star, yte_c),
            "log_Tstar": float(torch.log(torch.tensor(T_star))),
            "z_norm": float(z.norm(dim=1).mean()),
        }
        out.append(rec)
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in rec.items()}, flush=True)

    with open("results/e4/ece_corrected.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote results/e4/ece_corrected.json ({len(out)} runs)")


if __name__ == "__main__":
    main()
