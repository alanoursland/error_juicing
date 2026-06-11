# E4 lab report — fix taxonomy

Date: 2026-06-12. Written immediately after the 18 GPU runs landed
(`results/e4/*.json`, trained on the local 3080 Ti with the original
DataLoader pipeline; 160 epochs, ResNet-18 CIFAR stem, bias-free head, SGD
momentum 0.9, lr 0.1 with ×0.1 drops at epochs 80/120; seeds {0, 1, 2}).
Figure regenerates via `src/fig_e4.py`.

## Results (3-seed means)

| arm             | R (registered) | R (epochs ≥ 40) | final probe ‖z‖ | test ECE | test acc |
|-----------------|---------------:|----------------:|-----------------:|---------:|---------:|
| label_smoothing | 13.9           | 4.8             | 4.3              | 0.0691   | 0.9318   |
| focal           | 83.3           | 45.2            | 23.3             | 0.0296   | 0.9121   |
| weight_decay    | 154.6          | 88.7            | 10.3             | **0.0278** | **0.9509** |
| baseline        | 167.1          | 101.8           | 42.9             | 0.0540   | 0.9300   |
| temperature     | 543.7          | 261.6           | 95.7             | 0.0588   | 0.9351   |
| logitnorm       | 45825.1        | 1264.2          | 152.8            | 0.0808   | 0.9118   |

All arms reach train error ≤ 0.06% (fully interpolated).

## Results against registered predictions

**P10 (ECE correlates with integrated radial motion, Spearman ≥ 0.8) —
FALSIFIED.** Spearman(R, ECE) = +0.37 over arm means (+0.34 over individual
runs), far below the registered 0.8, and not driven by a single arm: the
relation is **U-shaped**. The two worst-calibrated arms sit at the two ends of
the radial-motion axis — label smoothing at the bottom (ECE 0.069,
*under*-confident: its bounded targets pin ‖z‖ at 4.3) and logitnorm at the
top (ECE 0.081, overconfident raw logits at ‖z‖ ≈ 153). The well-calibrated
arms (weight decay 0.028, focal 0.030) sit in the middle. The registered
fallback ("radial suppression is necessary but not sufficient") is also the
wrong shape — suppression can itself miscalibrate. The right claim, post-hoc
and clearly labeled as such: **calibration error tracks the distance between
the deployed confidence scale and the calibrated scale, in either direction.**
This is E2's "dial, not a switch" result (full-arm under-confidence),
reproduced in the calibration community's own testbed.

**P11a (LogitNorm and weight decay reduce R most) — HALF-FALSIFIED.** Weight
decay behaves as predicted at the weight level (‖z‖ 10.3 vs baseline 42.9, R
modestly below baseline). LogitNorm does the opposite of the prediction on raw
logits: its loss is scale-invariant in z, so it prices the *loss-side* scale
while leaving raw ‖z‖ entirely unpriced — and with wd = 0 (our
one-regularizer-at-a-time design), nothing else prices it either. Raw ‖z‖
explodes to ~1750 in the first epochs, then drifts to ~153. Wei et al.'s
small-norm observations come from recipes that include weight decay; LogitNorm
*alone* does not control the deployed scale. This interaction is a finding,
not a bug.

**P11b (temperature arm improves ECE while reducing R least) — PENDING
correct evaluation; R half holds.** The temperature arm has the largest R
among non-logitnorm arms (it lets ‖z‖ grow to 95.7 and cancels the effect
inside the loss with T — exactly the mechanism P11 predicted). But its ECE
verdict is currently mis-measured: the harness computes ECE on raw z, while
the arm's deployed model includes the learned T (a model parameter). Same
caveat applies to logitnorm if one takes z/‖z‖/τ as its deployment. Resolution
requires the checkpoints, which live on the training machine:
**run `python src/e4_recompute_ece.py` there and commit
`results/e4/ece_corrected.json`** — it computes deployment-corrected ECE,
post-hoc optimal temperature T*, and the scale gap |log T*| per run.

**P11c (correlation improves without the temperature arm) — FALSIFIED.**
Ex-temperature Spearman is 0.30, lower than with it (0.37). Consistent with
the U-shape: removing one arm doesn't linearize a non-monotone relation.

## Surprises (recorded before narrative shaping)

1. **The U-shape.** Miscalibration is two-sided in radial motion. The
   monotone-correlation framing of P10 inherited the calibration literature's
   overconfidence bias; under-confidence from over-suppression (label
   smoothing here, E2's full arm before it) is equally real. The draft's
   outward claim becomes sharper, not weaker: five heuristics are one
   mechanism — they move the deployed confidence scale — and they calibrate
   exactly insofar as they move it *toward* the calibrated value.
2. **LogitNorm relocates rather than removes the scale DOF** (loss-side
   priced, deployment-side free), and depends on weight decay to pin the
   deployed scale. The clean mechanistic statement: scale must be priced in
   the *deployed function*, not merely in the training objective.
3. **Weight decay is the best all-round volume control** (best ECE, best
   accuracy, controlled ‖z‖) — consistent with E3's finding that its decay
   term is the one optimizer-side force pointing radially inward at a rate
   proportional to ‖W‖.
4. The logitnorm init transient (‖z‖ ≈ 1750 in epoch 0) dominates its
   registered R; rank order is unchanged after excluding epochs < 40, so no
   verdict depends on the transient, but R's sensitivity to transients is a
   metric-design lesson for the draft's methods section.

## Verdict for the paper

E4 falsifies its registered monotone prediction and replaces it with a
two-sided law that unifies more of the data: baseline and LogitNorm
miscalibrate by unpriced scale growth, label smoothing by scale suppression,
weight decay and focal calibrate by landing near the calibrated scale, and
the temperature arm is the diagnostic case whose verdict needs the
deployment-corrected ECE (recompute script ready). Pending that one script
run, the experimental program (E1, E2, E2b, E3, E4) is complete and the
draft can start.
