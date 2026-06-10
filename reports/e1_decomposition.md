# E1 lab report — the decomposition

Date: 2026-06-10. Written immediately after the runs, before any drafting.

## Setup

- MNIST, MLP 784→256 (ReLU, biased) → bias-free linear head 256→10
  (theory.md §1). Standard CE, plain SGD lr 0.1, batch 128, 150 epochs,
  seeds {0, 1, 2}. Sensitivity arm: seed 0 with biased head and joint (W, b)
  projection. CPU; metrics in `results/e1/e1_seed{0,1,2}.json` and
  `e1_seed0_bias.json`; figures regenerate via `src/fig_e1.py`.
- Logged per step: ρ_global, ρ_row of the minibatch gradient; ‖W‖_F; batch loss.
  Per epoch: ρ of the closed-form full-train-set gradient; ρ of the epoch
  displacement W_end − W_start (both added by amendment A1, predictions.md);
  train loss/error, test accuracy, responsibility entropy. Per ~epoch:
  normalized direction drift, per-row norms.
- Train error hits exactly 0 at epoch 31/34/37 (seed 0/1/2); < 0.1% at ~16.

## Results against registered predictions

**P1 (norm growth) — PASS.** ‖W‖_F grows 1.8 → 11.6 with no plateau; relative
growth over the final quarter is 3.0% per seed (threshold: < 1% would falsify).
Test accuracy is flat (98.2–98.3%) and responsibility entropy collapses
0.26 → 0.0022 nats over the same span: the loss buys confidence, not structure.

**P2 (post-separation inflection) — FAIL as registered.** ρ stays bounded away
from 1 pre-separation (that half holds), but there is **no inflection aligned
with zero train error**. The displacement estimator rises smoothly from epoch 0
(0.31 → 0.68 by epoch 10 → 0.85), beginning long before separation; the
full-batch estimator rises late (≈ epoch 50+) and noisily, also not aligned
with the separation epoch (~34). Diagnosis in "Surprises" below.

**P3 (drift decay, Soudry-consistent) — PASS, stronger than registered.**
Windowed drift ‖Ŵ(t) − Ŵ(t−k)‖ decays 0.55 → 0.0005. On log-log axes the late
half is a clean power law with slope −0.98/−0.98/−1.05 (seeds 0/1/2). This is
quantitatively the Soudry rate: if Ŵ(t) converges like 1/log t, the k-step
windowed difference scales as k/(t log²t), i.e. slope −1 up to log corrections.
The instrument reproduces the theorem's rate in the (near-)linear regime.

**P4 (late loss reduction is radial) — SPLIT VERDICT, report both.**
Mean over the final quarter of training, by estimator (3-seed means):

| estimator                          | mean ρ_global, last 25% | verdict vs 0.5 |
|------------------------------------|------------------------|----------------|
| epoch displacement (A1-ii)         | **0.846**              | pass           |
| full-batch gradient (A1-i)         | 0.321                  | fail           |
| per-step minibatch (as registered) | 0.0016                 | fail           |

Parameter *motion* is overwhelmingly radial (85%). The *instantaneous* loss
reduction is only ~1/3 radial at epoch 150: direction convergence is
logarithmic, so at any finite time the full-batch gradient retains a
substantial tangential part — which then largely cancels within the epoch
(that is why ρ_disp ≫ ρ_full). The registered 0.5 threshold is met only by the
displacement estimator. Stated plainly: at 150 epochs, training *moves* almost
purely radially, but the loss is still being reduced partly by residual
tangential refinement.

**P5 (nesting and gap) — PASS.** ρ_row ≥ ρ_global at every one of ~210k logged
comparisons (no violations; implementation check passes). The gap is nonzero
but small: 0.006 (displacement), 0.031 (full-batch) on average. Per-component
volume drift exists but global temperature is the dominant juicing channel —
directly relevant to E2's arm comparison (P8) and good news for the
"temperature is the missing DOF" framing.

**Sensitivity arm (biased head, joint projection).** Same picture: ρ_disp last
quarter 0.866, ‖W‖ → 11.6, per-step ρ_joint at the noise floor. Nothing
qualitative changes; the bias-free canonical setting is not load-bearing.

## Surprises (recorded before narrative shaping)

1. **The per-step minibatch ρ is at the noise floor (~3× chance = 1/2560) for
   the entire run**, even at zero train error. Minibatch tangential noise is
   unbiased and dominates each step; the radial component is small but
   consistently signed and accumulates. This forced amendment A1 and the
   estimator hierarchy. The hierarchy itself (noise floor → 0.32 → 0.85 as
   aggregation increases) is arguably the most instructive single result of E1:
   *where the optimizer wanders and where it travels are different questions.*
2. **No separation-aligned inflection.** The binary pre/post-separation framing
   in theory.md §2 is too coarse for MNIST: the model is at ~95% train accuracy
   after one epoch, so the aggregate scale derivative (Σ over examples of
   p·z − z_y) is already dominated by correctly-classified examples almost
   immediately — radial descent pays from the start. The inflection prediction
   implicitly assumed the correct-fraction crosses 50% well into training. The
   sharper object is the *weighted correct fraction*, not the zero-error time.
   Revision for theory.md and any restated prediction: tie radial dominance to
   the per-example weighted sign aggregate, and test the inflection on a task
   where separation happens late (deferred; candidate: label-noise MNIST or
   CIFAR without augmentation).
3. **ρ_full ≪ ρ_disp.** Successive full-batch gradients disagree tangentially
   (oscillation around the slowly-converging direction) and agree radially, so
   tangential motion cancels within an epoch. Neither estimator is "the" truth:
   ρ_full measures instantaneous loss-reduction allocation, ρ_disp measures
   realized travel. The draft should present both and say which claim each
   supports.
4. **The ρ_row − ρ_global gap is small** (juicing is ~global). The EM framing
   predicted per-component volume drift; it exists (gap > 0 everywhere) but is
   an order of magnitude smaller than the global channel on this task.

## Verdict for the paper

The thesis figure works, with ρ_disp as the headline curve, ρ_full and the
per-step noise floor as the supporting hierarchy, train error overlaid, and
drift-vs-log-t as the Soudry-rate confirmation panel
(`reports/fig_e1_thesis.png`, `fig_e1_estimators.png`). E1 supports the core
claim — post-separation training reduces loss mainly by structure-preserving
scale growth — with the honest qualification from P4's split verdict. Proceed
to E2.
