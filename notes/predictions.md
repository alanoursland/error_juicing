# Registered predictions

Registered before any experiment was run, mirroring the methodology of
arXiv:2601.06478. Each prediction is numbered, maps to one experiment, names its
metric, and states what would falsify it. Lab reports must address every prediction
for their experiment, including the failed ones.

Definitions are from `notes/theory.md`: ρ_global / ρ_row (§3), ρ_step (§4),
logit-level radial motion R (§6). "Post-separation" means train error ≈ 0
(threshold: < 0.1%).

---

## E1 — the decomposition

**P1 (norm growth).** ‖W‖_F of the final layer grows without bound over training;
no plateau within the run. Consistent with log-t growth post-separation.
*Metric*: ‖W‖_F vs step. *Falsified if* ‖W‖_F plateaus (relative growth < 1% over
the last quarter of training) while train loss still decreases.

**P2 (post-separation inflection — the thesis prediction).** Pre-separation,
ρ_global stays bounded away from 1; once train error reaches ≈ 0, ρ_global inflects
upward and trends toward 1. The inflection is temporally aligned with the step
where train error reaches ≈ 0.
*Metric*: ρ_global(t) overlaid with train error(t); inflection located by the
maximum of the smoothed derivative of ρ_global.
*Falsified if* ρ_global shows no upward trend post-separation, or the inflection
occurs far from the zero-train-error step (more than ~20% of total training steps
away), or ρ_global is already ≈ 1 pre-separation.

**P3 (drift decay, Soudry-consistent).** Normalized direction drift
‖Ŵ(t) − Ŵ(t−k)‖_F decays toward zero late in training, consistent with
O(1/log t) direction convergence; meanwhile ‖W‖ keeps growing (P1). Plotted on a
log-t axis. Never phrased as "freeze".
*Metric*: drift vs log t. *Falsified if* drift stays flat or increases late in
training while train error is 0.

**P4 (late loss reduction is radial).** Post-separation, the radial share of the
per-step loss reduction approaches 1. For SGD the per-step loss change is
≈ −η‖G‖², splitting as ρ·(radial) + (1−ρ)·(tangential); so the metric is ρ_global
itself, time-averaged over the last quarter of training.
*Metric*: mean ρ_global over the final 25% of steps. *Falsified if* this mean is
below 0.5 — i.e. most late loss reduction would be structural, contradicting the
thesis.

**P5 (nesting and gap).** ρ_row ≥ ρ_global at every logged step (this is a theorem;
its violation indicates an implementation bug, and it is registered so the check is
mandatory). The gap ρ_row − ρ_global is nonzero: per-component volume drift exists
beyond global temperature.
*Metric*: both curves. *Falsified if* the gap is indistinguishable from 0 throughout
(would mean per-component volume adds nothing over global temperature — informative
for the EM framing, not fatal to the thesis).

## E2 — the intervention

**P6 (structural floor).** Removing the radial DOF (unit-norm rows, and separately
fixed global temperature) makes train loss plateau at a structural floor instead of
descending indefinitely; test accuracy is unchanged relative to baseline (within
seed noise); calibration improves (ECE decreases substantially, ≥ 30% relative).
*Falsified if* constrained runs lose accuracy (> 0.5% absolute), or ECE does not
improve, or loss keeps descending at the baseline rate.

**P7 (Adam–SGD gap closes).** The final-loss gap between Adam and SGD (the 2601
anomaly: Adam reaches ~50% lower loss at identical feature quality) shrinks to
near zero under the radial constraint.
*Metric*: (loss_SGD − loss_Adam) baseline vs constrained. *Falsified if* the gap
persists (> 25% of its baseline value) under unit-norm constraint — would mean the
anomaly is not (only) scale.

**P8 (global vs per-row attribution).** Comparing the two arms: if fixed global
temperature alone closes most of the Adam–SGD gap, juicing is predominantly global;
if unit-norm rows are required, per-component drift matters. Registered as a
directional question with no predicted winner — the answer feeds the EM framing
either way. (Exploratory; cannot be falsified, only answered.)

## E3 — optimizer asymmetry

**P9 (Adam pursues the radial direction).** Adam (and AdamW) allocates a
substantially larger fraction of its realized step to the radial subspace than SGD
at matched training stage: time-averaged ρ_step(Adam) > ρ_step(SGD) post-separation,
by at least a factor of 1.5, robust across the lr grid.
*Falsified if* ρ_step is comparable between optimizers, or SGD's is larger — either
would break the proposed mechanism (radial gradient is small-but-persistent; Adam
normalizes magnitude away and follows it).

## E4 — fix taxonomy

**P10 (one mechanism).** Across the six arms (baseline, weight decay, label
smoothing, LogitNorm, focal loss, learned-then-frozen temperature), final ECE
correlates positively with integrated radial motion R (theory.md §6), fix-agnostic:
Spearman rank correlation ≥ 0.8 across arms (averaged over seeds).
*Falsified if* the correlation is weak (< 0.5) or driven by a single arm. Fallback
claim, decided after data exists: radial suppression is necessary but not
sufficient for calibration.

---

## Amendment log

(Untouched until experiments run. Any change to a prediction after its experiment
started is recorded here with a date and reason; the original text above is never
edited.)
