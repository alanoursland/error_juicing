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

## Post-registration additions (each registered before its experiment ran)

**P11 (channel suppression; registered 2026-06-10 after E1–E3, before any
non-smoke E4 run).** E3 found two norm-growth channels: directed radial motion
(SGD-like advection) and tangential diffusion (Adam-like; ‖W‖² grows by
Pythagoras). For E4 (all arms SGD+momentum):
(a) the norm-bounding fixes — LogitNorm and weight decay — reduce integrated
radial motion R below baseline by the largest margins;
(b) the learned-then-frozen **temperature arm is the predicted outlier** in
the P10 scatter: it improves ECE while reducing R least among effective fixes,
because dividing logits by a learned T cancels the *effect* of scale on the
loss without suppressing ‖z‖ growth itself;
(c) consequently the P10 Spearman correlation computed without the temperature
arm exceeds the correlation with it.
*Falsified if* the temperature arm reduces R as much as LogitNorm/weight decay,
or sits on the same regression line as the other arms.

**P12 (SAE testbed mechanism; registered 2026-06-10 after E1–E3, before any
E2b/SAE run; repo for the 2601 testbed resolved to
`alanoursland/implicit_em_in_neural_activations`, `encoder_study/`).**
Analysis of the 2601 objective before running: under a → αa (generated by
jointly scaling the encoder's (W, b); ReLU is homogeneous), the variance term
−Σ_j log var(a_j) decreases as −2K·log α without bound, while the ReLU-sparse
LSE term saturates (inactive units pin min_j a_ij = 0). The objective
therefore contains a *persistent, logarithmically-paying* radial descent
direction — precisely the mechanism P9 posited and E3 refuted for CE
classifiers. Predictions for E2b (Encoder 784→64 ReLU, λ all 1.0, their
optimal lrs: SGD 0.01, Adam 1e-3, 100 epochs, seeds {1,2,3}):
(a) baseline reproduces the 2601 anomaly: Adam's final loss is substantially
lower (more negative) than SGD's, and Adam's ‖(W, b)‖ grows much larger;
(b) unlike the CE testbed, Adam's epoch-displacement radial fraction on
(W, b) is HIGH here (directed pursuit, not diffusion), because the radial
signal pays logarithmically forever — ρ_disp(Adam) is comparable to or
exceeds ρ_disp(SGD);
(c) projecting ‖(W, b)‖ to its init value post-step makes both optimizers
plateau at similar floors: the Adam−SGD final-loss gap shrinks below 25% of
its baseline value (P7's criterion, evaluated on its intended testbed);
(d) linear-probe accuracy on the features is unchanged by the constraint
(within seed noise) for both optimizers.
*Falsified if* the baseline gap fails to appear, the gap survives the
constraint, Adam's ρ_disp is at the CE-testbed noise level (diffusion-only),
or probe accuracy degrades materially under the constraint.

## Amendment log

(Untouched until experiments run. Any change to a prediction after its experiment
started is recorded here with a date and reason; the original text above is never
edited.)

**A2 (2026-06-10, after the first E2 grid, before its report).** The first E2
grid (baseline/row/global × SGD/Adam) showed that head-only constraints leak:
SGD plateaus at a structural floor (train loss 0.013 vs 0.0007 baseline) with
ECE improved ~65%, but Adam under either head constraint reaches *lower* train
loss than its own baseline (1e-4 vs 6.5e-3) — the homogeneous body
(Linear+ReLU) re-creates the scale DOF and Adam exploits it. This is the
README's "deep-network scale leakage" risk, already present with one hidden
layer, and it strengthens the E3 mechanism (Adam pursues scale wherever it
lives). A fourth arm `full` (head rows unit + body (W1, b1) jointly projected
to init norm) was added to close the leak; P6/P7 are evaluated on it as well
as on the original arms, with the head-only arms reported as the leakage
demonstration. Separately noted: the 2601 Adam-lower-loss anomaly does not
reproduce at baseline in this supervised CE testbed (SGD's final loss is the
lower one); P7's premise is specific to the SAE testbed, which is deferred.
The original P6/P7 text is unchanged above.

**A1 (2026-06-10, after the first E1 run, before analysis).** P2 and P4 were
written with the per-step minibatch ρ in mind. The first E1 run showed per-step
minibatch ρ_global ≈ 0.001–0.004 at zero train error — only a few times the
chance level 1/(K·d) ≈ 4e-4 — while ‖W‖_F grew steadily (3.9 → ~11). Diagnosis,
reached before further analysis: minibatch gradient noise is predominantly
tangential and unbiased (it cancels across steps), while the radial component is
small but consistently signed (it accumulates). The expected per-step loss change
is first-order in the **full-batch** gradient (E[ΔL] ≈ −η⟨∇L_full, E G_batch⟩ =
−η‖∇L_full‖²), so the radial share of expected loss reduction is ρ of the
full-batch gradient, not of the minibatch gradient. E1 and E3 therefore
additionally log, per epoch: (i) ρ of the closed-form full-train-set head
gradient, (ii) ρ of the epoch displacement W_end − W_start. P2 and P4 are
evaluated on estimator (i) (P4 also on (ii)); the per-step minibatch curves are
retained and reported as the noise-floor contrast. The per-step estimator's
failure to show the effect is itself a result (reported in E1's lab report), not
a falsification of P2/P4, because the registered claims concern where training
*reduces loss* and *moves the iterate*, which the aggregate estimators measure
and the noisy per-step estimator provably understates. Trajectories are
unaffected: the added logging consumes no randomness; identical seeds reproduce
the original runs exactly.
