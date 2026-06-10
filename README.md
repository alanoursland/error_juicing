# LSE Volume Control: Error Juicing in Log-Sum-Exp Objectives

Research project for paper #6 in the implicit EM series. Investigates a structural
pathology of log-sum-exp losses (including standard CrossEntropyLoss): the loss can
be reduced indefinitely by uniformly scaling outputs — sharpening softmax confidence —
without any change to learned structure. We call this **error juicing**.

## Thesis

For any LSE-family objective, the gradient with respect to each component output equals
its softmax responsibility (Oursland 2025, arXiv:2512.24780). Responsibilities depend
only on *relative* distances, and the argmin/argmax assignment is invariant under
uniform scaling of all outputs. The loss is not. Therefore every LSE objective contains
a descent direction — pure output scale — that reduces loss while leaving assignments,
features, and decision structure untouched.

In the implicit EM framing: LSE is a mixture marginal likelihood missing its
log-determinant volume term. Error juicing is the collapse pathology of unconstrained
implicit EM, expressed as norm growth. The phenomenon has been observed independently
in four literatures that do not cite each other:

- **Calibration**: cross-entropy “inherently induces weight magnification”
  (Guo et al. 2017; Mukhoti et al. 2020)
- **OOD detection**: logit norm grows throughout training, causing overconfidence;
  LogitNorm fixes it by constraining the norm (Wei et al., ICML 2022)
- **Implicit bias theory**: on separable data, GD on logistic/CE loss diverges in
  weight norm while only the direction converges to max-margin (Soudry et al. 2018)
- **Metric learning**: face-recognition losses normalize features and weights for
  exactly this reason (NormFace, CosFace, ArcFace)

This paper unifies them: all four observe the same missing volume term. We derive the
pathology from the gradient–responsibility identity, decompose training dynamics into
structural (tangential) and juicing (radial) components, demonstrate the decomposition
empirically, and show that known fixes work in proportion to how much radial motion
they suppress.

Predecessor papers (same author, same framework):

- arXiv:2410.19352 — linear+Abs layers compute Mahalanobis distance
- arXiv:2411.17932 — perturbation evidence that networks use distance metrics
- arXiv:2502.02103 — distance vs intensity representations; OffsetL2
- arXiv:2512.24780 — gradient descent on LSE is implicit EM (∂L/∂dⱼ = −rⱼ)
- arXiv:2601.06478 — decoder-free SAE derived from the theory; LSE collapses without
  volume control; InfoMax (variance + decorrelation) as the neural log-determinant

This paper extends 2512/2601: InfoMax controls the activation *distribution* but not
the overall *scale* of the distance metric. Scale is the volume degree of freedom the
regularizers left open. It also explains an anomaly observed in 2601: Adam reaches
~50% lower loss than SGD with identical feature quality.

## Repository layout

```
notes/      Theory development and references. Source material for the paper.
draft/      Paper draft (md first, then tex).
src/        Experiment implementations.
reports/    Lab reports: one md per experiment — setup, results, interpretation,
            surprises. Written immediately after each run, before drafting.
```

## Core technical object: the radial/tangential decomposition

Defined precisely in `notes/theory.md` (write this before any code). Summary:

For final-layer weight matrix W producing logits z = Wh + b, the **radial direction**
at W is the projection of the gradient onto span{W} (per-row or whole-matrix —
settle in theory.md): movement that rescales outputs. The **tangential component**
is the orthogonal complement: movement that changes output direction/structure.

Two measurement levels, both implemented:

1. **Weight-level** (exact, used where the final layer is linear): decompose
   ∇_W L into radial fraction ρ = ‖proj_W ∇‖² / ‖∇‖² per step.
1. **Logit-level** (architecture-agnostic, needed because deep homogeneous nets can
   hide scale in any layer): track ‖z‖ growth, softmax/responsibility entropy, and
   margin structure directly. Use this for ResNet experiments.

Open design decision (resolve in theory.md before src/): per-row vs whole-matrix
radial subspace, and whether bias participates. Default position: per-row, bias
excluded, with a sensitivity check.

## Experiments

Each experiment is a standalone script in `src/` with shared utilities in
`src/common.py`. Each produces a lab report in `reports/` and saves metrics as
json/csv next to the report. Figures regenerate from saved metrics, never require
retraining. Pin seeds; ≥3 seeds per configuration; report mean ± std.

### E1 — The decomposition (thesis figure)

`src/e1_decomposition.py`. MNIST, small MLP + linear head, standard CE, SGD.
Log per step: radial fraction ρ, ‖W‖ per row, normalized direction drift
‖Ŵ(t) − Ŵ(t−k)‖, train loss, probe/test accuracy, responsibility entropy.
**Prediction**: directions converge early and freeze; ‖W‖ grows without bound;
ρ → 1 as training proceeds; all late loss reduction is radial.
This plot is the paper. If it fails, stop and rethink before E2–E4.

### E2 — The intervention (causal proof)

`src/e2_intervention.py`. Same setup, constrain rows of W to unit norm (project after
each step) and/or fix a global temperature. Also run on the 2601 decoder-free SAE
as a second testbed (LSE + InfoMax, unsupervised).
**Predictions**: loss plateaus at the structural floor; the SGD–Adam final-loss gap
vanishes; accuracy/probe quality unchanged; calibration (ECE) improves substantially.
If the Adam–SGD gap closes under unit-norm, the 2601 anomaly is fully explained as
scale, and the null space is characterized.

### E3 — Optimizer asymmetry (novel result)

`src/e3_optimizer.py`. SGD vs Adam (vs AdamW), same architecture, lr sweep.
Log radial fraction per optimizer over training.
**Prediction**: Adam allocates a far larger share of its step budget to the radial
direction. Mechanism: the radial gradient is small-but-persistent; SGD steps scale
with magnitude and ignore it; Adam normalizes magnitude away and pursues it
indefinitely. Nobody in the calibration/implicit-bias literatures has this plot.
This is the experiment most likely to produce surprises — record them in the report.

### E4 — Fix taxonomy (outward-facing result)

`src/e4_fixes.py`. CIFAR-10, ResNet-18 (the calibration community’s testbed), CE loss.
Five regularizers, one at a time: weight decay, label smoothing, LogitNorm, focal
loss, learned-then-frozen temperature. Plus unregularized baseline.
Log: integrated radial motion (logit-level: ∫ d‖z‖), final ECE, test accuracy.
**Prediction**: a single scatter — calibration error vs integrated radial motion —
with a tight correlation, fix-agnostic. Five “unrelated” heuristics are one
mechanism: approximate volume control.

## Order of operations

1. `notes/theory.md` — radial/tangential decomposition defined precisely; the
   per-row vs whole-matrix question settled; predictions numbered.
1. `notes/related_work.md` — the four literatures, with the position against each:
   what they observed vs what we derive. Soudry et al. needs the most care: their
   theorem (norm diverges, direction → max-margin, linear separable case) is a
   special case of our decomposition, and a reviewer will check.
1. `notes/predictions.md` — falsifiable predictions registered before experiments,
   mirroring the methodology of 2601. Each prediction maps to one experiment.
1. `src/common.py` + `src/e1_decomposition.py`, then run E1 same day.
1. E2, E3 (reuse harness, config changes only), reports as you go.
1. E4 (long-running; launch and work on draft in parallel).
1. `draft/` — thesis figure first, prose around it.

## Conventions

- Research code, not a library. Plain scripts, minimal abstraction, no packaging.
- Unit-test only the math with exact properties: the decomposition’s orthogonality,
  radial+tangential norms summing to gradient norm, scale-invariance of the
  tangential component under W → αW. Smoke-test training scripts. Nothing else.
- Every figure in the paper regenerates from committed metrics via a script.
- Lab reports record what actually happened, including negative results and
  surprises, before any narrative shaping in the draft.
- Writing style for notes/draft: declarative, short sentences, predictions stated
  before evidence, limitations stated plainly. Match the register of
  arXiv:2601.06478. No em-dashes in LaTeX items; “EM” not “expectation–maximization”
  after first use.

## Known risks

- **Novelty pressure from Soudry et al.**: E1 alone reproduces (empirically) what
  they proved theoretically for linear models. The paper’s weight rests on E2–E4:
  the causal intervention, the optimizer asymmetry, and the unification of fixes.
- **Deep-network scale leakage**: homogeneity lets scale hide in any layer, so
  weight-level ρ on the last layer undercounts juicing in ResNets. The logit-level
  measurements are the defense; be explicit about this scoping in the draft.
- **E4 correlation may be loose**: the five fixes differ in side effects (label
  smoothing changes targets, focal loss reweights examples). If the scatter is
  noisy, the fallback claim is weaker but still publishable: radial suppression is
  necessary but not sufficient for calibration. Decide framing after data exists.