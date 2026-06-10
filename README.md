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

The final layer is **bias-free**: z = Wh. With a bias, scaling W alone gives
αWh + b, which is not a uniform scaling of z — the exact generator of z → αz is
joint (W, b) scaling. Dropping the bias makes the decomposition exact, matches
Soudry et al.’s setting (needed for the special-case argument), and follows the
bias-free final layers of arXiv:2502.02103. One sensitivity-appendix run keeps the
bias and projects (W, b) jointly, to show nothing qualitative changes.

The **radial direction** at W is the projection of the gradient onto a scale
subspace: movement that rescales outputs. The **tangential component** is the
orthogonal complement: movement that changes output direction/structure. Two scale
subspaces, both defined in theory.md and both logged:

- **ρ_global** — projection onto the one-dimensional global scale direction
  (W itself, whole-matrix inner product). Provably structure-preserving: argmax
  and all relative logits are invariant. Canonical for the thesis and abstract.
- **ρ_row** — projection onto the per-row radial subspace. Each row is a mixture
  component, so per-row scale is the per-component volume degree of freedom of
  the implicit EM framing. Not structure-preserving in general (unequal row
  scaling can change relative logits).

The global scale direction lies inside the per-row subspace, so ρ_row ≥ ρ_global
(state the nesting in theory.md). The gap ρ_row − ρ_global measures per-component
volume drift beyond global temperature — exactly what the implicit EM framing
predicts should exist, and what InfoMax-style fixes do and don’t control.

Two measurement levels, both implemented:

1. **Weight-level** (exact, used where the final layer is linear): decompose
   ∇_W L into radial fraction ρ = ‖proj ∇‖² / ‖∇‖² per step, for both subspaces.
1. **Logit-level** (architecture-agnostic, needed because deep homogeneous nets can
   hide scale in any layer): per-example radial fraction of the logit displacement,
   (ẑ·dz)² / ‖dz‖², averaged over examples per step (mean-per-step is the stabler
   estimator). Also track ‖z‖ growth, responsibility entropy (mean per-example
   softmax entropy on the train set, at eval checkpoints), and margin structure.
   Use this for ResNet experiments.

## Experiments

Each experiment is a standalone script in `src/` with shared utilities in
`src/common.py`. Each produces a lab report in `reports/` and saves metrics as
json/csv next to the report. Figures regenerate from saved metrics, never require
retraining. Pin seeds; ≥3 seeds per configuration; report mean ± std.

### E1 — The decomposition (thesis figure)

`src/e1_decomposition.py`. MNIST, small MLP + bias-free linear head, standard CE,
SGD. Log per step: ρ_global and ρ_row, ‖W‖ per row, normalized direction drift
‖Ŵ(t) − Ŵ(t−k)‖, train loss, train error, probe/test accuracy; responsibility
entropy at eval checkpoints.
**Predictions** (scoped to the post-separation regime): pre-separation, scaling has
a finite optimum and ρ stays bounded away from 1; once train error ≈ 0, the radial
direction becomes a pure descent direction and ρ_global inflects upward toward 1.
The inflection aligned with zero train error is registered as its own numbered
prediction. ‖W‖ grows without bound; direction drift decays toward zero,
consistent with Soudry et al.’s O(1/log t) rate — plot drift on a log-t axis so
the rate comparison is visible. All late loss reduction is radial.
Thesis figure, two panels: ρ_global(t) and ρ_row(t); ρ_global(t) overlaid with
train error(t), showing the aligned inflection.
This plot is the paper. If it fails, stop and rethink before E2–E4.

### E2 — The intervention (causal proof)

`src/e2_intervention.py`. Same setup, two separate intervention arms mirroring the
ρ_row/ρ_global distinction: (1) rows of W constrained to unit norm, (2) fixed
global temperature. Both enforced by post-step projection, not weight-norm
reparametrization — the point is to remove the radial DOF from the iterate without
changing the gradient geometry; state this distinction explicitly in the E2 report
(optimization-side reviewers will ask). Comparing the arms is itself informative:
if fixed temperature alone closes most of the Adam–SGD gap, juicing is mostly
global; if unit-norm is needed, per-row drift matters.
Second testbed: the 2601 decoder-free SAE (LSE + InfoMax, unsupervised). Its code
lives in a separate repo — resolve the repo name when E2 starts; the primary
CE-classifier testbed doesn’t depend on it, so E2 starts without it.
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
Log: integrated radial motion — headline metric is the integrated mean-per-step
radial fraction of dz (the logit-level ρ); ∫ d‖z‖ kept as a secondary curve, since
it is the LogitNorm literature’s own measurement — plus final ECE, test accuracy.
**Prediction**: a single scatter — calibration error vs integrated radial motion —
with a tight correlation, fix-agnostic. Five “unrelated” heuristics are one
mechanism: approximate volume control.
Execution contract (runs on local GPU; authored and smoke-tested on CPU first):
entry point `python src/e4_fixes.py --config configs/e4/<arm>.yaml --seed N
--out results/e4/`; resumable; metrics flushed per epoch (a killed run loses
≤ 1 epoch); never writes figures — figure scripts consume `results/e4/*.json`
only; `--smoke` runs 2 epochs on a data subset for pipeline verification; prints
estimated wall-clock per arm at startup so the 18 runs can be scheduled.

## Order of operations

1. `notes/theory.md` — radial/tangential decomposition defined precisely,
   formalizing the settled decisions: bias-free final layer, ρ_global and ρ_row
   both defined with the nesting ρ_row ≥ ρ_global, predictions scoped to the
   post-separation regime; predictions numbered.
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
- Training runs execute on local GPU hardware (3080 Ti); scripts are authored and
  smoke-tested on CPU first, and metrics (json/csv) are committed back to the repo.
- Lab reports record what actually happened, including negative results and
  surprises, before any narrative shaping in the draft.
- Writing style for notes/draft: declarative, short sentences, predictions stated
  before evidence, limitations stated plainly. Match the register of
  arXiv:2601.06478. No em-dashes in LaTeX items; “EM” not “expectation–maximization”
  after first use.

## Known risks

- **Novelty pressure from Soudry et al.**: E1 alone reproduces (empirically) what
  they proved theoretically for linear models. Frame this as confirmation, not
  liability: E1 reproduces their rate in the linear case (drift on a log-t axis),
  then E2–E4 go where the theorem doesn’t — the causal intervention, the optimizer
  asymmetry, and the unification of fixes. Never claim directions “freeze”
  anywhere in the paper.
- **Deep-network scale leakage**: homogeneity lets scale hide in any layer, so
  weight-level ρ on the last layer undercounts juicing in ResNets. The logit-level
  measurements are the defense; be explicit about this scoping in the draft.
- **E4 correlation may be loose**: the five fixes differ in side effects (label
  smoothing changes targets, focal loss reweights examples). If the scatter is
  noisy, the fallback claim is weaker but still publishable: radial suppression is
  necessary but not sufficient for calibration. Decide framing after data exists.