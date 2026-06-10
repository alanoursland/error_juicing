# E2 lab report — the intervention

Date: 2026-06-10. Written immediately after the runs, before drafting.

## Setup

- Same MNIST MLP as E1 (784→256→bias-free head). 80 epochs, seeds {0, 1, 2},
  SGD lr 0.1 and Adam lr 1e-3. Four arms:
  - **baseline** — unconstrained;
  - **global** — ‖W‖_F of the head projected to √10 after every step
    (fixed global temperature);
  - **row** — head rows projected to unit norm after every step;
  - **full** — row constraint on the head **plus** the body layer's (W1, b1)
    jointly projected to its init norm. Added by amendment A2 after the first
    grid exposed leakage (below).
- All constraints are post-step projections, not weight-norm
  reparametrization: the radial DOF is removed from the iterate without
  changing the gradient geometry. (Optimization-side reviewers: projection
  keeps the unconstrained gradient field and projects the *iterate*;
  reparametrization changes the gradient itself via the chain rule. The two
  have different dynamics; we test the former.)
- Metrics: `results/e2/*.json` (24 runs); figure regenerates via
  `src/fig_e2.py`. The second testbed (2601 decoder-free SAE) is deferred —
  separate repo, name to be resolved.

## Results (3-seed mean ± std)

| arm      | opt  | final train loss | test acc | test ECE |
|----------|------|------------------|----------|----------|
| baseline | sgd  | 0.0007 ± 0.0000  | 0.9824   | 0.0092   |
| baseline | adam | 0.0065 ± 0.0034  | 0.9806   | 0.0164   |
| global   | sgd  | 0.0129 ± 0.0003  | 0.9826   | **0.0033** |
| global   | adam | 0.0001 ± 0.0000  | 0.9838   | 0.0102   |
| row      | sgd  | 0.0133 ± 0.0004  | 0.9823   | **0.0032** |
| row      | adam | 0.0027 ± 0.0037  | 0.9833   | 0.0107   |
| full     | sgd  | 0.0519 ± 0.0009  | 0.9791   | 0.0165   |
| full     | adam | 0.0876 ± 0.0025  | 0.9707   | 0.0199   |

Constraint sanity: baseline head norm grows 3.2 → ~12; all constrained arms
hold ‖W‖_F = 3.162 exactly (figure panel D).

## Results against registered predictions

**P6 (structural floor / accuracy unchanged / ECE −30%) — PASS for SGD with
head constraints; the rest is a split that is more informative than a pass.**

- *SGD + row/global*: train loss plateaus at ~0.013, an 18× floor above
  baseline, while test accuracy is unchanged (Δ < 0.1%) and ECE improves
  **64–65%**. This is the registered prediction, confirmed cleanly.
- *Adam + row/global*: **no floor** — Adam reaches 0.0001–0.0027, lower than
  its own unconstrained baseline. See Surprise 1.
- *full*: a genuine floor for both optimizers (0.05–0.09), but accuracy drops
  0.3% (SGD) / 1.0% (Adam) and **ECE worsens** relative to baseline. See
  Surprise 2. P6's accuracy clause fails on full/adam; the ECE clause fails on
  both full arms.

**P7 (Adam–SGD gap closes) — NOT EVALUABLE AS REGISTERED on this testbed.**
The premise (Adam reaches lower loss than SGD at baseline — the 2601 anomaly)
does not reproduce in this supervised CE setting: baseline SGD's final loss is
~10× *lower* than Adam's (0.0007 vs 0.0065). The gap's sign then flips arm by
arm (+ under head constraints, − under full). The 2601 anomaly is apparently
specific to the SAE/LSE+InfoMax setting; evaluating P7 requires that testbed
(blocked on the separate repo). Recorded plainly rather than reinterpreted.

**P8 (global vs per-row attribution) — ANSWERED: juicing is global.** The row
and global arms are statistically indistinguishable in every metric (loss
floor 0.0129 vs 0.0133; ECE 0.0033 vs 0.0032; accuracy identical). Per-row
volume freedom beyond global temperature contributes nothing measurable on
this task — consistent with E1's small ρ_row − ρ_global gap (~0.006). For the
EM framing: the missing volume term is, empirically, one global scale, not K
per-component scales.

## Surprises (recorded before narrative shaping)

1. **Head-only constraints leak, and Adam finds the leak.** With the head
   frozen at fixed norm, the homogeneous body (Linear+ReLU) re-creates the
   scale DOF; Adam drives through it to *lower-than-baseline* loss while SGD
   barely uses it (SGD's persistence at the floor vs Adam's tunneling is
   visible in the loss curves, panel B — Adam's constrained curves oscillate
   violently around 1e-4). This is the README's "deep-network scale leakage"
   risk demonstrated with a single hidden layer, it directly previews E3's
   mechanism (Adam pursues the small-but-persistent scale gradient wherever it
   lives), and it means E2-style interventions must be stated as *whole-network*
   claims or they are not interventions at all. LogitNorm (output-level
   normalization) is immune to this leak by construction — worth one sentence
   in the draft.
2. **Closing the leak overshoots: the full constraint causes
   under-confidence.** With both layers norm-constrained, the model cannot
   reach the confidence its accuracy warrants; ECE *worsens* (0.0165/0.0199 vs
   0.0092 baseline) and accuracy drops — the body projection binds structural
   directions (feature learning needs norm growth in the body), not just
   scale. Volume control is therefore a *calibration dial, not a switch*: the
   best calibration in the study is head-only constraint + SGD (ECE 0.0033),
   where the radial DOF is suppressed at the output but features remain free.
   This refines E4's expectation: fixes should help in proportion to radial
   suppression *up to* the point where they bind structure.
3. **Baseline Adam is worse-calibrated than baseline SGD** (0.0164 vs 0.0092)
   at similar accuracy — consistent with Adam allocating more budget to
   confidence sharpening, which E3 measures directly.

## Verdict for the paper

The causal claim survives in its precise form: removing the scale DOF *of the
whole function* halts the loss descent (floor), and removing it at the output
improves calibration substantially at zero accuracy cost — but only for the
optimizer that wasn't actively hunting the leak. The leakage result upgrades a
README risk into a headline finding and hands E3 its motivation. P7 moves to
the SAE testbed. Proceed to E3.
