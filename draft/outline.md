# Paper outline — LSE Volume Control: Error Juicing in Log-Sum-Exp Objectives

Working outline for the draft. Register: declarative, short sentences,
predictions stated before evidence, limitations plain (match arXiv:2601.06478).
Every figure regenerates from committed metrics. Sources: notes/theory.md,
notes/predictions.md (P1–P12 + amendments A1–A2), reports/e{1,2,2b,3,4}*.md.

## Title and one-sentence claim

Title candidate: *Error Juicing: Every Log-Sum-Exp Objective Pays for
Confidence It Didn't Earn.*
(Alternative, more conservative: *Volume Control in Log-Sum-Exp Objectives:
A Radial/Tangential Decomposition of Training Dynamics.*)

One sentence: LSE-family losses contain a structure-preserving descent
direction — pure output scale — and a large, measurable share of late training
is spent on it; known fixes from four non-citing literatures work by pricing
that one degree of freedom, and they miscalibrate when they price it too much
as well as too little.

## 1. Introduction

- The pathology in one paragraph: loss falls, accuracy frozen, entropy
  collapsing. Confidence is being bought, not earned. We name it error juicing.
- The unification hook: four literatures observed it independently
  (calibration's weight magnification; LogitNorm's norm growth; implicit-bias
  norm divergence; metric learning's normalized faces). None cite the others.
  One missing volume term.
- Contributions list (each maps to one experiment + one figure):
  1. A decomposition that measures juicing (ρ_global canonical, exact for
     bias-free heads; estimator hierarchy) — E1.
  2. A causal intervention with a leakage result: scale must be priced in the
     whole function, not one layer — E2.
  3. An exact, closed-form instance: the 2601 SAE anomaly equals the
     −2K·log α volume payout, removed by constraint — E2b.
  4. Two transport mechanisms for the same pathology: advection (SGD) vs
     diffusion (Adam), selected by the objective's radial payout — E3.
  5. A two-sided law for fixes: calibration error tracks distance between
     deployed and calibrated confidence scale (U-shape, not monotone) — E4.
- Methodology note (short): predictions registered before runs
  (notes/predictions.md); the paper reports its falsifications (P2, P9, P10)
  and what replaced them. This is a feature of the series, keep it visible.

## 2. Theory (from notes/theory.md)

- 2.1 Setting: z = Wh, bias-free head; why (exact generator of z → αz);
  gradient–responsibility identity ∂L/∂z = p − y (cite 2512).
- 2.2 The master identity ⟨∇_W L, W⟩_F = ⟨∇_z L, z⟩ (Euler). Sign analysis:
  per-example scale derivative Σp_k z_k − z_y; correct-with-margin ⇒ negative.
  Pre/post-separation regimes. State plainly that the binary framing was
  refined by E1 (weighted majority, not zero-error time).
- 2.3 The decomposition: ρ_global (1-D, provably structure-preserving;
  canonical) and ρ_row (per-component volume, EM-facing); nesting
  ρ_row ≥ ρ_global; invariants table (what scale motion cannot change).
- 2.4 Estimators: per-step minibatch ρ is noise-floored (tangential noise is
  unbiased, radial signal is biased); full-batch gradient ρ = radial share of
  expected loss reduction; epoch-displacement ρ = radial share of realized
  travel. The hierarchy is a measurement contribution; A1 documents it.
- 2.5 Norm-growth channels: Δ‖W‖² = 2⟨W, ΔW⟩ + ‖ΔW‖² — advection + diffusion.
- 2.6 Implicit EM framing: LSE = mixture marginal missing its log-det; scale
  is the volume DOF InfoMax leaves open (sets up E2b). Keep short; the EM
  story is the series' thread, not this paper's main road.

## 3. E1 — the decomposition measured (thesis figure)

- Fig 1 = reports/fig_e1_thesis.png (two panels: estimator hierarchy with
  train error overlay; ‖W‖ growth + drift on log axes).
- Numbers to quote: ‖W‖ 1.8 → 11.6 with accuracy flat at 98.2% and entropy
  0.26 → 0.0022; ρ_disp last quarter 0.85; ρ_full 0.32; per-step 0.0016
  (≈ 4× chance). Drift log-log slope −1.00 ± 0.04 = the Soudry rate
  (windowed 1/log t); present as instrument validation, never as novelty.
- Honest paragraph: P2's inflection prediction failed; the rise is smooth
  from epoch 0 because the weighted majority is correct almost immediately on
  MNIST. The refinement (per-example weighted sign aggregate) and the deferred
  test (late-separating task) go here or in limitations.

## 4. E2 — the intervention (causal claim + leakage)

- Fig 2 = reports/fig_e2_intervention.png.
- The clean half: SGD + head constraint ⇒ loss floors 18× above baseline,
  accuracy unchanged, ECE −65%. The causal arrow holds.
- The leakage half (headline finding, was a README "risk"): Adam tunnels
  through the homogeneous body to lower-than-baseline loss with one hidden
  layer. Interventions must price the *function's* scale, not a layer's.
- The overshoot: full-network constraint floors both optimizers but produces
  under-confidence (ECE worsens) — first appearance of "dial, not switch."
- Methods note reviewers will ask about: post-step projection vs weight-norm
  reparametrization (we remove the DOF from the iterate, not the gradient
  geometry); the constrained-lr retuning lesson (from E2b's sweep) cited here.
- P8 answered: row vs global indistinguishable; juicing is one global scale
  (gap ρ_row − ρ_global ≈ 0.006). One sentence; supports temperature-scaling's
  empirical adequacy.

## 5. E2b — the exact case (the SAE anomaly is the volume term)

- Fig 3 = reports/fig_e2b_sae.png (anomaly + removal; norm trajectories;
  ρ_disp; diverging component).
- Closed form first (this is the section's spine): under a → αa the
  decorrelation term is invariant, ReLU-sparse LSE saturates, and the
  variance term pays −2K·log α forever. InfoMax controls shape, not scale.
- Quantitative chain: reproduction (−998.5 vs documented −999); both
  optimizers >94% radial; scale-accounting puts Adam (−260) and SGD (−228)
  in the same range; constraint + manifold-tuned lr closes the raw gap to 9%
  (P7 passes); probe accuracy *improves* to 0.936/0.933.
- The 2601 anomaly is fully explained as scale. Feature quality was never the
  difference.

## 6. E3 — two transport mechanisms (novel plot)

- Fig 4 = reports/fig_e3_optimizer.png (ρ_disp per optimizer; last-quarter
  bars; norm growth).
- Lead with the falsification: P9 predicted Adam pursues the radial direction
  hardest; every grid cell shows the opposite (SGD ρ_disp ≈ 0.8, Adam ≈ 0.003
  at lr 1e-3). Then the replacement: SGD advects (follows the gradient, which
  post-separation points along W); Adam diffuses (per-coordinate normalization
  keeps tangential churn fast; ‖W‖² grows 2.5× faster than SGD by Pythagoras).
- The synthesis with E2b: transport is selected by the objective's radial
  payout. Persistent log-payout (SAE variance term) ⇒ both optimizers advect
  (ρ_disp > 0.94). Collapsed payout (post-separation CE) ⇒ SGD advects by
  default, Adam diffuses. This is the paper's sharpest novel claim; nobody in
  the implicit-bias or calibration literature has either plot.
- AdamW: decay is the one inward radial force; previews E4's weight-decay arm.

## 7. E4 — fix taxonomy (outward-facing result)

- Fig 5 = reports/fig_e4_fixes.png (registered scatter; transient-removed;
  deployed-scale panel). Plus, when results/e4/ece_corrected.json lands:
  the scale-gap |log T*| x-axis version — likely the paper's version.
- Lead with the falsification again: P10's monotone correlation fails
  (Spearman 0.37). The relation is U-shaped: label smoothing under-confident
  at ‖z‖ 4.3 (ECE .069), logitnorm overconfident at ‖z‖ 153 (ECE .081),
  weight decay and focal near the calibrated scale (ECE .028/.030).
- The two-sided law (the outward claim): five heuristics are one mechanism —
  they move the deployed confidence scale — and calibrate exactly insofar as
  they move it toward the calibrated value. Volume control is a dial; both
  directions of error miscalibrate.
- LogitNorm finding: prices loss-side scale, leaves deployed scale unpriced;
  depends on weight decay in published recipes. Scale must be priced in the
  deployed function (echoes E2 leakage: same lesson, weight-space vs
  output-space).
- Pending: temperature-arm verdict on deployment-corrected ECE (P11b);
  integrate ece_corrected.json when committed.

## 8. Related work (from notes/related_work.md)

- Four short subsections, each: what they observed, what they fixed, what we
  add. Calibration (Guo; Mukhoti), OOD/LogitNorm (Wei), implicit bias
  (Soudry; Ji–Telgarsky; Lyu–Li), metric learning (L2-softmax, NormFace,
  CosFace, ArcFace).
- Soudry handled with the most care: their theorem is the linear-separable
  special case, our E1 reproduces its rate, E2–E4 go where it doesn't
  (interventions, optimizers beyond GD, fix unification). Citation-hygiene
  items from notes/related_work.md before freeze.
- Series positioning: one paragraph, 2410 → 2601, this paper supplies the
  scale component of the volume term.

## 9. Limitations and open questions

- MNIST/CIFAR scale; one architecture family per testbed.
- The inflection prediction (P2) remains untested on a late-separating task.
- Weight-level ρ is exact only at the head; deep homogeneous nets hide scale
  anywhere (logit-level measurements are the mitigation, E2 leakage is the
  demonstration).
- The U-shape's x-axis (deployed-scale gap) was identified post-hoc; a
  pre-registered replication on a second dataset would firm it up.
- Diffusive norm growth (E3) deserves its own theory: rate as a function of
  lr, dimension, and gradient noise; we only measured it.

## 10. Conclusion

- One degree of freedom, four literatures, two transport mechanisms, one
  dial. The volume term LSE forgot is measurable, removable, and — within
  this study — sufficient to explain the anomalies it created.

## Figure inventory

| Fig | Source | Status |
|-----|--------|--------|
| 1 | fig_e1_thesis.png | done |
| 2 | fig_e2_intervention.png | done |
| 3 | fig_e2b_sae.png | done |
| 4 | fig_e3_optimizer.png | done |
| 5 | fig_e4_fixes.png | done |
| 6 | fig_e4_corrected.png | done (deployment-corrected ECE, P11b, scale/shape decomposition) |

## Claims-to-evidence map (for internal checking, drop from the paper)

| Claim | Evidence | Registered? |
|-------|----------|-------------|
| Norm grows unbounded, structure frozen | E1 (P1, P3 pass) | yes |
| Late travel is mostly radial | E1 ρ_disp 0.85 (P4 split: disp pass, full 0.32) | yes + A1 |
| Removing scale DOF floors the loss, helps ECE | E2 SGD arms (P6) | yes |
| Scale leaks through homogeneous layers | E2 Adam arms (A2) | amendment |
| 2601 anomaly = volume term | E2b (P7 9%, P12a–d) | yes |
| Advection vs diffusion, objective-selected | E3 (P9 refuted) + E2b ρ_disp | refutation |
| Fix efficacy is two-sided in scale | E4 (P10 refuted, U-shape) | refutation |
| Juicing is global, not per-row | E1 gap + E2 P8 | yes |
