# Related work: four literatures, one missing volume term

Position for each: what they observed, what they fixed, what they did not derive.
The paper's claim is that all four observe the same object — the unconstrained
scale (volume) degree of freedom of an LSE objective — and that the
gradient–responsibility identity derives it rather than observes it.

## 1. Calibration

- Guo, Pleiss, Sun, Weinberger (2017), *On Calibration of Modern Neural Networks*.
  Observed: modern networks are overconfident; NLL overfits while accuracy
  saturates ("NLL overfitting"); miscalibration worsens as weight decay decreases.
  Fixed: post-hoc temperature scaling — a single learned α on the logits.
- Mukhoti, Kulharia, Sanyal, Golodetz, Torr, Dokania (2020), *Calibrating Deep
  Neural Networks using Focal Loss*. Observed: cross-entropy "inherently induces
  weight magnification"; focal loss damps it.

Position: temperature scaling is exactly an after-the-fact correction of the scale
DOF — it adjusts the one parameter our decomposition isolates, and its empirical
success is evidence that late training motion was predominantly radial. Neither
paper decomposes the training dynamics or identifies which optimizer steps produce
the magnification. We derive the magnification from ∂L/∂z = p − y (post-separation
sign analysis, theory.md §2) and measure its share of the step budget.

## 2. OOD detection

- Wei, Xie, Cheng, Feng, An, Li (ICML 2022), *Mitigating Neural Network
  Overconfidence with Logit Normalization (LogitNorm)*. Observed: logit norm ‖z‖
  grows throughout training and keeps growing after accuracy saturates, causing
  overconfidence that breaks OOD scores. Fixed: train with z/‖z‖ at fixed
  temperature — constraining the norm during training.

Position: LogitNorm is the logit-level intervention of our E2, discovered
empirically for a different downstream goal. Their ‖z‖-growth curve is our
secondary E4 metric. They do not connect to the implicit-bias theory, do not
decompose weight motion, and treat the norm constraint as a trick rather than a
restored volume term.

## 3. Implicit bias theory

- Soudry, Hoffer, Nacson, Gunasekar, Srebro (2018), *The Implicit Bias of Gradient
  Descent on Separable Data*. Proved: for linear predictors on linearly separable
  data with logistic-family (exponential-tail) losses, GD's weight norm diverges
  like log t, the direction converges to the max-margin solution at rate
  O(1/log t), and the loss → 0.
- Ji, Telgarsky (2019): risk and parameter convergence refinements of the same
  phenomenon.
- Lyu, Li (2020), *Gradient Descent Maximizes the Margin of Homogeneous Neural
  Networks*: extends norm-divergence + margin-direction results to deep
  homogeneous nets — the reason scale can hide in any layer (theory.md §6) and
  the justification for our logit-level measurements.

Position — **handle with the most care; a reviewer will check**. Their theorem is
the special case of our decomposition: linear model, separable data, gradient
descent — there, "radial component persists, tangential decays" is proved. We do
not claim novelty over the theorem. The claims that go beyond it: (i) the
decomposition as a *measurement instrument* for nonlinear networks and arbitrary
optimizers; (ii) the causal intervention (E2) — their analysis has no intervention;
(iii) optimizer asymmetry (E3) — their result is GD-specific, and Adam's implicit
bias is known to differ (Wang et al., on Adam's bias, can be cited if needed); (iv)
the unification with calibration/OOD/metric-learning fixes (E4). E1's empirics must
*reproduce* their rate (drift on log-t axis) — agreement with the theorem is
presented as validation of the instrument, not as a finding.

## 4. Metric learning / face recognition

- Ranjan, Castillo, Chellappa (2017), *L2-constrained Softmax Loss*; Wang et al.
  (2017), *NormFace*; Wang et al. (2018), *CosFace*; Deng et al. (2019), *ArcFace*.
  Observed: softmax losses let feature and weight norms grow, inflating
  confidence without improving angular separation; verification quality lives in
  the angles. Fixed: normalize features and/or classifier weights to a sphere and
  operate at a fixed scale s — removing exactly the radial DOF, per-row (weight
  normalization) and globally (fixed s).

Position: this community engineered both of our E2 arms (per-row unit norm = their
weight normalization; fixed global temperature = their scale s) a decade ago,
justified geometrically. The implicit-EM reading explains *why* the sphere is the
right manifold: it is the constraint surface that removes the volume DOF the LSE
objective fails to price.

## 5. Predecessor series (same framework)

- arXiv:2410.19352 — linear+Abs layers compute Mahalanobis distance.
- arXiv:2411.17932 — perturbation evidence for distance-metric representations.
- arXiv:2502.02103 — distance vs intensity representations; OffsetL2; precedent
  for bias-free final layers.
- arXiv:2512.24780 — GD on LSE is implicit EM; the gradient–responsibility
  identity ∂L/∂d_j = −r_j this paper starts from.
- arXiv:2601.06478 — decoder-free SAE; LSE collapses without volume control;
  InfoMax as neural log-determinant; the Adam-vs-SGD loss-gap anomaly E2 must
  explain.

Position: this paper supplies the volume term's *scale* component. InfoMax (2601)
controls the activation distribution (variance, decorrelation) but leaves the
overall scale of the distance metric open; error juicing is what flows through
that opening.

## Citation hygiene

- Quote "inherently induces weight magnification" only with page reference to
  Mukhoti et al. (2020) — verify exact wording against the published version
  before the draft freezes.
- Verify the Guo et al. weight-decay observation wording (their §, "the network
  becomes miscalibrated as weight decay decreases") before quoting.
- Wei et al. 2022 figure showing ‖z‖ growth after accuracy saturation: cite the
  specific figure number in the draft.
