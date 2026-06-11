# 8. Related work

Four literatures observed the same missing volume term. For each: what they
observed, what they fixed, what this paper adds.

**Calibration.** Guo et al. (2017) showed modern networks are overconfident,
that NLL keeps falling after accuracy saturates ("NLL overfitting"), and that
miscalibration worsens as weight decay decreases; their fix, temperature
scaling, is a single learned scale on the logits. Mukhoti et al. (2020)
observed that cross-entropy "inherently induces weight magnification" and
showed focal loss damps it. Temperature scaling is an after-the-fact
correction of exactly the degree of freedom our decomposition isolates — its
empirical adequacy with *one global parameter* matches our finding that the
ρ_row − ρ_global gap is small. Neither work decomposes the training dynamics
or identifies which optimizer steps produce the magnification; we derive the
magnification from the gradient–responsibility identity and measure its share
of the step budget. E4 adds the two-sided refinement: their fixes can also
overshoot into under-confidence.

**OOD detection.** Wei et al. (2022) observed that logit norms grow
throughout training, long after accuracy saturates, and that the growth
breaks softmax-based OOD scores; LogitNorm trains on normalized logits at a
fixed temperature. Their ‖z‖-growth curve is our secondary E4 metric. We add
the relocation result: LogitNorm prices the loss-side scale and leaves the
deployed scale unpriced, depending on weight decay in published recipes — and
the E2 generalization that any partial constraint leaves a channel open.

**Implicit bias.** Soudry et al. (2018) proved that for linear predictors on
separable data with exponential-tail losses, gradient descent's weight norm
diverges like log t while the direction converges to the max-margin solution
at rate O(1/log t); Ji and Telgarsky (2019) refined the rates; Lyu and Li
(2020) extended norm divergence and margin-direction convergence to deep
homogeneous networks — the formal reason scale can hide in any layer, which
E2's leakage demonstrates with one hidden layer. Their theorem is the
linear-separable special case of our decomposition: radial component
persists, tangential decays — proved, where we measure. E1 reproduces their
rate quantitatively (drift slope −1.00 on log-log axes) and we present that
agreement as validation of the instrument. The paper's claims begin where the
theorem stops: nonlinear features, causal interventions (their analysis has
none), optimizers beyond GD (E3's advection/diffusion split and its
objective-dependence), and the unification of fixes (E4).

**Metric learning.** L2-constrained softmax (Ranjan et al. 2017), NormFace
(Wang et al. 2017), CosFace (Wang et al. 2018), and ArcFace (Deng et al.
2019) normalize features and classifier weights onto spheres at a fixed scale
s, because verification quality lives in the angles while softmax rewards
norm growth. This community engineered both of E2's arms a decade ago —
weight normalization is the per-row constraint, the fixed s is the global
temperature — justified geometrically. The implicit EM reading explains why
the sphere is the right manifold: it removes the volume degree of freedom the
LSE objective fails to price.

**The predecessor series.** arXiv:2410.19352 (linear+Abs layers compute
Mahalanobis distance), arXiv:2411.17932 (perturbation evidence for distance
representations), arXiv:2502.02103 (distance vs intensity; bias-free final
layers), arXiv:2512.24780 (gradient descent on LSE is implicit EM; the
gradient–responsibility identity this paper starts from), arXiv:2601.06478
(decoder-free SAE; InfoMax as neural log-determinant; the Adam-vs-SGD anomaly
E2b closes). This paper supplies the volume term's scale component: InfoMax
controls the activation distribution, and scale is the degree of freedom it
leaves open — exhibited in closed form in Section 5.

*Citation hygiene before freeze:* verify the exact wording and page of the
Mukhoti et al. "weight magnification" quote; verify Guo et al.'s weight-decay
observation phrasing; cite the specific LogitNorm figure showing norm growth
after accuracy saturation.
