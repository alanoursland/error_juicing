# 1. Introduction

Late in the training of a well-behaved classifier, three curves disagree about
what is happening. The training loss falls steadily. Accuracy — train and
test — is frozen. The softmax entropy collapses toward zero. Nothing the
network *does* is changing; only how confidently it does it. The loss is
buying confidence, not competence. We call this **error juicing**.

The mechanism is structural, not incidental. For any log-sum-exp (LSE)
objective — standard cross-entropy included — the gradient with respect to
each component output equals its softmax responsibility. Responsibilities
depend only on relative outputs, and the argmax assignment is invariant under
uniform scaling of all outputs. The loss is not. Every LSE objective therefore
contains a descent direction, pure output scale, that reduces loss while
leaving assignments, features, and decision structure untouched. In the
implicit EM reading of the predecessor papers, LSE is a mixture marginal
likelihood missing its log-determinant volume term; error juicing is the
collapse pathology of unconstrained implicit EM, expressed as norm growth.

Four literatures have met this pathology independently, and none cites the
others. The calibration community observed that cross-entropy "inherently
induces weight magnification" and fixed it post hoc with temperature scaling.
The OOD-detection community observed that logit norms grow long after accuracy
saturates and fixed it with LogitNorm. The implicit-bias theorists proved
that on separable data the weight norm diverges while only its direction
converges. The metric-learning community normalized features and weights onto
spheres a decade ago because verification quality lives in the angles. One
missing volume term, four names.

This paper makes the degree of freedom measurable, removes it causally, and
prices the fixes. Contributions, each tied to one experiment and one figure:

1. **A measurement instrument** (E1). A radial/tangential decomposition of
   training motion at the final layer, with an exact structure-preserving
   radial direction and a hierarchy of estimators. On MNIST, 85% of
   late-training parameter travel is radial while accuracy is frozen, and the
   direction drift reproduces the known O(1/log t) rate quantitatively.
2. **A causal intervention with a leakage theorem-by-counterexample** (E2).
   Projecting the head onto a fixed-norm manifold floors the loss and improves
   calibration by 65% at zero accuracy cost — for SGD. Adam tunnels through a
   single homogeneous hidden layer to reach lower-than-baseline loss: scale
   must be priced in the whole function, not in one layer.
3. **An exact, closed-form instance** (E2b). The Adam-vs-SGD loss anomaly of
   the predecessor SAE paper equals the −2K log α payout of an unpriced
   variance term, reproduces to three significant figures, and disappears
   (residual gap 9%) under a joint norm constraint, with feature quality
   improving.
4. **Two transport mechanisms** (E3). SGD juices by advection: it follows the
   gradient, which post-separation points along the weights. Adam juices by
   diffusion: per-coordinate normalization keeps tangential churn fast, and
   the norm grows by Pythagoras, 2.5 times faster than SGD's directed growth.
   Which mechanism dominates is selected by the objective's radial payout,
   not by the optimizer alone.
5. **A two-sided law for fixes** (E4). Across five standard regularizers on
   CIFAR-10/ResNet-18, calibration error is not monotone in radial motion; it
   is U-shaped. Fixes calibrate exactly insofar as they move the deployed
   confidence scale toward the calibrated value. Suppressing scale overshoots
   into under-confidence just as growth overshoots into overconfidence.
   Volume control is a dial, not a switch.

A note on method. Every prediction in this paper was registered, with metric
and falsification criterion, before its experiment ran
(`notes/predictions.md`). Three registered predictions failed: the
post-separation inflection (P2), the optimizer-asymmetry mechanism (P9), and
the monotone fix correlation (P10). We report the failures alongside what
replaced them; in each case the replacement is the sharper result. The
amendment log records every post-hoc change of estimator or arm, dated, with
the original text unedited.
