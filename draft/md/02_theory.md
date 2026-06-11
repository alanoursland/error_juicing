# 2. Theory: the radial/tangential decomposition

## 2.1 Setting

Let h = f_θ(x) ∈ R^d be features and z = Wh the logits, W ∈ R^{K×d}. The
final layer has no bias. The choice is load-bearing: the juicing motion is
z → αz, and with a bias, scaling W alone gives αWh + b ≠ αz. Removing the
bias makes W-scaling the exact generator of output scaling. It also matches
the setting of Soudry et al. (2018) and the bias-free final layers of the
predecessor experiments (arXiv:2502.02103). A sensitivity run with the bias
kept and (W, b) projected jointly changes nothing qualitative (E1, appendix).

The loss is cross-entropy, L(z, y) = −z_y + LSE(z). The
gradient–responsibility identity of arXiv:2512.24780 gives
∂L/∂z_k = p_k − 1[k = y] with p = softmax(z): the gradient with respect to
each output is its responsibility. Everything below applies to any LSE-family
objective; CE is the supervised instance.

## 2.2 The master identity and the regime split

Because z is degree-1 homogeneous in W, the loss derivative along W is an
Euler identity:

> ⟨∇_W L, W⟩_F = ⟨∇_z L, z⟩ = Σ_k p_k z_k − z_y.

The right side is a softmax-weighted average of logits minus the true logit.
For a correctly classified example with a strict margin it is negative:
scaling up reduces loss, and L(αz) → 0 like exp(−α·margin). For a
misclassified example it turns positive for large α: scaling up amplifies the
error without bound. Hence the regime split. Pre-separation, the batch has a
finite optimal scale and the radial direction is not a pure descent
direction. Post-separation (train error ≈ 0), the radial direction is a
strict descent direction with no finite minimizer: the loss can fall forever
while nothing structural changes. E1 refines this binary statement — the
radial direction begins to pay as soon as the *responsibility-weighted
majority* of examples is correctly classified, which on easy data happens
within the first epoch.

## 2.3 Two radial subspaces

Let G be a gradient or update at W, with rows g_j, w_j and ŵ_j = w_j/‖w_j‖.

**Global (canonical).** The radial subspace is span{W}, one-dimensional.
With Ŵ = W/‖W‖_F,

> ρ_global = ⟨G, Ŵ⟩²_F / ‖G‖²_F.

This is the projection onto the exact generator of z → αz. It is provably
structure-preserving: argmax, logit ordering, and all relative logits are
invariant under W → αW, α > 0. The thesis figure and the registered
predictions use ρ_global.

**Per-row (EM-facing).** The radial subspace is span{e_j ŵ_jᵀ},
K-dimensional, and ρ_row = Σ_j ⟨g_j, ŵ_j⟩² / ‖G‖²_F. Row j is mixture
component j; per-row scale is the per-component volume degree of freedom.
It is not structure-preserving: unequal row scaling can flip the argmax.

The global direction lies inside the per-row subspace, so ρ_row ≥ ρ_global
for every G. The gap measures per-component volume drift beyond global
temperature. Empirically the gap is small (E1: ≈ 0.006; E2: row and global
constraints indistinguishable): juicing is one global scale. Both
decompositions satisfy orthogonality, the Pythagorean identity, and
scale-invariance of the subspaces under W → αW; these exact properties are
unit-tested to 1e-10 (`src/test_decomposition.py`).

What scale motion cannot change: argmax predictions, logit ordering, the
normalized direction Ŵ, the features, train and test accuracy. What it does
change: the loss, softmax confidence and entropy, calibration, ‖W‖ and ‖z‖.
Everything a practitioner calls "what the network learned" is in the first
list. Everything the loss rewards late in training is in the second.

## 2.4 Estimators: where the optimizer wanders vs where it travels

ρ can be evaluated on three objects, and they answer different questions.

1. **Per-step minibatch gradient.** Noise-dominated. Minibatch tangential
   noise is unbiased and large; the radial signal is small but consistently
   signed. On MNIST the per-step ρ_global never leaves the noise floor
   (≈ 4× chance = 1/Kd) even at zero train error.
2. **Full-batch gradient (per checkpoint).** The expected per-step loss
   change is first-order in the full-batch gradient,
   E[ΔL] ≈ −η‖∇L_full‖², so ρ of the full-batch gradient is the radial share
   of expected loss reduction.
3. **Epoch displacement W_end − W_start.** The radial share of realized
   travel. Tangential gradient components oscillate and cancel within an
   epoch; radial components accumulate.

The hierarchy (noise floor → 0.32 → 0.85 on MNIST, E1) is a measurement
result in its own right: where the optimizer wanders and where it travels are
different questions. For plain SGD the gradient and step coincide; for
adaptive optimizers we additionally distinguish ρ of the realized update
(ρ_step), which E3 uses.

## 2.5 Two norm-growth channels

For an update ΔW at W the norm change splits exactly:

> ‖W + ΔW‖² − ‖W‖² = 2⟨W, ΔW⟩ + ‖ΔW‖².

The first term is **advection**: signed, gradient-driven, the channel the
regime analysis predicts. The second is **diffusion**: nonnegative, driven by
step energy regardless of direction — tangential churn in high dimension
inflates the norm by Pythagoras. An optimizer can grow the norm with almost
no radial displacement at all. E3 shows the two channels are real and that
SGD and Adam weight them oppositely.

## 2.6 The implicit EM reading

In the framing of the predecessor series, LSE is the marginal likelihood of a
mixture whose log-determinant volume term has been dropped. InfoMax-style
regularizers (variance + decorrelation, arXiv:2601.06478) restore control of
the activation *distribution* but not of its overall *scale*: scale is the
volume degree of freedom the regularizers leave open. E2b exhibits this in
closed form — an objective whose variance term pays −2K log α for pure scale
growth, forever. We keep the EM thread short here; it is the series'
through-line, not this paper's main road.
