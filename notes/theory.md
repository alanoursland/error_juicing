# Theory: the radial/tangential decomposition

This note defines the paper's core object precisely. Everything in `src/` implements
what is written here. Decisions recorded in the README (bias-free final layer,
ρ_global canonical, ρ_row logged alongside, post-separation scoping) are formalized,
not re-litigated.

## 1. Setting

Model: features h = f_θ(x) ∈ R^d, logits z = Wh with W ∈ R^{K×d}. The final layer
has **no bias**. Reason: the juicing motion is z → αz. With a bias, scaling W alone
gives αWh + b ≠ αz; the exact generator of z → αz is joint (W, b) scaling. Removing
the bias makes W-scaling the exact generator. This matches Soudry et al.'s setting
and the bias-free final layers of arXiv:2502.02103. A sensitivity-appendix run keeps
the bias and projects (W, b) jointly.

Loss: cross-entropy on label y,

    L(z, y) = −z_y + LSE(z),    LSE(z) = log Σ_k exp(z_k).

Gradient–responsibility identity (arXiv:2512.24780):

    ∂L/∂z_k = p_k − 1[k = y],    p = softmax(z).

p_k is the responsibility r_k of component k in the implicit-EM reading. Everything
below applies to any LSE-family objective; CE is the supervised instance.

Per-example weight gradient:

    ∇_W L = (p − y) hᵀ.                                            (1)

Minibatch gradients are averages of (1); all decompositions below are applied to the
minibatch gradient G (and, where stated, to the actual parameter update ΔW).

## 2. The scale direction and the master identity

Global scaling of logits, z → αz, is realized in parameters by W → αW. The
directional derivative of the loss along W is, by linearity of z in W,

    ⟨∇_W L, W⟩_F = ⟨∇_z L, z⟩ = Σ_k p_k z_k − z_y.                 (2)

Identity (2) is the bridge between the weight-level and logit-level views: the
component of the weight gradient along the global scale direction equals the
derivative of the loss with respect to logit scale. It holds per example and in
expectation. It is an Euler identity (z is degree-1 homogeneous in W).

**Sign analysis.** Σ_k p_k z_k is a softmax-weighted average of the logits, so
Σ_k p_k z_k ≤ max_k z_k, with equality only in the deterministic limit.

- If the example is **correctly classified with a strict margin** (z_y > z_k for all
  k ≠ y): Σ_k p_k z_k < z_y, so (2) is **negative** — scaling up reduces loss.
  As α → ∞, L(αz) → 0 like exp(−α·margin).
- If the example is **misclassified** (z_y < max_k z_k): for large α,
  Σ_k p_k z_k → max_k z_k > z_y, so (2) becomes **positive** — scaling up
  increases loss without bound (linearly in α).

Consequence, stated as the regime split used throughout the paper:

- **Pre-separation** (train error > 0): the batch loss has a finite optimal scale
  α*. The radial direction is not a pure descent direction; ρ stays bounded away
  from 1.
- **Post-separation** (train error ≈ 0): every example has negative scale
  derivative; the global scale direction is a strict descent direction with no
  finite minimizer. Loss can be reduced indefinitely with no change to any
  structural quantity (Section 5). This is **error juicing**.

## 3. The two decompositions

Let G ∈ R^{K×d} be a gradient (or update) at parameter W. Write w_j, g_j ∈ R^d for
the j-th rows, ŵ_j = w_j/‖w_j‖, and ⟨·,·⟩_F for the Frobenius inner product.

### 3.1 Global (canonical)

Radial subspace: span{W}, one-dimensional. With Ŵ = W/‖W‖_F:

    G_rad = ⟨G, Ŵ⟩_F Ŵ,    G_tan = G − G_rad,
    ρ_global = ‖G_rad‖²_F / ‖G‖²_F = ⟨G, Ŵ⟩²_F / ‖G‖²_F.

This is the projection onto the exact generator of z → αz. It is the canonical ρ:
the thesis figure, the abstract, and registered prediction P2 use ρ_global.

### 3.2 Per-row (EM-facing)

Radial subspace: span{e_1ŵ_1ᵀ, …, e_Kŵ_Kᵀ}, K-dimensional (e_j the j-th standard
basis vector). The basis elements are mutually orthogonal, so the projection is

    G_rad = Σ_j ⟨g_j, ŵ_j⟩ e_j ŵ_jᵀ,
    ρ_row = Σ_j ⟨g_j, ŵ_j⟩² / ‖G‖²_F.

Interpretation: row j is mixture component j (2512 framing); scaling row j alone is
the volume degree of freedom of component j. ρ_row measures total per-component
volume motion.

### 3.3 Properties (each one unit-tested in `src/test_decomposition.py`)

1. **Orthogonality**: ⟨G_rad, G_tan⟩_F = 0, for both decompositions.
2. **Pythagoras**: ‖G_rad‖² + ‖G_tan‖² = ‖G‖²; hence ρ ∈ [0, 1].
3. **Subspace scale-invariance**: both radial subspaces depend on W only through
   the directions ŵ_j (resp. Ŵ), so they are invariant under W → αW, α > 0. The
   tangential component of a fixed G is unchanged by W → αW.
4. **Nesting**: W = Σ_j ‖w_j‖ e_j ŵ_jᵀ lies in the per-row subspace, so the global
   subspace is contained in the per-row subspace, and therefore

       ρ_row ≥ ρ_global    for every G.

   The gap ρ_row − ρ_global measures per-component volume drift beyond global
   temperature.
5. **Purity**: G ∝ W ⇒ ρ_global = ρ_row = 1. G with ⟨g_j, ŵ_j⟩ = 0 for all j ⇒
   ρ_row = ρ_global = 0.

### 3.4 Structure preservation

Only the global direction is structure-preserving. For α > 0, softmax(αz) preserves
the ordering of z, the argmax, and all pairwise logit differences up to a common
factor; the induced classifier is identical. Per-row rescaling with unequal factors
α_j changes relative logits (α_j w_jᵀh vs α_k w_kᵀh) and can flip the argmax —
counterexample: K = 2, z = (1.0, 0.9), α = (1, 2) flips the prediction. The
headline claim ("loss descends along a structure-preserving direction") therefore
uses ρ_global. ρ_row is reported as the EM-facing quantity, never as the basis of
the structure-preservation claim.

## 4. ρ_grad vs ρ_step

For plain SGD, ΔW = −η G, so the radial fraction of the gradient and of the step
coincide. For Adam-family optimizers they do not: the update is a coordinate-wise
rescaled gradient, and rescaling does not commute with projection. Define

    ρ_grad = radial fraction of G,
    ρ_step = radial fraction of ΔW (the realized parameter change).

E3's claim is about step budget, so E3 logs ρ_step per optimizer (and ρ_grad for
reference). E1 uses plain SGD, where the two coincide; E1 logs the common value.

Mechanism hypothesis for E3 (registered as P6): the radial gradient component is
small but persistent in sign; SGD's step length scales with gradient magnitude and
therefore mostly ignores it; Adam normalizes per-coordinate magnitude away and
pursues the persistent direction indefinitely.

## 5. What "structure" means (invariants table)

Quantities unchanged by motion along the global scale direction W → αW, α > 0:

| Quantity                          | Invariant? |
|-----------------------------------|------------|
| argmax prediction                 | yes        |
| logit ordering                    | yes        |
| normalized direction Ŵ            | yes        |
| features h (upstream params)      | yes        |
| relative logit ratios z_i/z_j     | yes        |
| train/test accuracy               | yes        |
| loss                              | **no**     |
| softmax confidence / entropy      | **no**     |
| calibration (ECE)                 | **no**     |
| ‖W‖, ‖z‖                          | **no**     |

The pathology in one row each: everything a practitioner calls "what the network
learned" is in the upper block; everything the loss rewards late in training is in
the lower block.

## 6. Logit-level decomposition (architecture-agnostic)

Deep homogeneous networks can place scale in any layer (e.g. ReLU nets satisfy
f(αθ) = α^L f(θ) layer-wise), so weight-level ρ on the final layer undercounts
juicing in ResNets. The logit-level decomposition measures the same object at the
output, regardless of where the parameters put it.

Fix a probe set of n examples (held fixed across training; drawn from the train
set). At logging step t with logits z_i(t) per example i, define the displacement
dz_i = z_i(t) − z_i(t−1) between consecutive logged points and its radial fraction

    ρ_logit,i(t) = (ẑ_i · dz_i)² / ‖dz_i‖²,    ẑ_i = z_i(t−1)/‖z_i(t−1)‖.

Estimator: the **mean over probe examples at each logging step** (mean-per-step;
stabler than integrating per-example first). The E4 headline metric is the
integral over training,

    R = Σ_t mean_i [ ρ_logit,i(t) · ‖dz_i(t)‖ ],

i.e. radial-weighted logit motion. The unweighted curve ∫ d‖z‖ = Σ_t mean_i
(‖z_i(t)‖ − ‖z_i(t−1)‖) is logged as a secondary curve: it is the LogitNorm
literature's own measurement and the bridge to it.

Auxiliary logit-level signals, logged at eval checkpoints: mean ‖z‖, responsibility
entropy (mean per-example softmax entropy on the train set), margin distribution.

## 7. Relation to Soudry et al. (2018)

Their theorem (linear predictor, linearly separable data, logistic-family loss,
gradient descent): ‖w(t)‖ grows like log t, w(t)/‖w(t)‖ converges to the max-margin
direction at rate O(1/log t), loss → 0. In our terms: post-separation, the
tangential component decays (direction converges, slowly) while the radial component
persists (norm diverges) — their result is the special case of the decomposition
where the model is linear and the data separable, proved rather than measured.

Two consequences for the paper:

1. E1's drift curve must be plotted on a log-t axis and the prediction phrased as
   "drift decays toward zero, consistent with O(1/log t)" — never "directions
   freeze". The empirics should *reproduce* their rate in the linear case; this is
   confirmation, not a threat.
2. The paper's contribution begins where the theorem stops: nonlinear features,
   the causal intervention (E2), the optimizer dependence of radial allocation
   (E3, outside their GD analysis), and the unification of fixes (E4).

## 8. Implementation contract

- `src/common.py` implements `radial_tangential_global`, `radial_tangential_row`,
  `rho_global`, `rho_row`, the logit-level estimator, and ECE.
- Unit tests cover exactly the properties in §3.3 (orthogonality, Pythagoras,
  subspace scale-invariance, nesting, purity) with exact tolerances (float64,
  atol 1e-10). Nothing else is unit-tested (README convention).
- ρ is computed on the minibatch gradient actually used for the step (E1) and on
  the realized update ΔW (E3).
