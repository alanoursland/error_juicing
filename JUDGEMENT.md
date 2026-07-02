# Judgement: Error Juicing / LSE Volume Control

External assessment of this research project — methodology, results, novelty,
and likely impact. Written 2026-07-02 by Claude (Fable 5) after reading the
theory notes, all five lab reports, the full draft, both feedback reviews, the
source code and tests, and checking novelty claims against the literature.

## Overall verdict

**Real research, not slop. Methodologically excellent. Scientifically solid.
Novelty narrower than the draft claims. Impact: modest but genuine — TMLR or
strong-workshop tier as positioned; could aim higher if repositioned.**

Grades: A− as a research project, B+ as a paper in its current draft state.

## Methodology (the standout)

The epistemics are better than the median accepted ML paper:

- Predictions registered with metrics and falsification criteria *before*
  experiments ran (`notes/predictions.md`). Four failed (P2, P9, P10, P11a/c);
  every failure is disclosed, diagnosed mechanistically, and the post-hoc
  reframing is labeled post-hoc. A dated amendment log preserves original text.
- The radial/tangential decomposition in `src/common.py` matches
  `notes/theory.md` exactly; `src/test_decomposition.py` verifies all seven
  properties (orthogonality, Pythagoras, scale-invariance, nesting, purity,
  joint projection, logit estimator) at tight tolerances. All pass.
- Reproducibility is real: pinned seeds, 3 seeds per config, YAML configs,
  atomic metric writes, resumable checkpoints, every figure regenerated from
  committed JSON.
- Negative results reported prominently: Adam's scale leakage through a
  homogeneous hidden layer, full-network constraints inducing under-confidence,
  LogitNorm's deployed-‖z‖ explosion, learned temperature going stale.

The estimator hierarchy (per-step minibatch ρ at noise floor; epoch-displacement
ρ as the signal; full-batch gradient ρ in between) is itself a citable
methodological contribution.

## Results, ranked by how much the community would care

1. **E3 — advection vs. diffusion (best result, currently buried at #4).**
   SGD grows the weight norm by *directed* radial motion (ρ_disp ≈ 0.8); Adam
   grows it 2.5× faster almost entirely by *diffusion* — tangential churn
   inflating the norm by Pythagoras (ρ_disp ≈ 0.003 at lr 1e-3). The corollary
   that the transport mechanism is selected by the objective's radial payout,
   not the optimizer (E2b: both optimizers go directed when the payout is
   logarithmic-forever), is sharp and, as far as checked, not stated anywhere
   in the literature. This should be the paper's headline.

2. **E4 — scale/shape decomposition of miscalibration (most practically
   useful).** One global scalar repairs 63–90% of every fix's miscalibration;
   label smoothing is the exception because it distorts target *shape*, which
   no temperature repairs (ECE at optimal T* double all other arms). The
   U-shape (suppression overshoots into under-confidence just as growth
   overshoots into overconfidence) is a genuine two-sided refinement of
   folklore. Caveat: post-hoc after P10 was falsified (Spearman 0.37 vs.
   registered ≥ 0.8), one dataset, unreplicated.

3. **E2b — closed-form volume accounting (most rigorous, least transferable).**
   The predecessor SAE's Adam-vs-SGD anomaly reproduces exactly (−998.5 ± 0.6),
   the −2K·log α variance payout accounts for 74% of the gap, the residual
   under joint constraint is 9% of baseline, and constrained features *improve*
   (0.894 → 0.936). Beautiful science, but it lives inside the author's own
   five-paper framework; the audience that can value it without accepting that
   scaffold is small.

4. **E1 and E2 — confirmatory.** E1 empirically reproduces Soudry et al.'s
   theorem (drift slope −0.98 to −1.05 on log-log axes) and the draft says so.
   E2's interventions (norm constraint → loss floors, calibration +65%,
   accuracy unchanged) were engineered a decade ago by the metric-learning
   community and by AdamP. The leakage finding is a nice sharpening of Lyu–Li
   homogeneity, not a new phenomenon.

## Novelty threats (must be handled before submission)

The intro's hook — "four literatures have met this pathology independently,
and none cites the others" — is factually shaky. Mukhoti et al. cite Guo;
Wei et al. cite the calibration literature; and at least three uncited works
already sit across these literatures:

- **Bai et al., "A Geometric Perspective towards Neural Calibration via
  Sensitivity Decomposition" (arXiv:2110.14577, 2021).** Already decomposes
  softmax behavior into norm vs. angular components and connects it to
  calibration. Structurally the same decomposition idea applied to the same
  problem. **The single biggest novelty threat; must be cited and
  differentiated.**
- **Heo et al., "AdamP" (arXiv:2006.08217, ICLR 2021).** Observed that
  momentum optimizers induce excessive norm growth on scale-invariant weights
  and built an optimizer that *projects out the radial component of the
  update* — the E2 intervention and half of E3's observation exist as an
  engineering artifact. Uncited.
- **Poggio et al., "Complexity control by gradient descent in deep networks"
  (Nature Communications, 2020).** States that under separability norms grow
  to infinity and only normalized outputs matter.

The honest reframing — "we add a measurement instrument and causal accounting
to a connection several papers have gestured at" — is a demotion but remains
defensible. The E3 transport-mechanism split survives all three threats.

## The implicit-EM framing: mostly liability

Every experiment stands on standard language ("CE is argmax-invariant but not
loss-invariant under logit scaling"). The "missing log-determinant volume
term" framing requires readers to buy arXiv:2512/2601, adds no predictive
content for E1–E4, and pattern-matches to grand-unified-personal-theory
framing that triggers "reject: overclaimed." It pays rent in exactly one
place: E2b, where the volume term is computed in closed form. Recommendation:
demote implicit-EM to a discussion-section interpretation and present E2b as
where it becomes exact.

## What caps the impact

- **Scale.** MNIST + CIFAR-10/ResNet-18 is 2019-scale evidence in 2026. The
  field's attention is on transformers, where norm growth is also documented,
  AdamW is the default, and the AdamW-cuts-diffusive-growth finding is
  directly relevant. One transformer-scale run is the single highest-leverage
  addition.
- **The U-shape is post-hoc and single-dataset.** A pre-registered replication
  on a second dataset would close the paper's weakest flank.
- **Audience.** Calibration researchers (E4), optimization/implicit-bias
  people (E3), SAE-interpretability crowd (E2b). Real but niche.

## Outstanding draft defects (flagged by both feedback reviews, still unfixed)

1. §2.3 says "all relative logits are invariant" under scaling — false; only
   argmax, ordering, and ratios are. Load-bearing and reviewer-catchable.
2. E4's T* protocol (test-set optimized?) is unstated — and the 63–90% repair
   number is the strongest unification claim in the paper.
3. E2's under-confidence claim has no signed evidence (cite E4's T* = 0.79 or
   drop the directional claim).
4. The SAE sparsity assumption (every example has an inactive unit) is
   asserted, never measured.
5. Intro claim unscoped: "Every LSE objective contains a descent direction"
   needs "once separated."

All are one-sentence-to-one-paragraph fixes.

## Is it AI slop?

No. Slop's signature is unfalsifiable framing, results never run, derivative
claims made in ignorance of their ancestry, and prose that survives no
skeptical contact. This project registered falsifiable predictions, ran the
experiments, reported four refutations with mechanistic diagnoses, unit-tested
its math, committed its metrics, and engages its neighbors in detail. Its risk
is not fakeness — it is *rediscovery with better instrumentation*, a
respectable but harder-to-sell genre. The gap between this and slop is the
gap between "P9: REFUTED with opposite sign, here is the actual mechanism"
and a paper that would never have registered P9 at all.

## Recommendations, in priority order

1. Fix the five draft defects above (one editing pass).
2. Cite and differentiate from Bai et al. 2021, AdamP, and Poggio et al. 2020.
3. Lead with E3's advection/diffusion split; reframe related work as
   unification-first rather than defense-first.
4. Demote implicit-EM to interpretation; let E2b carry it.
5. Add one transformer-scale run of the E3/E4 measurements.
6. Decide: ship the U-shape as labeled post-hoc (TMLR-appropriate) or
   replicate on a second dataset first (needed for a top-venue attempt).

Realistic venue as-is: TMLR (values rigor and honest negative results, no
significance bar). With items 2–5: a credible NeurIPS/ICLR submission led by
E3, though it will still have to win the "isn't this Soudry + AdamP +
sensitivity-decomposition?" argument.
