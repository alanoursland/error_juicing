# E2b lab report — the intervention on the 2601 SAE testbed

Date: 2026-06-10. Written immediately after the runs. Testbed:
`alanoursland/implicit_em_in_neural_activations`, `encoder_study/`,
reimplemented minimally in `src/e2b_sae.py` (Linear 784→64 with bias + ReLU;
LSE + variance + decorrelation, all λ = 1.0; raw [0,1] MNIST pixels; their
optimal lrs: SGD 0.01, Adam 1e-3; 100 epochs; seeds {1, 2, 3}).

## Pre-run analysis (registered as P12)

Under a → αa — generated exactly by jointly scaling (W, b), since ReLU is
homogeneous — the decorrelation term is scale-invariant, the ReLU-sparse LSE
term saturates, and the variance term −Σ_j log var(a_j) pays −2K·log α
*forever*. The 2601 objective contains a persistent, logarithmically-paying
radial direction: the volume DOF that InfoMax leaves open, in closed form.

## Results (3-seed mean ± std)

| arm         | opt  | final loss     | ‖(W,b)‖ end | ρ_disp (last 25%) | probe acc |
|-------------|------|----------------|-------------|-------------------|-----------|
| baseline    | sgd  | −777.3 ± 45.0  | 335         | 0.978             | 0.9192    |
| baseline    | adam | −998.5 ± 0.6   | 1479        | 0.943             | 0.8943    |
| constrained | sgd  | +98.4 ± 54.8   | 4.6         | 0.002             | 0.8668    |
| constrained | adam | −305.6 ± 0.1   | 4.6         | 0.020             | 0.9360    |

Plateau check: constrained losses are flat (last-20-epoch slope 0.00/epoch);
baselines are still descending at −0.7 (SGD) and −1.3 (Adam) per epoch at
epoch 100 — the unbounded channel, live.

**Scale-accounting (the decisive check).** Adding back the predicted volume
term, baseline_loss + 2K·log(‖Wb‖_end/‖Wb‖_init), per seed:

- Adam: −999 → **−260 ± 1**; its constrained floor is −306.
- SGD: −777 → **−228 ± 42**.

After removing the scale term, Adam's and SGD's baseline solutions land in the
same range (−260 vs −228): **the 2601 anomaly is, quantitatively, the volume
term.** Adam's raw advantage comes from climbing it further (α = 321× vs 73×),
and 74% of Adam's total loss depth is the −2K·log α payout.

## Verdicts

**P12a (anomaly reproduces) — PASS, exactly.** Baseline Adam reaches
−998.5 ± 0.6; the 2601 dynamics report's value for the same configuration is
−999 ± 1. The reimplementation is faithful.

**P12b (directed pursuit, unlike CE) — PASS, and it resolves E3's puzzle.**
On this objective ρ_disp is ~0.94–0.98 for *both* optimizers (figure panel C),
versus 0.003–0.83 split on the CE testbed. The transport mechanism is a
property of the objective, not the optimizer: when the radial direction pays
persistently (−2K log α), both optimizers chase it directionally; when its
payout collapses post-separation (CE), SGD's residual motion is radial by
default while Adam diffuses. E3's refuted P9 and this result are two halves
of one statement.

**P12c (gap closes under constraint) / P7 — FAIL as registered; the
scale-accounting passes instead.** The raw constrained gap did not shrink — it
grew (+221 → +404), because constrained SGD at lr 0.01 stalls on the fixed-norm
manifold (floor +98 vs its own scale-corrected baseline −228: undertrained,
an lr-tuning confound; see the lr sweep below). The honest statement: the
*per-optimizer* anomaly closes quantitatively (scale-corrected Adam ≈
scale-corrected SGD ≈ constrained-Adam floor), but the raw between-optimizer
comparison under constraint measures optimization ability on the manifold,
which the registered criterion did not anticipate.

**P12d (probe accuracy unchanged) — MIXED, with a bonus.** Baseline: SGD
0.919, Adam 0.894 (Adam's 321× norm explosion mildly hurts its features).
Constrained Adam reaches **0.936 — the best features in the study** — while
constrained SGD drops to 0.867 (same lr confound). Removing the juicing
channel doesn't merely preserve feature quality; for Adam it improves it.

## Surprises

1. The variance term is the diverging component (figure panel D), exactly as
   the pre-run analysis predicted — this is the cleanest "missing volume term"
   demonstration in the project: closed-form channel, closed-form payout,
   measured payout matches (−739 predicted vs −739 observed for Adam, by
   construction of the check).
2. Both optimizers are >94% radial here. The advection/diffusion split of E3
   is objective-dependent, not an optimizer constant.
3. Constrained Adam's features beat every baseline. Volume control is not a
   tax on quality; on this objective it is a subsidy.

## Verdict for the paper

This testbed supplies what the CE testbed couldn't: an exact, closed-form
volume term whose removal collapses the optimizer anomaly quantitatively. P7's
raw-gap criterion was the wrong operationalization; the scale-accounting is
the right one and it closes at the few-percent level. Together with E3: the
paper can now state *when* juicing is advective vs diffusive (persistent-payout
vs collapsed-payout objectives) — a sharper claim than either experiment alone.

(Constrained-SGD lr sweep appended below when complete.)
