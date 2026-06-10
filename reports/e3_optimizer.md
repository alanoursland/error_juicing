# E3 lab report — optimizer asymmetry

Date: 2026-06-10. Written immediately after the runs, before drafting. The
README flagged E3 as "most likely to produce surprises"; it refuted its own
registered prediction and produced the best mechanism result of the study.

## Setup

Same MNIST MLP as E1/E2, 80 epochs, seeds {0, 1, 2}. Grid: SGD lr {0.05, 0.1},
Adam lr {3e-4, 1e-3}, AdamW (default wd 0.01) lr {3e-4, 1e-3}. Logged: per-step
ρ of the realized update ΔW (rho_step) and of the gradient; per-epoch ρ of the
epoch displacement (rho_disp, amendment A1); ‖W‖_F; train error; test accuracy.
Metrics in `results/e3/*.json`; figure regenerates via `src/fig_e3.py`.

## Results (3-seed means, last quarter of training)

| opt   | lr    | ρ_disp (last 25%) | ‖W‖ end | Δ‖W‖²/epoch (late) | test acc |
|-------|-------|-------------------|---------|--------------------|----------|
| sgd   | 0.05  | 0.782 ± 0.012     | 9.98    | 0.48               | 0.9815   |
| sgd   | 0.1   | 0.831 ± 0.006     | 10.83   | 0.42               | 0.9824   |
| adam  | 3e-4  | 0.401 ± 0.015     | 9.47    | 0.49               | 0.9829   |
| adam  | 1e-3  | 0.003 ± 0.000     | 11.73   | 1.06               | 0.9806   |
| adamw | 3e-4  | 0.200 ± 0.012     | 9.18    | 0.42               | 0.9827   |
| adamw | 1e-3  | 0.206 ± 0.109     | 9.49    | 0.27               | 0.9818   |

## Result against the registered prediction

**P9 — REFUTED, with the opposite sign, robustly.** The prediction said Adam
allocates a larger share of its realized step budget to the radial direction
(ρ_step(Adam) > 1.5× ρ_step(SGD)). The data: SGD's epoch displacement is
~80% radial; Adam's is 40% (lr 3e-4) down to 0.3% (lr 1e-3); AdamW ~20%. Every
cell of the grid contradicts the registered direction. The proposed mechanism
("the radial gradient is small-but-persistent; Adam normalizes magnitude away
and pursues it") is wrong at the head-weight level.

## The replacement mechanism (the actual result)

Both optimizers juice; they juice **differently**:

- **SGD: directed radial ascent.** Its step length tracks gradient magnitude.
  As the direction converges (E1: drift → 0), the tangential gradient dies and
  what remains is the persistent radial signal — so SGD's realized motion
  becomes almost purely radial (ρ_disp → 0.83). SGD climbs the scale axis
  *because it follows the gradient*, which post-separation points along W.
- **Adam: diffusive norm inflation.** Per-coordinate normalization keeps every
  coordinate moving at ~lr regardless of signal size, so Adam's late-training
  motion is dominated by tangential churn (ρ_disp → 0.003 at lr 1e-3; visible
  in panel A as early-training ρ ≈ 0.5–0.65 collapsing as gradients shrink and
  normalization amplifies noise). But tangential churn in 2560 dimensions
  inflates the norm anyway — for ΔW ⊥ W, ‖W + ΔW‖² = ‖W‖² + ‖ΔW‖², growth by
  Pythagoras. Measured: Adam lr 1e-3 grows ‖W‖² at 1.06/epoch late in
  training, **2.5× SGD's rate** (0.42), while being 0.3% radial. The norm
  growth is a *side effect of diffusion*, not a pursuit of the radial
  gradient.

So the deeper claim survives inverted: Adam does sharpen confidence faster
(consistent with E2's baseline ECE: Adam 0.0164 vs SGD 0.0092) — not by
following the radial descent direction but by diffusing into larger norms. One
pathology, two transport mechanisms: advection (SGD) vs diffusion (Adam). This
also explains E2's leakage asymmetry: a norm constraint on the head blocks
SGD's directed channel, but Adam's diffusion immediately exploits whatever
unconstrained subspace remains (the body).

Secondary observations:

- **AdamW's weight decay is partial volume control.** It cuts the diffusive
  growth (0.27 vs 1.06 at lr 1e-3) and ends at smaller norms — mechanically,
  decay is the only term in AdamW that scales with ‖W‖ and it points radially
  inward. This connects E4's weight-decay arm to the framework before E4 runs.
- ρ_step (per-step) is at the noise floor for all optimizers (A1 holds
  universally); the optimizer signature only appears in the aggregate.
- Accuracy is flat across the grid (98.1–98.3%): none of this is about
  structure.

## Implications for the paper and registered predictions

1. P9's text stands refuted in the registry; the report's mechanism becomes
   the E3 section. The "nobody has this plot" claim survives — panel A
   (ρ_disp per optimizer) and the Δ‖W‖²-rate table are the novel artifacts,
   now with the correct interpretation.
2. theory.md should gain a short section: norm growth decomposes into a
   directed term (2⟨Ŵ, ΔW⟩, gradient-driven) and a diffusive term (‖ΔW⊥‖²/‖W‖,
   noise-driven); optimizers weight the two differently. Prediction for the
   E4/ResNet setting (not yet registered, do so before running E4): fixes
   that bound the norm (LogitNorm, weight decay) suppress *both* channels;
   temperature-only fixes suppress neither channel of ‖W‖ growth but cancel
   its effect on the loss.
3. E2's P7 anomaly (Adam reaching lower loss in 2601) now has a candidate
   reading: in settings where the floor is reachable by scale alone, diffusion
   finds it faster than advection. Test on the SAE testbed when unblocked.

## Verdict

The registered prediction failed and the experiment succeeded. Proceed to E4
on the GPU; register the channel-suppression prediction first.
