# LSE Volume Control: Error Juicing in Log-Sum-Exp Objectives

Draft sections. Each file is one section of the paper; this index mirrors
`../outline.md`. Figures live in this directory as `.png` (for these md
files) and `.pdf` (vector, for LaTeX). All figures regenerate from committed
metrics via `src/fig_e*.py`.

Title candidate: *Error Juicing: Every Log-Sum-Exp Objective Pays for
Confidence It Didn't Earn.*
Conservative alternative: *Volume Control in Log-Sum-Exp Objectives: A
Radial/Tangential Decomposition of Training Dynamics.*

## Sections

1. [Introduction](01_introduction.md)
2. [Theory: the radial/tangential decomposition](02_theory.md)
3. [E1 — the decomposition measured](03_e1_decomposition.md)
4. [E2 — the intervention](04_e2_intervention.md)
5. [E2b — the exact case: the SAE anomaly is the volume term](05_e2b_sae.md)
6. [E3 — two transport mechanisms](06_e3_optimizers.md)
7. [E4 — fix taxonomy](07_e4_fixes.md)
8. [Related work](08_related_work.md)
9. [Limitations and open questions](09_limitations.md)
10. [Conclusion](10_conclusion.md)

## Figure inventory

| Fig | File | Section | Status |
|-----|------|---------|--------|
| 1 | fig_e1_thesis.{png,pdf} | E1 | done |
| 1s | fig_e1_estimators.{png,pdf} | E1 (appendix) | done |
| 2 | fig_e2_intervention.{png,pdf} | E2 | done |
| 3 | fig_e2b_sae.{png,pdf} | E2b | done |
| 4 | fig_e3_optimizer.{png,pdf} | E3 | done |
| 5 | fig_e4_fixes.{png,pdf} | E4 | needs scale-gap panel once `results/e4/ece_corrected.json` lands |
