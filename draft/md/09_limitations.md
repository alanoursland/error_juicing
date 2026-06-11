# 9. Limitations and open questions

**Scale of evidence.** MNIST and CIFAR-10; one architecture family per
testbed (MLP, ResNet-18, one-layer SAE); three seeds per configuration.
The decomposition and the closed-form SAE analysis are exact; every claim
about magnitudes (85% radial travel, the U-shape, the 2.5× diffusion rate) is
an empirical claim at this scale and should be read as such.

**The inflection prediction remains untested.** P2 failed on MNIST for an
identifiable reason — separation effectively happens in the first epoch — and
its refined form (radial dominance tracks the responsibility-weighted correct
fraction) has not yet been tested on a task that separates late. Label-noise
MNIST or unaugmented CIFAR are the natural candidates.

**Weight-level measurement is exact only at the head.** Deep homogeneous
networks can place scale in any layer (Lyu and Li 2020); E2's leakage result
is the constructive demonstration. The logit-level measurements are the
mitigation, not a resolution: they observe the deployed scale without
locating it. A layer-wise accounting of where scale accumulates is open.

**The U-shape's final form was identified post-hoc.** P10's monotone form
was registered and failed; the scale/shape decomposition of Section 7 is the
replacement. The deployment-corrected recomputation confirmed its sign
structure (label smoothing alone under-confident) and the dominance of the
global-scale component (63–90% removable by one scalar), but also showed
that deployed ECE is not monotone in the scale gap |log T*| — the cost of a
given log-scale error depends on the confidence distribution's shape. None
of this refined form was pre-registered; a pre-registered replication on a
second dataset is the proper check.

**Diffusive norm growth lacks a theory.** E3 measured it: Adam at lr 1e-3
grows ‖W‖² 2.5× faster than SGD while moving 0.3% radially. The dependence of
the diffusion rate on learning rate, dimension, gradient noise, and the
optimizer's normalization is unmodeled here, as is the boundary in
objective-space (E2b vs E1) where transport switches from advection to
diffusion. Both invite analysis.

**Constraint experiments and learning-rate transfer.** E2b showed that
projecting the iterate changes the geometry the learning rate was tuned for;
the unconstrained-optimal rate manufactured a spurious optimizer gap until a
sweep corrected it. Our own E2 full-arm numbers carry the same caveat — its
under-confidence finding is robust across the arms but its exact floors were
not lr-retuned per arm.
