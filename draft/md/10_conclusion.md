# 10. Conclusion

Every log-sum-exp objective contains a direction that reduces loss without
changing anything a practitioner would call learning. We made that direction
measurable (the radial/tangential decomposition and its estimator hierarchy),
watched it absorb 85% of late-training travel while accuracy stood still,
removed it and watched the loss floor, exhibited one objective where its
payout is a closed-form −2K log α and explains a previously unexplained
optimizer anomaly to within 9%, and showed that the same pathology travels by
two different mechanisms — advection under SGD, diffusion under Adam — with
the choice made by the objective's radial payout, not the optimizer alone.

The unification is the point. Weight magnification in calibration, logit-norm
growth in OOD detection, norm divergence in implicit-bias theory, and the
normalized spheres of metric learning are one observation: the mixture
likelihood that LSE silently approximates is missing its volume term, and
training spends the surplus on confidence. The fixes from all four
literatures price that term — and E4's U-shape shows they are a dial, not a
switch: a model can be charged too much for volume as easily as too little,
and miscalibration is the two-sided distance between the confidence scale a
model deploys and the one its accuracy has earned.

Three of twelve registered predictions failed, and each failure bought the
paper its sharpest result: the estimator hierarchy, the transport-mechanism
split, and the two-sided law. The volume term LSE forgot is measurable,
removable, and — within this study — sufficient to explain the anomalies it
created.
