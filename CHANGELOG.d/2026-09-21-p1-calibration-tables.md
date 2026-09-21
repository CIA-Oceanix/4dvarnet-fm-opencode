## 2026-09-21: Add measured and planned calibration tables to P1 §5.5 -- and flag that C4 is unmeasured

**Summary:** §5.5 gains two tables: the blend family's measured accuracy /
dispersion trade, and an empty-celled plan for a single calibration protocol
across every class. Building the second surfaced a gap bigger than the missing
rank histograms.

**1. The measured trade was in the repo but not in the draft.**
`reports/l96/outputs/l96_fm_sampler_benchmark.md` carries a blend sweep at N=30
that the section described only in prose:

| configuration | RMSE | spread/skill |
|---|---|---|
| mean-only | 0.4934 | 0.000 |
| cold (sampler alone) | 0.5686 | 1.202 |
| warm start (tau0=0.3) | 0.4268 | 0.284 |
| blend (lambda0=0.5, p=1) | **0.4202** | **0.380** |

The sampler alone is the only configuration near the reliability target (0.967)
and the worst on RMSE. **The continuous blend dominates the hard warm start on
both axes at once** -- better RMSE and better calibration -- which the previous
prose did not say and which is the one place the trade is not strictly
adversarial. Flagged as not cross-quotable with `tab:price`: it comes from the
sampler study's own configuration (variance-weighted guidance, N_outer=50).

**2. The gap this exposed: C4 is argued from construction, not measurement.**
The measured table covers the generative family and contains **no
ensemble-Kalman row**. C4 -- the claim that the ensemble-Kalman class carries a
structural ceiling because `Psi_NG = 0` -- is a claim about exactly those
schemes. EnKF and ETKF are ensemble methods whose spread is directly measurable,
and it has never been measured in this study.

That `Psi_NG = 0` is a mathematical fact about the update, and the accuracy
consequence is visible throughout the overview. But the *calibration*
consequence -- the part that makes this symptom S-b rather than a restatement of
S-a -- is asserted. The draft now says so in the text, and
`tab:calibration-planned` lays out one protocol at uniform N=30 on the
`tab:price` configuration, so accuracy and dispersion come from the same runs.

The open question is stated precisely: does the ensemble-Kalman class fail in the
specific way the missing non-Gaussian branch predicts, or is it merely
under-dispersive in the ordinary way inflation exists to fix? Only the second
table can separate those, and until it is filled C4 is the paper's weakest claim.

**Verification:** `pdflatex` x3 -- 16 pages, 0 undefined references or citations;
every `\ref` checked against a matching `\label`, none dangling.
