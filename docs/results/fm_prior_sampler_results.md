# FM-prior conditional sampling on L96 — moved

The measurements, derivations and diagnosed negative results that were drafted here
now live in the generated report

    reports/l96/outputs/l96_fm_sampler_benchmark.md

which is produced by `reports/l96/generate_l96_fm_sampler_report.py` from the JSON
written by `reports/l96/run_sda_sampler_experiments.py`. The sampler itself is
`evaluation/fm_sampler.py` (`Cold`, `Warm`, `Blend`, `Decoupled`), unit-tested in
`tests/test_fm_sampler.py`.

This file is kept as a pointer rather than deleted because commit 91a6fea references
it. Do not add results here: the report is generated from the runs, so anything
written by hand in a second place will drift away from the numbers.

Related:

* `docs/results/cfm_affine_velocity_decomposition.md` — the operator decomposition the
  schemes are built on (`Psi = mu_p + K_tau(x_tau - beta mu_p) + Psi_NG`).
* `docs/results/psi_decomposition_results.md` — its measurement.
